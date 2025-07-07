# 共卡

import socket
import ray
from ray.util.placement_group import placement_group
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import os
import torch.nn as nn
import torch.distributed as dist
import torch


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, gpu_id)


def _create_placement_group(num_gpus):
    """Create a placement group with the specified number of GPUs."""
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    ray.get(pg.ready())
    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                )
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    pg_reordered_bundle_indices = [bundle_info[0] for bundle_info in sorted(bundle_infos, key=sort_key)]
    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        print(
            f"  bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


@ray.remote(
    num_gpus=1,
    runtime_env={
        "env_vars": {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
        }
    },
)
class TrainRayActor:
    def __init__(self, world_size, rank, master_addr, master_port):
        self._world_size = world_size
        self._rank = rank
        self.master_addr, self.master_port = master_addr, master_port

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])

        # 初始化分布式
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(f"cuda:{local_rank}")
        dist.init_process_group(backend='nccl')

        self.model = SimpleModel()

    def wake_up(self):
        self.model = self.model.cuda()

    def sleep(self):
        # 清空模型的显存
        self.model = self.model.cpu()
        torch.cuda.empty_cache()

    def forward(self, data):
        data = data.cuda()
        output = self.model(data)
        return output


class ActorGroup:
    def __init__(
            self,
            num_nodes,
            num_gpus_per_node,
            pg: tuple[PlacementGroup, list[int]],
            num_gpus_per_actor=1
    ) -> None:
        self._num_nodes = num_nodes
        self._num_gpus_per_node = num_gpus_per_node

        # Allocate the GPUs for actors w/o instantiating them
        self._allocate_gpus_for_actor(pg, num_gpus_per_actor)

    def _allocate_gpus_for_actor(self, pg, num_gpus_per_actor):
        world_size = self._num_nodes * self._num_gpus_per_node

        # Use placement group to lock resources for models of same type
        assert pg is not None
        pg, reordered_bundle_indices = pg
        # Create worker actors
        self._actor_handlers = []
        master_addr, master_port = None, None
        for rank in range(world_size):
            actor = TrainRayActor.options(
                num_cpus=num_gpus_per_actor,
                num_gpus=num_gpus_per_actor,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=reordered_bundle_indices[rank],
                ),
            ).remote(world_size, rank, master_addr, master_port)
            if rank == 0:
                master_addr, master_port = ray.get(actor.get_master_addr_and_port.remote())
            self._actor_handlers.append(actor)

    def forward(self, data):
        # 均匀切分给每个 worker actor
        num_actors = len(self._actor_handlers)
        data_split = torch.chunk(data, num_actors)
        # 运行后收集结果
        futures = [actor.forward.remote(data_split[i]) for i, actor in enumerate(self._actor_handlers)]
        results = ray.get(futures)
        # 合并结果
        return torch.cat(results, dim=0)

    def sleep(self):
        # 调用每个 worker 的 wake up 方法
        futures = [actor.sleep.remote() for actor in self._actor_handlers]
        ray.get(futures)

    def wake_up(self):
        # 调用每个 worker 的 wake up 方法
        futures = [actor.wake_up.remote() for actor in self._actor_handlers]
        ray.get(futures)


def train():
    # allocate the GPUs
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8

    num_gpus = actor_num_nodes * actor_num_gpus_per_node

    print(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices = _create_placement_group(num_gpus)
    train_group = ActorGroup(
        num_nodes=actor_num_nodes,
        num_gpus_per_node=actor_num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.6  # 假设一共有 2 个 actor 要共卡，只要不能全占就行
    )

    eval_group = ActorGroup(
        num_nodes=actor_num_nodes,
        num_gpus_per_node=actor_num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=0.4  # 假设一共有 2 个 actor 要共卡，只要不能全占就行
    )

    for i in range(10):
        print(f"current step {i+1}")

        # 先简单训练一次
        data = torch.randn(64, 10)  # 假设输入数据

        eval_group.sleep()
        train_group.wake_up()

        output = train_group.forward(data)
        print("train output shape:", output.shape)

        # 释放显存
        train_group.sleep()
        eval_group.wake_up()

        eval_output = eval_group.forward(data)
        print("eval output shape:", eval_output.shape)
