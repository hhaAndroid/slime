# 共卡

import socket
from ray.util.placement_group import placement_group
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
import os
import torch.nn as nn
import torch.distributed as dist
import torch
import ray
from torch.distributed._composable.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.device_mesh import init_device_mesh


@ray.remote(num_gpus=1)  # 关键参数，让每个 actor 只能看到一张卡
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
    # InfoActor 只为了获取 gpu_ids 信息用
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    # 重新排序 bundle_infos
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


def is_port_available(port):
    """Return whether a port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            return True
        except socket.error:
            return False
        except OverflowError:
            return False


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
        # 如果是 None 说明肯定是 rank0 节点，此时直接获取当前节点的 IP 和 free 端口
        if master_addr:
            self.master_addr, self.master_port = master_addr, master_port
        else:
            self.master_addr, self.master_port = self._get_current_node_ip_and_free_port(start_port=20000)

        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self._world_size)
        os.environ["RANK"] = str(self._rank)
        # TODO: currently this doesn't work as ray has already set torch.cuda.device_count().
        # os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])
        os.environ["LOCAL_RANK"] = str(ray.get_gpu_ids()[0])

        # 不能在下面初始化分布式，否则会卡死

    @staticmethod
    def _get_current_node_ip_and_free_port(start_port=10000, consecutive=1):
        address = ray._private.services.get_node_ip_address()  # 获取当前节点 ip 地址
        # strip ipv6 address
        address = address.strip("[]")

        # find the port where port, port + 1, port + 2, ... port + consecutive - 1 are all available
        port = start_port
        while not all(is_port_available(port + i) for i in range(consecutive)):
            port += 1

        return address, port

    def get_master_addr_and_port(self):
        return self.master_addr, self.master_port

    # 必须要单独调用，如果放到 __init__ 里面会因为 dist.init_process_group 卡住
    def init(self):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(f"cuda:{local_rank}")
        dist.init_process_group(backend='nccl')

        # 构建 fsdp 模型
        self.model = SimpleModel()

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        )
        world_size = dist.get_world_size()

        fsdp_mesh = init_device_mesh('cuda', (world_size,))

        fully_shard(
            self.model,
            mesh=fsdp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True
        )

    # 假装操作
    def wake_up(self):
        self.model = self.model.cuda()

    def sleep(self):
        # 清空模型的显存
        self.model = self.model.cpu()
        torch.cuda.empty_cache()

    def forward(self, data):
        data = data.cuda()
        output = self.model(data)
        return output.cpu()


# 管理所有 actor
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

    def init(self):
        # 初始化每个 worker 的模型
        futures = [actor.init.remote() for actor in self._actor_handlers]
        ray.get(futures)


def train():
    if not ray.is_initialized():
        ray.init()

    # allocate the GPUs
    actor_num_nodes = 1
    actor_num_gpus_per_node = 8

    num_gpus = actor_num_nodes * actor_num_gpus_per_node

    print(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices = _create_placement_group(num_gpus)

    print('init train_group')
    train_group = ActorGroup(
        num_nodes=actor_num_nodes,
        num_gpus_per_node=actor_num_gpus_per_node,
        pg=(pg, actor_pg_reordered_bundle_indices),
        num_gpus_per_actor=0.1  # 假设一共有 2 个 actor 要共卡，只要不能全占就行,否则后续会因为资源不够而卡死
    )
    train_group.init()

    print('init eval_group')
    eval_group = ActorGroup(
        num_nodes=actor_num_nodes,
        num_gpus_per_node=actor_num_gpus_per_node,
        pg=(pg, actor_pg_reordered_bundle_indices),
        num_gpus_per_actor=0.4  # 假设一共有 2 个 actor 要共卡，只要不能全占就行
    )
    eval_group.init()

    for i in range(10):
        print(f"current step {i + 1}")

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


# 单节点直接 python demo_1.py 即可
if __name__ == '__main__':
    train()
    ray.shutdown()
