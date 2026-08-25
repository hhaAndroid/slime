"""Probe Ray placement-group resource behavior.

Run:
    python hha_code/ray_pg_resource_probe.py

What this checks:
1. Binding an actor to a bundle that contains GPU does not by itself grant GPU ids.
2. An actor must request GPU for ray.get_gpu_ids() to be non-empty.
3. A zero-resource actor can colocate with a GPU actor in the same bundle.
4. A second GPU actor should stay pending if the bundle's only GPU is already held.
"""

from __future__ import annotations

import argparse
import os
import time

import ray
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote
class PlainActor:
    def info(self):
        return {
            "kind": "plain_default_resources",
            "gpu_ids": ray.get_gpu_ids(),
            "node": ray.util.get_node_ip_address(),
            "pid": os.getpid(),
        }


@ray.remote(num_cpus=0)
class ZeroResourceActor:
    def info(self):
        return {
            "kind": "zero_resource",
            "gpu_ids": ray.get_gpu_ids(),
            "node": ray.util.get_node_ip_address(),
            "pid": os.getpid(),
        }


@ray.remote(num_gpus=1, num_cpus=0)
class GPUActor:
    def info(self):
        return {
            "kind": "gpu_1_cpu_0",
            "gpu_ids": ray.get_gpu_ids(),
            "node": ray.util.get_node_ip_address(),
            "pid": os.getpid(),
        }


def wait_value(ref, timeout_s: float):
    ready, _ = ray.wait([ref], timeout=timeout_s)
    if not ready:
        return "PENDING"
    return ray.get(ready[0])


def print_resources(title: str):
    print(f"\n[{title}]")
    print("cluster_resources:", ray.cluster_resources())
    print("available_resources:", ray.available_resources())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-cpus", type=int, default=4)
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()

    ray.init(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        include_dashboard=False,
        ignore_reinit_error=True,
        log_to_driver=False,
    )

    pg = None
    try:
        print_resources("after ray.init")

        pg = placement_group([{"CPU": 1, "GPU": 1}], strategy="PACK")
        print("waiting for placement group...")
        ray.get(pg.ready())
        print_resources("after pg.ready")

        strategy = PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=0,
        )

        print("\nCASE 1: PlainActor bound to GPU-containing bundle, but does not request GPU")
        plain = PlainActor.options(scheduling_strategy=strategy).remote()
        print("PlainActor.info:", wait_value(plain.info.remote(), args.timeout))
        print_resources("after PlainActor")

        print("\nCASE 2: GPUActor in same bundle while PlainActor is alive")
        print("If PlainActor did not consume GPU, GPUActor should run.")
        gpu = GPUActor.options(scheduling_strategy=strategy).remote()
        print("GPUActor.info:", wait_value(gpu.info.remote(), args.timeout))
        print_resources("after first GPUActor")

        print("\nCASE 3: second GPUActor in same bundle while first GPUActor is alive")
        print("Bundle has only one GPU, so this should usually stay PENDING.")
        second_gpu = GPUActor.options(scheduling_strategy=strategy).remote()
        print("SecondGPUActor.info:", wait_value(second_gpu.info.remote(), args.timeout))
        print_resources("after second GPUActor attempt")

        print("\nCASE 4: ZeroResourceActor in same bundle while GPUActor is alive")
        print("This should run because it requests num_cpus=0 and num_gpus=0.")
        zero = ZeroResourceActor.options(scheduling_strategy=strategy).remote()
        print("ZeroResourceActor.info:", wait_value(zero.info.remote(), args.timeout))
        print_resources("after ZeroResourceActor")

        print("\nCleaning actors...")
        for actor in [plain, gpu, second_gpu, zero]:
            try:
                ray.kill(actor, no_restart=True)
            except Exception as exc:
                print(f"ray.kill ignored: {exc!r}")
        time.sleep(1)
        print_resources("after actor cleanup")

    finally:
        if pg is not None:
            remove_placement_group(pg)
        ray.shutdown()


if __name__ == "__main__":
    main()
