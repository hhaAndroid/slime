#!/usr/bin/env python3
"""
Item 1: CUDA IPC 基本模型。

运行方式：
    python playground/cuda_ipc_learning/01_basic_cuda_ipc.py

这个脚本演示：
1. producer 进程创建 CUDA tensor。
2. consumer 进程通过 torch.multiprocessing.Queue 收到这个 tensor。
3. consumer 原地修改 tensor。
4. producer 在自己的进程里观察修改是否可见。

如果 producer 侧能看到 consumer 的修改，就说明跨进程传递的不是普通的
CPU bytes 拷贝，而是 PyTorch 借助 CUDA IPC 打开的共享 GPU storage。
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.multiprocessing as mp


def consumer_main(queue: mp.Queue, done: mp.Event, device: int, add_value: float) -> None:
    torch.cuda.set_device(device)

    tensor = queue.get()
    print(
        f"[consumer pid={os.getpid()}] received: "
        f"device={tensor.device}, shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"data_ptr={tensor.data_ptr()}, values={tensor.cpu().tolist()}",
        flush=True,
    )

    tensor.add_(add_value)
    torch.cuda.synchronize(device)

    print(
        f"[consumer pid={os.getpid()}] after add_({add_value}): "
        f"values={tensor.cpu().tolist()}",
        flush=True,
    )
    done.set()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal CUDA IPC tensor sharing demo")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--size", type=int, default=4, help="Number of elements in the demo tensor")
    parser.add_argument("--add-value", type=float, default=10.0, help="Value added by the consumer process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This demo requires a CUDA GPU.")
    if args.device >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device {args.device} does not exist. device_count={torch.cuda.device_count()}")

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    done = ctx.Event()

    torch.cuda.set_device(args.device)
    tensor = torch.arange(args.size, dtype=torch.float32, device=f"cuda:{args.device}")
    torch.cuda.synchronize(args.device)

    print(
        f"[producer pid={os.getpid()}] before send: "
        f"device={tensor.device}, shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"data_ptr={tensor.data_ptr()}, values={tensor.cpu().tolist()}",
        flush=True,
    )

    consumer = ctx.Process(target=consumer_main, args=(queue, done, args.device, args.add_value))
    consumer.start()

    queue.put(tensor)

    # producer 必须保持 tensor 存活，直到 consumer 用完这个 CUDA IPC handle。
    done.wait()
    torch.cuda.synchronize(args.device)

    print(
        f"[producer pid={os.getpid()}] after consumer modification: "
        f"data_ptr={tensor.data_ptr()}, values={tensor.cpu().tolist()}",
        flush=True,
    )

    consumer.join()
    if consumer.exitcode != 0:
        raise RuntimeError(f"consumer exited with code {consumer.exitcode}")

    torch.cuda.ipc_collect()
    print("[producer] torch.cuda.ipc_collect() done", flush=True)


if __name__ == "__main__":
    main()
