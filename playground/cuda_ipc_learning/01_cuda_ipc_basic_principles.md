# 01 CUDA IPC 基本原理

本文对应学习计划里的 Item 1：CUDA IPC 基本模型。

本节先讲原理，再看最小代码。核心目标不是记 API，而是建立一个准确的心智模型：CUDA IPC 到底共享了什么、没有共享什么，以及为什么 slime + SGLang 的 colocated 权重同步能利用它避免 GPU -> CPU -> GPU 的数据拷贝。

## 1. 最小心智模型

CUDA IPC 解决的是同一台机器上不同进程之间共享 CUDA 显存的问题。可以跨卡，但是不能跨机。

**1 为什么能跨卡**

同一台机器上的多个 GPU 都由本机 CUDA driver 管理。producer 在 GPU0 上创建 allocation 后，CUDA driver 能在另一个本机进程里打开这个 allocation，并返回一个 device pointer。consumer 如果运行在 GPU1 上，能否直接访问

GPU0 那块显存，取决于本机 GPU 拓扑和 P2P 能力：同机 GPU0 allocation

```
-> CUDA IPC handle

-> 同机另一个进程打开

-> 如果 GPU1 能 peer access GPU0，则 GPU1 kernel 可直接访问
```

跨卡访问底层依赖的是同机 GPU 间互联和地址映射能力，例如 PCIe P2P、NVLink、NVSwitch，以及 CUDA peer access / UVA 相关机制。CUDA IPC 负责“跨进程打开 allocation”，P2P/UVA 负责“另一个 GPU 是否能寻址和访问这块显存”。

所以 CUDA IPC 本身不是把数据搬到另一张卡，它只是让另一个进程拿到访问这块 allocation 的能力。

**2 为什么不能跨机**

跨机时，另一台机器的 CUDA driver 管不到本机 GPU 的物理显存 allocation。CUDA IPC handle 是本机 driver 生成的不透明 token，它只在同一个 CUDA driver/OS 实例管理的资源空间里有意义。

普通跨进程传输如果要传一个 GPU tensor，直觉路径可能是：

```text
producer GPU tensor -> 拷到 CPU -> 通过 IPC 传 CPU bytes -> consumer 再拷到 GPU
```

CUDA IPC 的路径不是这样。它更接近：

```text
producer 在 GPU 上创建 allocation
producer 导出 CUDA IPC handle
consumer 通过普通进程通信拿到 handle
consumer 调 CUDA runtime 打开 handle
consumer 得到一个能访问同一块 GPU allocation 的 device pointer
```

所以共享的是底层 GPU memory allocation，不是 Python tensor 对象本身。

## 2. Producer、Handle、Consumer

可以把 CUDA IPC 看成三个角色：

```text
producer 进程
  持有原始 CUDA allocation
  可以导出 IPC handle

CUDA IPC handle
  一个很小的不透明 token
  能通过 Queue / socket / Ray 等普通 IPC 机制传给别的进程

consumer 进程
  收到 handle
  调 CUDA runtime 打开 handle
  在自己的进程里得到一个 device pointer
  用这个 pointer 访问 producer 那块 GPU allocation
```

重要结论：

- producer 和 consumer 各自有自己的 Python tensor 对象。
- 两个 Python tensor 对象可以指向同一个底层 GPU allocation。
- consumer 修改 tensor 内容后，producer 侧能看到同一块 GPU memory 的变化。

## 3. CUDA IPC Handle 是什么

CUDA IPC handle 可以理解成 CUDA driver 能识别的跨进程显存访问凭证。

它不是：

```text
tensor 数据
CPU buffer
GPU pointer
文件路径
普通 Python 对象
```

它更像：

```text
“请 CUDA driver 在另一个进程里打开 producer 进程中的这块 GPU allocation”
```

CUDA Runtime API 里的使用方式大致是：

```c
cudaIpcMemHandle_t handle;
cudaIpcGetMemHandle(&handle, device_ptr);

void* remote_ptr;
cudaIpcOpenMemHandle(&remote_ptr, handle, flags);
```

`cudaIpcMemHandle_t` 是 opaque struct。opaque 的意思是：用户可以整体传递它，但不能解析它的内部字段，也不应该依赖其内部布局。

从功能上推断，handle 内部必须让 CUDA driver 能找到原始 allocation，例如它可能关联：

```text
GPU / device 身份
allocation 的 driver 内部标识
跨进程映射需要的 driver metadata
访问权限相关信息
```

但这些都只是功能意义上的理解，不是公开 ABI。

## 4. IPC Handle 多大

CUDA memory IPC handle 本体通常是固定大小。

CUDA Runtime API 里常见定义可以理解成：

```c
#define CUDA_IPC_HANDLE_SIZE 64

typedef struct cudaIpcMemHandle_st {
    char reserved[CUDA_IPC_HANDLE_SIZE];
} cudaIpcMemHandle_t;
```

也就是说，CUDA IPC memory handle 本体通常是 64 bytes。

但要区分：

```text
CUDA IPC handle 本体
  约 64 bytes
  只负责打开 GPU allocation，但是不知道类型，shape 等，可以理解为只提供最基础功能。要重建完整 tensor，pytorch 帮我们做了很多事情

完整 tensor 跨进程 payload
  CUDA IPC handle
  tensor dtype / shape / stride / offset 等 metadata
  Python / PyTorch 序列化包装
  可能还有 name、bucket metadata 等上层信息
```

slime + SGLang 里传的不是裸 64 bytes handle，而是 PyTorch / SGLang serializer 包装过的对象。

## 5. 打开 Handle 是否需要 Shape 和 Dtype

只打开 CUDA IPC memory handle，不需要知道 shape、dtype、stride。

CUDA 层只做这件事：

```text
handle -> raw device pointer
```

所以 API 返回的是类似 `void*` 的 device pointer。CUDA runtime 不知道这块显存应该被解释成：

```text
float32[4]
bf16[4096, 4096]
int8[1024]
某个模型参数
```

如果要把打开后的 raw device pointer 正确变成 tensor，就必须额外知道：

```text
dtype
shape
stride
storage offset
numel / nbytes
tensor name
```

因此应该分两层理解：

```text
CUDA IPC open:
  handle -> raw device pointer

Tensor reconstruction:
  raw device pointer + dtype + shape + stride / offset -> tensor view
```

在 PyTorch 里，通过 multiprocessing 发送 CUDA tensor 时，这些细节被封装了。PyTorch 序列化 payload 里不只包含 CUDA IPC handle，还包含足够的 tensor metadata，用于在 consumer 进程里重建 `torch.Tensor`。

## 6. 它不是 Pointer

CUDA IPC handle 不是 producer 进程里的 `data_ptr()`。

producer 里某个 tensor 的 pointer 可能是：

```text
0x7f0000000000
```

consumer 打开同一个 handle 后得到的 pointer 可能是：

```text
0x7e8000000000
```

两个地址不同是正常的，因为两个进程有不同的虚拟地址空间。

关键不是两个 `data_ptr()` 数值是否相同，而是它们是否映射到底层同一个 GPU allocation。我们在最小 demo 里看到 consumer 修改值后 producer 也看到变化，就证明二者共享了同一块底层 GPU memory。

## 7. 从 cuda:0 传到 cuda:1 会怎样

假设 producer 在 `cuda:0` 上创建 tensor：

```text
tensor.device == cuda:0
storage physically on GPU0
```

consumer 进程即使当前 device 是 `cuda:1`，通过 CUDA IPC 打开的仍然是 GPU0 上的那块 allocation。它不会自动把数据迁移到 GPU1。

通常应理解为：

```text
producer:
  tensor.device == cuda:0
  数据物理上在 GPU0

consumer current_device:
  cuda:1

consumer received tensor:
  tensor.device 通常仍是 cuda:0
  数据仍在 GPU0
```

如果 consumer 想在 GPU1 上得到一份本地数据，需要显式复制：

```python
y = x.to("cuda:1")
```

如果不复制而直接从 GPU1 访问 GPU0 memory，则依赖 GPU 间 peer access / P2P 能力，并且操作和性能都要谨慎看待。

这对 slime + SGLang 很重要：colocated 权重同步要求训练 rank 和 SGLang TP worker 的 GPU 映射关系正确。否则可能出现 device mismatch、意外 P2P 访问、性能下降，甚至直接失败。

## 8. 生命周期要求

CUDA IPC 的一个关键约束是：producer 侧原始 allocation 必须在 consumer 使用期间保持有效。

如果 producer 太早释放 tensor / storage，consumer 手里的 handle 可能指向已经无效或被复用的 GPU memory。

所以在 slime 的 colocated sender 里能看到类似设计：

```text
long_live_tensors.append(flattened_tensor_data)
```

这些对象不能在 SGLang 还没完成反序列化和加载权重之前被释放。等 SGLang 的 Ray 调用返回后，slime 才能删除引用并调用：

```python
torch.cuda.ipc_collect()
```

`ipc_collect()` 的作用不是同步 tensor 值，而是帮助 PyTorch 清理 CUDA IPC 相关缓存和已经不再被 consumer 使用的 IPC 资源。

## 9. 同步语义

CUDA IPC handle 只解决“能不能访问同一块 memory”的问题，不自动解决“什么时候写完、什么时候读”的问题。

如果 producer 写 tensor、consumer 立刻读，就需要保证写操作已经完成。常见方式包括：

```text
CUDA stream 同步
CUDA event
torch.cuda.synchronize()
上层协议保证先后顺序
```

在最小 demo 中，我们用 `torch.cuda.synchronize()` 和进程间 event 保证顺序。

在真实 slime + SGLang 权重同步中，上层还会配合：

```text
pause_generation
flush_cache
Ray 调用返回
barrier / 同步点
```

但本轮学习重点不是完整调度，而是 CUDA IPC tensor path 本身。

这里有一个容易误解的点：既然 tensor 要先通过 IPC 传给 consumer，为什么还会出现“producer 写 tensor、consumer 立刻读”的风险？

原因是 IPC 传递的是“访问这块显存的能力”，不是“某一时刻已经完成的 tensor 数据快照”。CUDA 操作通常是异步的，producer 里的写入操作可能只是被 enqueue 到 CUDA stream 上，CPU 线程并不会默认等待它真正执行完。

例如 producer 侧：

```python
x = torch.empty(4, device="cuda")
x.fill_(1.0)      # 可能只是 enqueue 一个 CUDA kernel
queue.put(x)      # CPU 线程可以立刻把 IPC payload 发给 consumer
```

consumer 侧：

```python
x = queue.get()
print(x.cpu())    # 在 consumer 进程 / stream 上 enqueue 读取。可能这边读到的数据是 empty，也可能是 1
```

如果 producer 的 `fill_` 和 consumer 的读取不在同一个 stream 依赖链里，consumer 不一定自动等待 producer 的写入完成。风险不是 handle 传早了，而是两个进程里的 CUDA stream 之间没有天然的先后依赖。

所以更准确的顺序要求是：

```text
producer enqueue 写入
producer 等写入完成，或传递可等待的同步信号
consumer 打开 IPC handle
consumer 在确认写入完成后读取
```

最简单但较粗的方式是：

```python
torch.cuda.synchronize()
queue.put(x)
```

更细粒度的方式是使用 CUDA event，让 consumer wait producer 记录的 event 后再读。

## 10. 对应到 slime + SGLang

在本轮关注的 colocated 权重同步路径里，可以这样对应：

```text
slime / Megatron 进程
  持有训练后的 CUDA 权重 tensor
  把多个权重 flatten 成 bucket
  用 SGLang MultiprocessingSerializer 序列化 payload
  通过 Ray actor 调 SGLang update_weights_from_tensor

SGLang rollout 进程
  反序列化 payload
  PyTorch 打开 CUDA IPC handle
  得到 CUDA tensor
  根据 metadata reconstruct named tensors
  load 到模型权重中
```

核心分层是：

```text
CUDA IPC handle
  负责跨进程打开 GPU allocation

tensor metadata
  负责把 raw storage 解释成 tensor

weight metadata
  负责知道这个 tensor 应该加载到哪个模型参数
```

## 11. 本节最小代码

对应脚本：

```text
01_basic_cuda_ipc.py
```

这个脚本验证：

```text
producer 创建 CUDA tensor: [0, 1, 2, 3]
consumer 收到 tensor 后原地 add_(10)
producer 再读取 tensor，看到 [10, 11, 12, 13]
```

这个现象说明两个进程访问的是同一块底层 GPU allocation，而不是普通的 CPU bytes 拷贝。

同时，脚本会打印 producer 和 consumer 中的 `data_ptr()`。它们可能不同，这不影响共享语义。

## 12. 本节需要掌握的结论

- CUDA IPC handle 是不透明 token，不是 tensor 数据，也不是 pointer。
- CUDA IPC memory handle 本体通常是固定 64 bytes。
- 打开 handle 不需要 shape / dtype；重建 tensor 才需要 metadata。
- 共享的是底层 GPU allocation，不是 Python tensor 对象。
- 不同进程里的 `data_ptr()` 可以不同。
- 从 `cuda:0` 共享出去的数据仍然物理位于 GPU0，不会自动变成 GPU1 数据。
- producer 必须保持原始 allocation 存活，直到 consumer 完成使用。
- `torch.cuda.ipc_collect()` 用于清理 IPC 相关缓存，不是用来同步数据值。

