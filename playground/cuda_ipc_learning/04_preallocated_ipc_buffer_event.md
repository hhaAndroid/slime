# 04 预分配 IPC Buffer 和 Event 同步复用

本文对应本轮学习主线里的第 4 个内容：在 FlattenedTensorBucket 基础上的进一步优化。

重点不是重新讲 flatten bucket，而是说明：相比 `03_flattened_tensor_bucket.md` 里“每个 bucket 构造一个新的 flattened tensor 并发送”的方式，xtuner + lmdeploy 的 pytorch / lmdeploy 共卡权重同步做了什么优化。

## 1. 03 文档里的基础方案

前一节的基础方案可以概括为：

```text
每次权重同步 bucket:
  named tensors
    -> FlattenedTensorBucket
    -> 新的 flattened_tensor
    -> serialized payload 携带 flattened_tensor + metadata
    -> consumer 反序列化并 reconstruct
```

这个方案已经减少了 tensor 数量和 CUDA IPC handle 数量：

```text
很多小 tensor
  -> 一个大 flattened tensor
```

但是它仍然有一个特点：

```text
每个 bucket 通常都会新建一个 flattened_tensor
每个 bucket 都要把这个 flattened_tensor 的 IPC 信息发给 consumer
consumer 每次都要重新构造 / 接收这个 IPC tensor
```

也就是说，它优化了“很多 tensor”这个问题，但没有继续优化“每个 bucket 都新建和传输 flattened tensor”这个问题。

## 1.1 为什么同一块 IPC Buffer 可以复用

这个优化能成立的关键原因是：CUDA IPC handle 描述的是一块 CUDA allocation，而不是 allocation 里的某一版具体内容。

可以把三类信息分开看：

```text
IPC handle / IPC mapping:
  指向同一块 GPU allocation
  只要 allocation 不变，它就不需要变

buffer content:
  这块 allocation 里的实际权重数据
  producer 每次同步都可以覆盖写入新内容

metadata:
  当前这批权重在 buffer 里的布局说明
  每个 bucket 都可能变化
```

所以第一次发送时：

```text
producer 分配 ipc_buffer
producer 发送 ipc_buffer 的 IPC 信息
consumer 打开 handle，得到 cached IPC tensor
```

后续同步时：

```text
producer 继续往同一块 ipc_buffer 写入新权重
consumer 继续使用之前 cached 的 IPC tensor
payload 只需要带新的 metadata
```

这和 CPU shared memory 的心智模型类似：

```text
第一次把 shared memory fd 发给对方
双方 mmap 同一块内存
后续 producer 直接覆盖这块内存内容
consumer 不需要重新 mmap，只需要知道这次数据如何解释
```

但这里有几个前提：

```text
同一块 CUDA allocation 没有被换掉
producer 进程仍然持有这块 allocation
consumer 之前打开的 IPC mapping 仍然有效
buffer 容量足够放下当前 bucket
dtype / metadata 解释方式符合双方协议
```

如果 producer 重新分配了新的 tensor，consumer 缓存的 IPC tensor 仍然指向旧 allocation。因此代码在这些情况下必须重新发送 `flattened_tensor`：

```text
第一次没有 cached buffer
buffer 不够大，需要 resize
dtype 发生切换
```

所以一句话总结：

```text
可以复用，是因为 IPC handle / mapping 绑定的是稳定的 GPU allocation；
只要 allocation 不变，handle 不需要变，变的只是 allocation 里的内容和本次 metadata。
```

## 2. 04 的核心优化

xtuner + lmdeploy 的优化目标是：

```text
不要每个 bucket 都创建和发送新的 flattened_tensor
而是预分配一块可复用的 CUDA IPC buffer
后续 bucket 只把新权重写进这块 buffer
consumer 复用之前打开过的 IPC tensor
```

简化对比：

```text
03 基础方案:
  bucket1 -> flattened_tensor_1 -> 发送 tensor handle + metadata
  bucket2 -> flattened_tensor_2 -> 发送 tensor handle + metadata
  bucket3 -> flattened_tensor_3 -> 发送 tensor handle + metadata

04 优化方案:
  第一次:
    分配 ipc_buffer -> 发送 buffer handle + metadata + event handle
  后续:
    复用 ipc_buffer -> 只发送 metadata
```

这里的关键变化是：

```text
flattened_tensor 从“一次性 bucket 产物”
变成“长期复用的 IPC 通信缓冲区”
```

## 3. Producer 侧缓存了什么

xtuner 里初始化了几类状态：

```python
self._default_ipc_tensor_bytes
self._ipc_tensor_bytes_dict_by_dtype
self._update_params_ipc_tensor_dict_by_dtype
self._last_update_params_ipc_tensor_dtype
self._update_params_ipc_event
```

源码位置：

```text
/mnt/shared-storage-user/huanghaian/code/temp/xtuner/xtuner/v1/rl/trainer/update_weighter.py
```

这些状态的含义：

```text
_default_ipc_tensor_bytes:
  默认预分配 buffer 大小

_ipc_tensor_bytes_dict_by_dtype:
  每种 dtype 当前 buffer 的字节容量

_update_params_ipc_tensor_dict_by_dtype:
  每种 dtype 对应的可复用 CUDA IPC buffer

_last_update_params_ipc_tensor_dtype:
  上一次使用的 dtype，用于判断 dtype 是否切换

_update_params_ipc_event:
  跨进程 CUDA event，用于同步 producer / consumer 对同一块 buffer 的读写
```

这里按 dtype 缓存，是因为 lmdeploy 的这个 FlattenedTensorBucket 要求同一个 bucket 里的 tensor dtype 相同。

## 4. 如何判断是否需要重新发送 IPC tensor

关键逻辑在 `_build_lmdeploy_flattened_tensor_data`：

```python
state_dict_dtype = state_dict[next(iter(state_dict))].dtype
update_params_ipc_tensor = self._update_params_ipc_tensor_dict_by_dtype.get(state_dict_dtype, None)
state_dict_bytes = self._compute_state_dict_bytes(state_dict)
ipc_tensor_bytes = self._ipc_tensor_bytes_dict_by_dtype.get(
    state_dict_dtype,
    self._default_ipc_tensor_bytes,
)
dtype_changed = (
    self._last_update_params_ipc_tensor_dtype is not None
    and state_dict_dtype != self._last_update_params_ipc_tensor_dtype
)
need_resize = state_dict_bytes > ipc_tensor_bytes
send_ipc_tensor = dtype_changed or need_resize or update_params_ipc_tensor is None
```

可以读成：

```text
如果是第一次，没有 cached ipc buffer:
  需要发送 flattened_tensor

如果 dtype 变了:
  需要发送 flattened_tensor

如果当前 bucket 比已有 buffer 大:
  需要重新分配更大的 buffer，并发送 flattened_tensor

否则:
  不需要发送 flattened_tensor，只发送 metadata
```

这是优化的核心判断。

## 5. 如何复用预分配 Buffer

创建 buffer 的逻辑：

```python
def _create_ipc_tensor(size_in_bytes: int, dtype: torch.dtype):
    return torch.empty(size_in_bytes, dtype=torch.uint8, device=DEVICE).view(dtype)
```

这里先按 byte 分配，再 view 成目标 dtype。

如果没有 buffer 或者需要扩容：

```python
ipc_tensor_bytes = max(ipc_tensor_bytes, state_dict_bytes)
self._ipc_tensor_bytes_dict_by_dtype[state_dict_dtype] = ipc_tensor_bytes
update_params_ipc_tensor = self._create_ipc_tensor(
    ipc_tensor_bytes,
    state_dict_dtype,
)
self._update_params_ipc_tensor_dict_by_dtype[state_dict_dtype] = update_params_ipc_tensor
```

然后把这个预分配 buffer 传给 lmdeploy 的 `FlattenedTensorBucket`：

```python
flattened_tensor_bucket = flattened_tensor_bucket_cls(
    named_tensors=list(state_dict.items()),
    flattened_tensor=update_params_ipc_tensor,
)
```

这和 03 里的基础版本不同。

03 基础版本通常是：

```text
FlattenedTensorBucket 自己 torch.cat 出一个新的 flattened_tensor
```

04 优化版本是：

```text
FlattenedTensorBucket 把当前 bucket 内容写入已有 update_params_ipc_tensor
```

lmdeploy 里的实现是：

```python
torch.cat(flattened_tensor_list, dim=0, out=flattened_tensor[:current_idx])
self.flattened_tensor = flattened_tensor
```

所以它没有返回一个新的 buffer，而是复用调用者传进来的 buffer。

## 6. Payload 结构如何变化

03 基础方案里，每次 payload 都像：

```python
{
    "flattened_tensor": flattened_tensor,
    "metadata": metadata,
}
```

04 优化方案里，payload 分两类。

第一次 / dtype 切换 / resize 时：

```python
{
    "metadata": metadata,
    "require_clone": False,
    "flattened_tensor": cached_ipc_buffer,
    "event_ipc_handle": event_handle,
}
```

后续复用同一块 buffer 时：

```python
{
    "metadata": metadata,
    "require_clone": False,
}
```

也就是说：

```text
后续 payload 不再携带 flattened_tensor
consumer 默认继续使用之前缓存的 IPC tensor
metadata 描述当前 buffer 前面哪一段代表哪些权重
```

这就是相比 03 的主要优化：减少 repeated flattened tensor IPC handle 传输和 consumer 端重建。

## 7. Consumer 侧如何缓存 IPC Tensor

lmdeploy 接收侧在 `update_params` 里处理：

```text
/mnt/shared-storage-user/huanghaian/code/lmdeploy/lmdeploy/pytorch/engine/model_agent/agent.py
```

如果 payload 里有 `flattened_tensor`：

```python
self._update_params_ipc_tensor = _construct(
    weights["flattened_tensor"],
    require_clone=require_clone,
)
```

如果 payload 里没有 `flattened_tensor`：

```python
elif self._update_params_ipc_tensor is None:
    raise ValueError(...)
```

含义是：

```text
第一次必须收到 flattened_tensor，用它打开并缓存 IPC tensor
后续可以不再收到 flattened_tensor，直接复用 self._update_params_ipc_tensor
```

每次真正 reconstruct 时：

```python
flattened_tensor = self._update_params_ipc_tensor # 核心是这个不需要每次都重新调用 _construct 而是复用之前的
bucket = FlattenedTensorBucket(
    flattened_tensor=flattened_tensor,
    metadata=metadata, # 这个每次都需要传，因为要区分是啥参数
)
return list(bucket.reconstruct_tensors())
```

所以 consumer 侧也从：

```text
每次从 payload 得到新的 flattened_tensor
```

变成：

```text
缓存一个长期 IPC tensor
每次只用新的 metadata 解释这块 buffer
```

## 8. 为什么需要 Event 同步

一旦复用同一块 buffer，就引入一个新问题：

```text
producer 可能在 consumer 还没读完时覆盖 buffer
consumer 可能在 producer 还没写完时读取 buffer
```

03 基础方案里，每个 bucket 通常有自己的 flattened tensor，生命周期相对独立。

04 优化方案里，多次 bucket 共用同一个 buffer，因此必须有明确同步协议。

xtuner 创建 interprocess CUDA event：

```python
self._update_params_ipc_event = DEVICE_MODULE.Event(interprocess=True)
```

producer 写完 buffer 后：

```python
self._update_params_ipc_event.record()
```

如果需要首次发送或重新发送 buffer，还会把 event handle 放进 payload：

```python
flattened_tensor_data["event_ipc_handle"] = self._update_params_ipc_event.ipc_handle()
```

consumer 侧打开 event：

```python
self._update_params_ipc_event = torch.cuda.Event.from_ipc_handle(
    device=torch.cuda.current_device(),
    handle=weights["event_ipc_handle"],
)
```

然后读取 buffer 前等待 producer 写完：

```python
self._update_params_ipc_event.wait()
```

consumer load 完权重后，再 record：

```python
self._update_params_ipc_event.record()
```

producer 下一次复用 buffer 前等待：

```python
self._update_params_ipc_event.wait()
```

所以这个 event 被用作一个简单的 producer / consumer 交接信号。

## 9. 复用协议

可以把整个协议理解成：

```text
第一次:
  producer 分配 ipc_buffer
  producer 把 bucket 内容写入 ipc_buffer
  producer record event
  producer 发送 ipc_buffer handle + event handle + metadata

  consumer 打开 ipc_buffer
  consumer 打开 event
  consumer wait event
  consumer reconstruct tensors
  consumer load_weights
  consumer record event

后续:
  producer wait event，确认 consumer 已完成上一批读取
  producer 覆盖 ipc_buffer，写入下一批 bucket
  producer record event
  producer 只发送 metadata

  consumer 复用 cached ipc_buffer
  consumer wait event
  consumer 用新 metadata reconstruct
  consumer load_weights
  consumer record event
```

这就是“单 buffer 复用”的核心流程。

## 10. require_clone=False 的含义

payload 里会带：

```python
"require_clone": False
```

结合 lmdeploy 的 `_construct`：

```python
ipc_tensor = func(*args)
return ipc_tensor.clone() if require_clone else ipc_tensor
```

`require_clone=False` 表示 consumer 不把 IPC tensor clone 成自己的独立 tensor，而是直接缓存并复用这块 IPC tensor。

这样才能实现：

```text
后续 producer 写同一块 buffer
consumer 通过 cached IPC tensor 看到新内容
```

如果 clone 了，consumer 缓存的是自己的独立副本，后续 producer 覆盖原始 buffer，consumer 的 clone 不会自动变化，这个复用优化就失效了。

所以：

```text
require_clone=False 是 buffer 复用成立的必要条件
event 同步是 require_clone=False 安全使用的必要条件
```

## 11. 这个优化的代价和限制

相比 03 基础方案，这个优化更高效，但协议也更复杂。

主要限制：

```text
1. 同一个 bucket 要求 dtype 一致
   lmdeploy 的 FlattenedTensorBucket 不使用 uint8 byte view 混合 dtype，
   而是按原 dtype flatten，所以一个 bucket 里 dtype 必须相同。

2. 需要 producer / consumer 都支持 CUDA event IPC
   否则无法可靠同步复用 buffer。

3. 生命周期更敏感
   producer 侧 cached buffer 必须长期存活，
   consumer 侧 cached IPC tensor 也必须维护好。

4. 需要 resize / dtype change 协议
   buffer 不够大或 dtype 切换时，必须重新发送 flattened_tensor。

5. 对同步协议要求更高
   如果 barrier / event 使用不当，producer 可能覆盖 consumer 尚未读取的数据。
```

xtuner 里还有一个 barrier 注释专门说明这个风险：

```text
Without barrier, some ranks ... would write next iter state_dict into the ipc tensor before lmdeploy load current iter weight.
```

也就是说，这个优化的性能来自复用同一块内存；风险也来自复用同一块内存。

## 12. 和 03 方案的最终对比


| 维度                    | 03 基础 FlattenedTensorBucket      | 04 预分配 IPC Buffer                                |
| --------------------- | -------------------------------- | ------------------------------------------------ |
| flattened tensor 生命周期 | 每个 bucket 通常新建                   | 长期缓存并复用                                          |
| payload 是否携带 tensor   | 每次携带                             | 首次 / resize / dtype change 才携带                   |
| 后续 payload            | tensor + metadata                | metadata                                         |
| consumer 行为           | 每次接收新的 IPC tensor                | 缓存并复用 IPC tensor                                 |
| 同步复杂度                 | 相对简单                             | 需要 interprocess CUDA event                       |
| 是否允许混合 dtype          | SGLang 版本用 uint8 view，可支持多 dtype | lmdeploy 版本要求 bucket 内 dtype 一致                  |
| 主要收益                  | 减少小 tensor 数量                    | 进一步减少 repeated allocation / IPC handle / rebuild |
| 主要风险                  | producer tensor 生命周期             | buffer 被过早覆盖                                     |


## 13. 本节需要掌握的结论

- 03 的核心是把多个 tensor 合成一个 flattened tensor。
- 04 的核心是把 flattened tensor 进一步变成可复用的 IPC buffer。
- 第一次或 buffer 变化时才发送 `flattened_tensor`；后续只发送 metadata。
- consumer 缓存 `self._update_params_ipc_tensor`，后续用新 metadata 解释同一块 buffer。
- `require_clone=False` 保证 consumer 缓存的是共享 IPC tensor，而不是独立副本。
- interprocess CUDA event 负责 producer / consumer 对同一块 buffer 的读写交接。
- 这个优化减少 repeated allocation、IPC handle 传输和 rebuild 开销，但需要更严格的生命周期和同步协议。
