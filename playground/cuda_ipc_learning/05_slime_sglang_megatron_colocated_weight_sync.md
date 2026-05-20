# 05 slime + SGLang 共卡 IPC 权重同步主流程

本文只关注 slime + SGLang 共卡场景下的 **IPC 权重同步细节**。

不展开：

```text
Megatron 如何训练
Megatron 权重如何转换成 HF 权重
并行切分细节
分布式 NCCL 权重同步
```

本文只回答一个问题：

```text
slime 已经拿到一批要发给 SGLang 的 CUDA named tensors 后，
它如何通过 CUDA IPC 让 SGLang 接收并加载这些权重？
```

## 1. 主线概览

共卡 IPC 权重同步的核心路径是：

```text
slime actor_model.update_weights()
  -> UpdateWeightFromTensor.update_weights()
  -> 生成一批 hf_named_tensors
  -> _send_to_colocated_engine()
  -> FlattenedTensorBucket
  -> MultiprocessingSerializer.serialize(...)
  -> Ray 调 SGLang engine.update_weights_from_tensor
  -> SGLang deserialize
  -> reconstruct tensors
  -> model.load_weights(...)
```

前面 01-03 的内容分别对应这里的：

```text
CUDA IPC:
  解释为什么 SGLang 可以跨进程访问 slime 侧 CUDA allocation

PyTorch serializer:
  解释 MultiprocessingSerializer 为什么能传 CUDA tensor

FlattenedTensorBucket:
  解释为什么多个权重 tensor 会先合成一个 flattened tensor
```

## 2. 权重同步入口

训练循环里会调用：

```python
actor_model.update_weights()
```

Megatron actor 在 colocated 模式下选择：

```python
UpdateWeightFromTensor
```

也就是本文关注的 CUDA IPC tensor path。

真正核心入口是：

```text
slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py::update_weights
```

它做的事情可以简化成：

```text
1. 暂停 SGLang generation
2. fl
3. 逐批拿到要同步的 HF/SGLang named tensorush SGLang caches
4. 每批调用 _send_hf_params
5. 等 SGLang 更新完成
6. 清理 CUDA IPC 相关缓存
7. 恢复 SGLang generation
```

其中第 3 步“如何得到 HF/SGLang named tensors”不是本文重点。本文从“已经有一批 `hf_named_tensors`”开始看。

## 3. 一批待同步权重是什么形态

进入 IPC 发送路径时，slime 手里是一批：

```python
hf_named_tensors: list[tuple[str, torch.Tensor]]
```

可以理解成：

```text
[
  ("model.layers.0.self_attn.q_proj.weight", cuda_tensor),
  ("model.layers.0.self_attn.k_proj.weight", cuda_tensor),
  ...
]
```

这些 tensor 已经在 CUDA 上，目标是让共卡的 SGLang 进程接收并加载它们。

## 4. _send_hf_params

`UpdateWeightFromTensor.update_weights()` 对每批 `hf_named_tensors` 调：

```python
refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
ray.get(refs)
del long_lived_tensors, hf_named_tensors
torch.cuda.ipc_collect()
```

这里最重要的是三个点：

```text
refs:
  SGLang 侧 Ray update 调用的 future

long_lived_tensors:
  producer 侧必须暂时保活的 CUDA tensor 对象

ray.get(refs) 之后再 del:
  确保 SGLang consumer 已经用完 IPC tensor 后，再释放 producer 引用
```

这对应 01 文档里的生命周期原则：

```text
producer 侧 allocation 必须覆盖 consumer 使用期间
```

## 5. _send_to_colocated_engine 核心逻辑

真正构造 IPC payload 的函数是：

```text
slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py::_send_to_colocated_engine
```

核心步骤：

```text
1. 把 named tensors 放进 FlattenedTensorBucket
2. 得到 flattened_tensor 和 metadata
3. 构造 flattened_tensor_data
4. 用 MultiprocessingSerializer.serialize(..., output_str=True)
5. 收集本次要发给 SGLang 的 serialized payload list
6. 调 SGLang engine.update_weights_from_tensor.remote(...)
```

关键代码形态：

```python
flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
metadata = flattened_tensor_bucket.get_metadata()
flattened_tensor_data = {
    "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
    "metadata": metadata,
}
long_live_tensors.append(flattened_tensor_data)
serialized_tensors.append(
    MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
)
```

这里的 `flattened_tensor_data` 是本文最关键的数据结构。

## 6. flattened_tensor_data 里有什么

结构：

```python
{
    "flattened_tensor": cuda_tensor,
    "metadata": metadata,
}
```

其中：

```text
flattened_tensor:
  一个 CUDA tensor
  里面按 byte 或连续元素保存多个权重 tensor 的内容
  这个 tensor 会被 PyTorch serializer 走 CUDA IPC

metadata:
  每个原始权重 tensor 的 name / shape / dtype / start_idx / end_idx
  SGLang 侧靠它 reconstruct
```

注意：

```text
CUDA IPC 负责共享 flattened_tensor 的底层 GPU allocation
metadata 负责告诉 SGLang 如何把 flattened_tensor 切回 named tensors
```

这是 01 和 03 的组合。

## 7. MultiprocessingSerializer 做了什么

slime 使用的是 SGLang 提供的：

```python
MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
```

它内部使用：

```python
ForkingPickler(buf).dump(obj)
```

因为 `flattened_tensor_data` 里含有 CUDA tensor，所以 PyTorch 会触发 CUDA tensor reducer：

```text
CUDA tensor
  -> CUDA IPC handle + tensor rebuild metadata
```

然后 `output_str=True` 会把 pickle bytes 做 base64，变成可以放进 Ray / JSON-like payload 的字符串。

所以 serialized payload 里逻辑上包含两层信息：

```text
PyTorch CUDA tensor IPC 信息:
  用于让 SGLang 打开 flattened_tensor

FlattenedTensorBucket metadata:
  用于让 SGLang reconstruct 权重 tensor
```

## 8. 为什么要 long_live_tensors

`long_live_tensors` 保存的是：

```python
flattened_tensor_data
```

它的目的不是给业务逻辑再用一次，而是保证：

```text
flattened_tensor_data["flattened_tensor"]
```

在 SGLang 侧反序列化和加载权重完成前不会被 Python GC / PyTorch allocator 释放。

如果 producer 侧太早释放，consumer 侧 IPC tensor 可能失效。

所以 slime 的顺序是：

```text
serialize payload
发给 SGLang
ray.get(refs) 等 SGLang 返回
del long_lived_tensors
torch.cuda.ipc_collect()
```

这条顺序是整个 IPC 权重同步里非常关键的生命周期协议。

## 9. SGLang 接收侧入口

SGLang 侧收到的是：

```python
engine.update_weights_from_tensor.remote(
    serialized_named_tensors=...,
    load_format="flattened_bucket",
    weight_version=...
)
```

核心接收路径：

```text
../sglang/python/sglang/srt/managers/scheduler_update_weights_mixin.py::update_weights_from_tensor
../sglang/python/sglang/srt/managers/tp_worker.py::update_weights_from_tensor
../sglang/python/sglang/srt/model_executor/model_runner.py::update_weights_from_tensor
```

先只看本质：

```text
SGLang worker 从 serialized_named_tensors 中取出属于自己的 payload
MultiprocessingSerializer.deserialize(...)
model_runner.update_weights_from_tensor(..., load_format="flattened_bucket")
```

后续关于多个 worker 如何对应 payload，单独在下一篇分析。

## 10. SGLang 反序列化后得到什么

反序列化：

```python
flattened_tensor_data = MultiprocessingSerializer.deserialize(serialized_payload)
```

得到：

```python
{
    "flattened_tensor": rebuilt_cuda_tensor,
    "metadata": metadata,
}
```

这里的 `rebuilt_cuda_tensor` 是 PyTorch 根据 CUDA IPC handle 在 SGLang 进程中打开的 CUDA tensor。

它和 slime producer 侧原始 flattened tensor 不是同一个 Python 对象，但底层可以访问同一块 GPU allocation。

## 11. SGLang 如何 reconstruct 权重

当：

```python
load_format == "flattened_bucket"
```

SGLang model runner 会进入：

```python
_update_weights_from_flattened_bucket(...)
```

逻辑是：

```python
flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
metadata = flattened_tensor_bucket_dict["metadata"]

bucket = FlattenedTensorBucket(
    flattened_tensor=flattened_tensor,
    metadata=converted_metadata,
)
reconstructed_tensors = bucket.reconstruct_tensors()
self.model.load_weights(reconstructed_tensors)
```

也就是：

```text
flattened CUDA tensor
  + metadata
  -> list[(weight_name, weight_tensor)]
  -> model.load_weights(...)
```

## 12. 一次 IPC 权重同步的数据流

可以压缩成这张图：

```text
slime process
  hf_named_tensors
    |
    | FlattenedTensorBucket
    v
  {
    flattened_tensor: CUDA tensor,
    metadata: [...]
  }
    |
    | MultiprocessingSerializer / ForkingPickler
    | CUDA tensor reducer creates IPC payload
    v
  serialized string
    |
    | Ray actor call
    v
SGLang process
  deserialize
    |
    | PyTorch rebuild CUDA tensor from IPC handle
    v
  {
    flattened_tensor: rebuilt CUDA tensor,
    metadata: [...]
  }
    |
    | FlattenedTensorBucket.reconstruct_tensors()
    v
  [(name, tensor), ...]
    |
    | model.load_weights(...)
    v
  SGLang model weights updated
```

## 13. 和 04 预分配 Buffer 优化的区别

slime + SGLang 当前这条路径更接近 03 的基础方案：

```text
每批 hf_named_tensors
  -> 构造 flattened_tensor
  -> payload 携带 flattened_tensor + metadata
```

04 文档里的 lmdeploy 优化是：

```text
预分配一个长期 IPC buffer
第一次发送 buffer handle
后续只发送 metadata
```

所以 04 是这一类 IPC 权重同步的进一步优化思路，不是本文描述的 slime + SGLang 主路径默认行为。

## 14. 本节需要掌握的结论

- slime 共卡权重同步的核心是 `UpdateWeightFromTensor`。
- 本文不关心权重如何从训练框架转换来，只关心已经有 CUDA named tensors 后如何发给 SGLang。
- slime 把多个 CUDA weight tensors 合成 `flattened_tensor + metadata`。
- `MultiprocessingSerializer` 让 `flattened_tensor` 走 PyTorch CUDA IPC。
- `long_lived_tensors` 用于保护 producer 侧 CUDA allocation 生命周期。
- SGLang 反序列化后拿到 rebuilt CUDA tensor 和 metadata。
- SGLang 用 `FlattenedTensorBucket.reconstruct_tensors()` 还原 named tensors。
- 最后通过 `model.load_weights(...)` 更新推理模型权重。

