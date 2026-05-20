# 03 FlattenedTensorBucket

本文对应本轮学习主线里的 FlattenedTensorBucket。

本节目标是理解：为什么 slime + SGLang 不直接把很多权重 tensor 一个个通过 CUDA IPC 传过去，而是先把它们合并成一个 flattened bucket，再用 metadata 恢复。

## 1. 为什么需要 FlattenedTensorBucket

如果直接把每个权重 tensor 单独传给 SGLang，会有几个问题。

### 1.1 Tensor 数量太多

一个模型可能有成百上千个参数。如果每个参数都单独序列化、单独产生 CUDA IPC 相关信息，Python 层和序列化层的开销会很明显。

### 1.2 IPC handle 和 metadata 太碎

CUDA IPC handle 本体不大，但一个完整的 PyTorch CUDA tensor payload 不只有 handle，还包括：

```text
device
storage size / offset
dtype
shape
stride
requires_grad
IPC refcount / event 等信息
```

很多小 tensor 会导致这些 metadata 被重复携带很多次。

### 1.3 Ray / Python 传输对象更复杂

slime 最终通过 Ray actor 调用 SGLang engine。相比传很多小对象，传一个较大的 flattened tensor 加一份 metadata 更容易管理，也更符合批量权重同步的需求。

### 1.4 方便按 buffer size 分桶

slime 有：

```text
--update-weight-buffer-size
```

它可以控制每次同步的权重 bucket 大小，避免一次构造或传输过大的中间 buffer。

所以 FlattenedTensorBucket 的核心目标是：

```text
多个 named tensors
  -> 合并成一个大 flattened tensor
  -> 额外记录 metadata
  -> 接收端再根据 metadata 切回原始 tensors
```

它主要减少的是：

```text
tensor 对象数量
CUDA IPC handle 数量
pickle / PyTorch metadata 数量
Python / Ray 调用负担
```

注意：FlattenedTensorBucket 不是完全没有拷贝。构造 bucket 时通常会在 GPU 上 `torch.cat` 出一个新的 contiguous flattened tensor。它避免的是跨进程 GPU -> CPU -> GPU 的完整数据搬运，而不是避免所有 GPU 内部拷贝。

## 2. 核心数据结构

SGLang 源码位置：

```text
../sglang/python/sglang/srt/weight_sync/tensor_bucket.py
```

核心 metadata 结构：

```python
@dataclass
class FlattenedTensorMetadata:
    name: str
    shape: torch.Size
    dtype: torch.dtype
    start_idx: int
    end_idx: int
    numel: int
```

每个字段含义：

```text
name:
  原始权重名

shape:
  原始 tensor shape

dtype:
  原始 tensor dtype

start_idx / end_idx:
  该 tensor 在 flattened byte tensor 中的起止位置

numel:
  该 tensor 转成 uint8 byte view 后的元素个数，也就是字节数
```

注意这里的 `numel` 不是原 tensor 按 dtype 计算的元素数量，而是转成 `torch.uint8` 之后的 byte 数量。

## 3. 发送端如何构造 bucket

构造逻辑可以简化为：

```python
current_idx = 0
flattened_tensors = []
metadata = []

for name, tensor in named_tensors:
    flattened = tensor.flatten().view(torch.uint8)
    flattened_tensors.append(flattened)

    numel = flattened.numel()
    metadata.append(
        FlattenedTensorMetadata(
            name=name,
            shape=tensor.shape,
            dtype=tensor.dtype,
            start_idx=current_idx,
            end_idx=current_idx + numel,
            numel=numel,
        )
    )
    current_idx += numel

flattened_tensor = torch.cat(flattened_tensors, dim=0)
```

关键点是：

```python
tensor.flatten().view(torch.uint8)
```

它把不同 dtype 的 tensor 都转换成 byte 视图：

```text
bf16 tensor -> byte view
fp32 tensor -> byte view
int8 tensor -> byte view
```

这样多个不同 dtype / shape 的 tensor 就能拼接进同一个 `torch.uint8` flattened tensor。

## 4. 接收端如何 reconstruct

接收端拿到：

```text
flattened_tensor
metadata
```

然后对每个 metadata 做：

```python
tensor = (
    flattened_tensor[meta.start_idx : meta.end_idx]
    .view(meta.dtype)
    .reshape(meta.shape)
)
```

恢复成：

```python
(meta.name, tensor)
```

所以 reconstruct 的本质是：

```text
byte range
  -> 按原 dtype 解释
  -> reshape 回原 shape
  -> 绑定回原 name
```

## 5. 和 CUDA IPC 的关系

FlattenedTensorBucket 本身不是 CUDA IPC。

它只是把很多 tensor 合并成一个更适合 IPC 传输的大 tensor：

```text
很多 CUDA tensors
  -> 一个 CUDA uint8 flattened tensor
```

真正的 CUDA IPC 发生在这个 `flattened_tensor` 被 PyTorch / ForkingPickler 序列化时。

也就是说：

```text
FlattenedTensorBucket:
  减少 tensor 数量，生成 flattened_tensor + metadata

PyTorch CUDA IPC serializer:
  为 flattened_tensor 生成 CUDA IPC handle + PyTorch tensor rebuild 信息
```

二者解决的是不同层次的问题。

## 6. slime 发送侧源码映射

slime 源码位置：

```text
slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py
```

核心逻辑在 `_send_to_colocated_engine`：

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

这里有三层含义：

```text
FlattenedTensorBucket:
  把多个权重 tensor 合成一个 flattened_tensor

MultiprocessingSerializer:
  让 PyTorch 对 flattened_tensor 走 CUDA IPC 序列化

long_live_tensors:
  保持 producer 侧 flattened_tensor 存活，直到 SGLang consumer 用完
```

## 7. SGLang 接收侧源码映射

SGLang 接收侧相关位置：

```text
../sglang/python/sglang/srt/model_executor/model_runner.py
```

在 `_update_weights_from_flattened_bucket` 里，SGLang 从反序列化结果里取出：

```python
flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
metadata = flattened_tensor_bucket_dict["metadata"]
```

然后：

```python
bucket = FlattenedTensorBucket(
    flattened_tensor=flattened_tensor,
    metadata=converted_metadata,
)
reconstructed_tensors = bucket.reconstruct_tensors()
self.model.load_weights(reconstructed_tensors)
```

也就是：

```text
serialized payload
  -> deserialize
  -> flattened_tensor + metadata
  -> reconstruct named tensors
  -> load_weights
```

## 8. 完整数据流

发送侧：

```text
[("w1", tensor1), ("w2", tensor2), ...]
  -> FlattenedTensorBucket
  -> {
       "flattened_tensor": 一个 CUDA uint8 tensor,
       "metadata": 每个原 tensor 的 name / shape / dtype / start / end
     }
  -> MultiprocessingSerializer.serialize(..., output_str=True)
  -> Ray actor call
```

接收侧：

```text
serialized string
  -> MultiprocessingSerializer.deserialize(...)
  -> {
       "flattened_tensor": 已经通过 CUDA IPC rebuild 的 CUDA tensor,
       "metadata": metadata
     }
  -> FlattenedTensorBucket(...).reconstruct_tensors()
  -> [("w1", restored_tensor1), ("w2", restored_tensor2), ...]
  -> model.load_weights(...)
```

## 9. 本节需要掌握的结论

- FlattenedTensorBucket 的目标是减少跨进程同步时的 tensor 数量和 metadata / IPC handle 数量。
- 它把多个 named tensors 转成一个 `torch.uint8` flattened tensor。
- metadata 记录每个 tensor 的 name、shape、dtype、start/end byte offset。
- reconstruct 时按 byte range 切片，再 `view(dtype).reshape(shape)`。
- FlattenedTensorBucket 本身不是 CUDA IPC；CUDA IPC 发生在 flattened tensor 被 PyTorch serializer 处理时。
- 构造 bucket 通常会有一次 GPU 内部 `torch.cat` 拷贝。
- slime 发送侧负责 flatten + serialize + 保持 long-lived 引用。
- SGLang 接收侧负责 deserialize + reconstruct + load_weights。
