# 02 PyTorch CUDA Tensor IPC 序列化

本文对应学习计划里的 Item 2：PyTorch CUDA tensor 序列化。

本节目标是理解 PyTorch 在跨进程传递 CUDA tensor 时做了什么。重点不是 CUDA driver 的底层细节，而是 PyTorch 如何把一个 CUDA tensor 拆成“可跨进程传输的重建配方”，以及 slime + SGLang 为什么可以直接复用这套机制做 colocated 权重同步。

## 1. PyTorch IPC 的核心思想

PyTorch 跨进程传 CUDA tensor 时，不会把 GPU 数据内容拷贝成普通 bytes。

它做的是：

```text
CUDA tensor
  -> CUDA IPC handle
  -> tensor metadata
  -> PyTorch rebuild 信息
```

consumer 进程收到这些信息后，再重建出一个新的 Python tensor 对象。这个新 tensor 对象可以指向 producer 侧原始 GPU allocation。

所以这里传递的不是：

```text
完整 tensor 数据
```

而是：

```text
如何在另一个进程里重新打开并解释这块 CUDA storage
```

## 2. PyTorch 会描述哪些内容

对一个 CUDA tensor，PyTorch 需要描述两类信息。

第一类是 storage / CUDA IPC 相关信息：

```text
CUDA IPC handle
device
storage size
storage offset
IPC ref counter 信息
event handle / 是否需要 event 同步
```

第二类是 tensor 视图相关信息：

```text
tensor type
dtype
shape
stride
tensor offset
requires_grad
```

这和上一节的结论一致：

```text
打开 CUDA IPC handle 不需要 shape / dtype
但是把 raw storage 重建成 tensor 必须需要 shape / dtype / stride / offset
```

## 3. PyTorch 序列化过程

简化过程如下：

```text
producer 进程
  创建 CUDA tensor
  把 tensor 交给 multiprocessing Queue / ForkingPickler
  PyTorch 发现这是 CUDA tensor
  调用 tensor reducer
  导出 CUDA IPC handle
  打包 handle + tensor metadata + rebuild 信息(包括 rebuild_fn 和 meta 信息)

普通 IPC 通道
  只传递这些小 payload

consumer 进程
  反序列化 payload
  调用传递过来的 rebuild 函数
  打开 CUDA IPC handle
  根据传递过来的 metadata 重建 torch.Tensor
```

这也是为什么 PyTorch 传 CUDA tensor 可以避免 GPU -> CPU -> GPU 的完整数据搬运。

## 4. 为什么 reducer 要带 rebuild 函数

一个容易误解的问题是：既然 args 里已经有 CUDA IPC handle、device、shape、stride、offset 等信息，为什么还要传一个 `rebuild_fn`？

关键点是：

```text
metadata 只描述 tensor 应该长什么样
rebuild_fn 描述这组 metadata 应该用哪套逻辑解释和执行
```

如果我们已经非常确定：

```text
传的一定是 CUDA tensor
args 一定匹配 PyTorch 的 CUDA tensor rebuild 参数
producer / consumer 使用兼容的 PyTorch 版本
接收端知道 args 每个位置的含义
```

那么理论上可以在接收端直接 import 对应函数：

```python
from torch.multiprocessing.reductions import rebuild_cuda_tensor

tensor = rebuild_cuda_tensor(*args)
```

也就是说，在“只支持 CUDA tensor，并且愿意依赖 PyTorch 内部参数协议”的前提下，直接 import 固定 rebuild 函数是可以工作的。

但是 PyTorch 的 reducer 机制是通用协议。它不希望接收端猜：

```text
这组 args 应该交给 rebuild_cuda_tensor？
还是 CPU shared memory 的 rebuild 函数？
还是 storage / tensor subclass / 其他对象的 rebuild 函数？
```

所以 reducer 返回：

```python
(rebuild_fn, rebuild_args)
```

含义是：

```text
用这个 rebuild_fn
解释这组 rebuild_args
重建原始对象
```

这让 payload 自描述：它不仅包含“参数”，也包含“参数该由谁解释”。

还要注意，pickle 通常不是把函数代码本体序列化进去，而是记录函数引用，例如：

```text
module path + function name
```

反序列化时，接收端再 import 到这个函数并调用它。因此“传 rebuild_fn”通常不是传一大段函数代码，而是传一个可定位的函数引用。

所以可以这样总结：

```text
如果自己写死只传 CUDA tensor：
  可以直接 import rebuild_cuda_tensor(*args)
  但强依赖 PyTorch 内部 args 顺序和版本

如果使用 PyTorch reducer / pickle 协议：
  reducer 返回 rebuild_fn + args
  接收端不需要猜该用哪个 rebuild 逻辑
```

slime + SGLang 当前路径更接近后一种：业务层直接把包含 CUDA tensor 的对象交给 `ForkingPickler`，让 PyTorch 自动选择 reducer 和 rebuild 函数，而不是业务代码手写 `rebuild_cuda_tensor(*args)`。

## 5. 有哪些实现方式

在本轮学习范围内，可以分成三种方式。

### 方式一：直接用 torch.multiprocessing Queue / Pipe

示意：

```python
queue.put(cuda_tensor)
tensor = queue.get()
```

这是最直观的方式。PyTorch 会在内部自动调用 CUDA tensor reducer，consumer 侧自动 rebuild。

优点：

```text
代码简单
自动处理 reduce / rebuild
```

限制：

```text
适合普通 multiprocessing 场景
业务层不容易拿到中间 payload
```

### 方式二：直接用 ForkingPickler dump 包含 CUDA tensor 的对象

示意：

```python
buf = BytesIO()
ForkingPickler(buf).dump(obj)
payload = buf.getvalue()
```

如果 `obj` 里包含 CUDA tensor，`ForkingPickler` 会自动触发 PyTorch 的 tensor reducer。

反序列化后，PyTorch 会自动调用 rebuild 逻辑，用户拿到的就是重建后的 tensor。

SGLang 的 `MultiprocessingSerializer` 使用的就是这条路。

```python

class MultiprocessingSerializer:
    @staticmethod
    def serialize(obj, output_str: bool = False):
        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        buf.seek(0)
        output = buf.read()

        if output_str:
            # Convert bytes to base64-encoded string
            output = pybase64.b64encode(output).decode("utf-8")

        return output

    @staticmethod
    def deserialize(data):
        if isinstance(data, str):
            # Decode base64 string to bytes
            data = pybase64.b64decode(data, validate=True)

        return SafeUnpickler(io.BytesIO(data)).load()

serialized_data = MultiprocessingSerializer.serialize(state_dict, output_str=True)
```

### 方式三：手动调用 reduce_tensor，再自己 rebuild

示意：

```python
from torch.multiprocessing.reductions import reduce_tensor

item = reduce_tensor(cuda_tensor)
```

`item` 不是 tensor，而是：

```python
(rebuild_fn, rebuild_args)
```

也就是一份“重建配方”。

如果你把这个配方传给另一个进程，接收端需要手动执行：

```python
tensor = rebuild_fn(*rebuild_args)
```

这种方式更显式，但更容易依赖 PyTorch 内部参数顺序。本轮只需要理解它的语义，不建议把它作为默认实现方式。目前 lmdeploy 采用的是这种方式，sglang 采用的更简单的方式。

## 6. 直接 dump tensor 和手动 reduce_tensor 的区别

这两种写法看起来接近，但语义不同。

### 直接 dump

```python
ForkingPickler(buf).dump(cuda_tensor)
```

含义：

```text
dump 时自动 reduce
load 时自动 rebuild
用户直接得到 tensor
```

### 手动 reduce 后再 dump

```python
item = reduce_tensor(cuda_tensor)
ForkingPickler(buf).dump(item)
```

含义：

```text
用户先把 tensor 变成 (rebuild_fn, rebuild_args)
dump 的是这个重建配方
load 后拿到的也是这个重建配方
用户需要自己调用 rebuild_fn(*rebuild_args)
```

所以如果接收端只是 unpickle，但没有调用 `rebuild_fn(*args)`，它并不会得到真正的 tensor。

## 7. 你关心的 serialize_state_dict 代码

代码：

```python
def serialize_state_dict(state_dict: dict) -> str:
    import base64
    from io import BytesIO
    from multiprocessing.reduction import ForkingPickler

    from torch.multiprocessing.reductions import reduce_tensor

    data = [(k, reduce_tensor(v)) for k, v in state_dict.items()]
    buf = BytesIO()
    ForkingPickler(buf).dump(data)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
```

关键在这一行：

```python
data = [(k, reduce_tensor(v)) for k, v in state_dict.items()]
```

它把每个 tensor 变成：

```python
(rebuild_fn, rebuild_args)
```

所以 `data` 的形态大致是：

```python
[
    ("layer.weight", (rebuild_cuda_tensor, (...))),
    ("layer.bias",   (rebuild_cuda_tensor, (...))),
]
```

后面的：

```python
ForkingPickler(buf).dump(data)
```

只是把这个 Python list 序列化成 bytes。

最后：

```python
base64.b64encode(...).decode("utf-8")
```

只是把 bytes 包装成字符串，方便通过 JSON / HTTP / Ray 参数等通道传输。

因此，这段代码序列化的是：

```text
state_dict key
每个 tensor 的 PyTorch rebuild 配方
```

而不是直接序列化一个“反序列化后自动变成 tensor 的 state_dict”。

## 8. 你看到的 _construct 代码

代码：

```python
def _construct(item, require_clone: bool = True):
    func, args = item
    args = list(args)
    args[6] = torch.cuda.current_device()  # device id.
    ipc_tensor = func(*args)
    return ipc_tensor.clone() if require_clone else ipc_tensor
```

这段代码说明它走的是：

```text
手动 reduce_tensor
传输 (rebuild_fn, rebuild_args)
接收端手动 rebuild
```

逐步拆解：

### 8.1 取出重建函数和参数

```python
func, args = item
```

这里的 `item` 就是：

```python
(rebuild_fn, rebuild_args)
```

### 8.2 修改 device 参数

```python
args = list(args)
args[6] = torch.cuda.current_device()
```

这说明这段代码假设 `args[6]` 是 device id 或相关 device 字段，然后强行改成当前 CUDA device。

这个动作比较敏感，因为它依赖 PyTorch 当前版本里 `rebuild_args` 的参数顺序。这个顺序不是业务代码应该长期依赖的稳定接口。

从意图上看，它可能是为了让接收端在当前 rank / 当前 GPU 上构造 tensor。

### 8.3 调用 rebuild 函数

```python
ipc_tensor = func(*args)
```

这一步才真正执行重建：

```text
打开 CUDA IPC handle
构造 CUDA storage
包装成 tensor
```

### 8.4 是否 clone

```python
return ipc_tensor.clone() if require_clone else ipc_tensor
```

这里的语义非常重要。

如果：

```python
require_clone = False
```

返回的是 IPC 打开的共享 tensor：

```text
零拷贝
依赖 producer allocation 生命周期
consumer 修改可能影响 producer
```

如果：

```python
require_clone = True
```

会先通过 IPC tensor 读数据，再 clone 成当前进程自己持有的新 tensor：

```text
多一次 GPU copy
不再依赖 producer 的原始 allocation
consumer 修改不影响 producer
```

所以这段 `_construct` 的完整语义是：

```text
收到 reduce_tensor 生成的重建配方
修改 device 参数
手动调用 rebuild_fn
得到 IPC tensor
可选 clone 成本进程自己的 tensor
```

## 9. 和 slime colocated 权重同步的关系

slime 通过下面文件复用 SGLang 的工具：

```text
slime/backends/megatron_utils/sglang.py
```

其中包括：

```python
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
```

在 colocated 权重同步里，slime 大致构造：

```python
flattened_tensor_data = {
    "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
    "metadata": metadata,
}
```

然后：

```python
MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
```

这里的 `flattened_tensor` 是 CUDA tensor。`ForkingPickler` 会自动触发 PyTorch CUDA tensor reducer。

SGLang 侧反序列化后得到：

```python
{
    "flattened_tensor": 已经 rebuild 好的 CUDA tensor,
    "metadata": metadata,
}
```

然后再根据 metadata reconstruct 出原始 named tensors，并加载到模型权重里。

所以 slime + SGLang 当前路径的重点是：

```text
业务层不手动 reduce_tensor
业务层直接 serialize 包含 CUDA tensor 的对象
PyTorch / ForkingPickler 自动负责 CUDA IPC reduce/rebuild
业务层只负责额外的 weight metadata
```

## 12. 本节需要掌握的结论

- PyTorch 跨进程传 CUDA tensor 时，传的是 CUDA IPC handle + tensor metadata + rebuild 信息，不是 GPU 数据 bytes。
- `ForkingPickler.dump(obj)` 如果遇到 CUDA tensor，会自动触发 PyTorch tensor reducer。
- `reduce_tensor(tensor)` 返回的是 `(rebuild_fn, rebuild_args)`，也就是重建配方。
- 如果手动保存 `reduce_tensor` 的返回值，接收端需要手动调用 `rebuild_fn(*args)`。
- `rebuild_fn` 主要让 payload 自描述：接收端知道这组 args 应该由哪套 rebuild 逻辑解释。
- 如果确定只支持 CUDA tensor，也可以直接 import 固定的 `rebuild_cuda_tensor`，但这会强依赖 PyTorch 内部参数协议和版本。
- `rebuild_fn` 必须存在，因为 metadata 只描述形状和类型，不描述如何打开 storage、处理 IPC refcount、构造 tensor。
- `_construct` 代码里的 `func(*args)` 是真正 rebuild tensor 的地方。
- `_construct` 里的 `clone()` 会把 IPC 共享 tensor 复制成本进程自己的 tensor，代价是多一次 GPU copy。
- SGLang 的 `MultiprocessingSerializer` 走的是直接 dump 对象的方式，不需要业务层手动调用 `reduce_tensor`。
- slime colocated 权重同步复用 SGLang serializer，把 CUDA flattened bucket tensor 交给 PyTorch IPC 机制处理。

