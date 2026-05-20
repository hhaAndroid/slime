# CUDA IPC 权重同步学习计划

这个计划聚焦 slime + SGLang 在 colocated 场景下使用的 CUDA IPC tensor
权重同步路径。分布式 NCCL 权重同步和完整训练循环调度，本轮先不展开。

## 学习范围

- 从最基本原理理解 CUDA IPC。
- 每个概念都配一个最简依赖的可运行代码。
- 每个最小 demo 都要能映射回 slime / SGLang 的真实源码。
- 主线聚焦 colocated 路径：
`slime UpdateWeightFromTensor -> CUDA tensor 序列化 -> Ray actor 调用 -> SGLang update_weights_from_tensor`。

## 暂不学习

- `update_weights_from_distributed` 对应的分布式权重同步。
- 训练进程和 rollout engine 之间的 NCCL 权重更新 process group 初始化。
- 完整 rollout / training 调度细节；只保留理解安全替换权重所需的部分。
- SGLang checkpoint-engine 的 `/update_weights_from_ipc` 路径；除非后续为了对比再单独看。

## 每个 Item 的学习方式

每个 item 都按同样的结构推进：

1. 原理：先建立一个足够小的心智模型。
2. 最简可运行代码：创建或运行一个最小脚本来验证该概念。
3. 预期输出：说明应该看到什么，以及这个输出证明了什么。
4. 源码映射：把 demo 对应到 slime / SGLang 的真实函数。
5. 问题和 debug：停下来围绕这个 case 细问，再进入下一个 item。

## 学习 Items

### 1. CUDA IPC 基本模型

文档：`01_cuda_ipc_basic_principles.md`

脚本：`01_basic_cuda_ipc.py`

目标：
理解一个进程如何把 CUDA tensor 共享给另一个进程，而不是通过 CPU 内存
拷贝 tensor 内容。

最小 case：

- producer 进程创建一个 CUDA tensor。
- consumer 进程通过 `torch.multiprocessing` 收到它。
- consumer 读取并修改 tensor。
- producer 验证修改是否可见。

关键概念：

- CUDA IPC handle。
- producer / consumer 进程角色。
- 共享 GPU storage 和复制 Python 对象的区别。
- tensor 生命周期要求。
- CUDA 同步。

真实源码映射：

- 这是 slime 把 CUDA tensor 通过序列化 payload 发给 SGLang 的概念基础。
- 后续会映射到 `MultiprocessingSerializer.serialize(...)` 和 SGLang 反序列化。

完成标准：

- 能解释哪些东西被共享了，哪些东西没有共享。
- 能解释为什么 consumer 使用期间 producer 侧 tensor 必须保持存活。

### 2. PyTorch CUDA Tensor 序列化

文档：`02_pytorch_cuda_tensor_ipc_serialization.md`

脚本：`02_torch_serializer.py`

目标：
理解 PyTorch 如何为 multiprocessing 序列化 CUDA tensor，以及它为什么不同于
直接复制 tensor bytes。

最小 case：

- 用 multiprocessing 相同机制序列化一个 CUDA tensor。
- 在另一个进程里反序列化。
- 打印 device、shape、dtype、values，以及若干 storage / debug 信息。

关键概念：

- `ForkingPickler`。
- PyTorch tensor reducer。
- Python 序列化背后隐藏的 CUDA IPC handle。
- 为什么普通 JSON 无法表达这个 payload。

真实源码映射：

- SGLang serializer：
`../sglang/python/sglang/srt/utils/common.py::MultiprocessingSerializer`。
- slime 通过下面文件间接引入这个 serializer：
`slime/backends/megatron_utils/sglang.py`。

完成标准：

- 能解释为什么序列化 CUDA tensor 仍然可以保留 GPU 共享语义。
- 能解释为什么这件事只在合适的 multiprocessing / runtime 上下文中成立。

### 3. FlattenedTensorBucket

文档：`03_flattened_tensor_bucket.md`

脚本：`04_flattened_bucket.py`

目标：
理解为什么要把很多 named weight tensors flatten 成一个 bucket，以及如何再恢复。

最小 case：

- 创建几个 named CUDA tensors；如果支持，可以包含不同 shape 和 dtype。
- 构造 `FlattenedTensorBucket`。
- 打印 metadata。
- reconstruct tensors，并和原始 tensors 比较。

关键概念：

- 减少 tensor 数量和 CUDA IPC handle 数量。
- 通过 `view(torch.uint8)` 做字节级 flatten。
- metadata 字段：name、shape、dtype、start/end offsets。
- 在不改变逻辑 tensor 值的前提下重建 tensor。

真实源码映射：

- SGLang：
`../sglang/python/sglang/srt/weight_sync/tensor_bucket.py::FlattenedTensorBucket`。
- slime sender 创建：
`{"flattened_tensor": ..., "metadata": ...}`。
- SGLang receiver 在下面函数里 reconstruct：
`../sglang/python/sglang/srt/model_executor/model_runner.py::_update_weights_from_flattened_bucket`。

完成标准：

- 能手动追踪一个 tensor 从原始 tensor 到 flattened byte range，再到 reconstructed tensor。
- 能解释为什么 slime 要按 update buffer size 分 bucket。

### 4. 预分配 IPC Buffer 和 Event 同步复用

文档：`04_preallocated_ipc_buffer_event.md`

脚本：后续如需实践，再新增 `04_preallocated_ipc_buffer_event.py`

目标：
理解在普通 `FlattenedTensorBucket` 之上的进一步优化：预分配并复用同一块 CUDA IPC buffer，后续只传 metadata，并用 interprocess CUDA event 保证读写顺序。

最小 case：

- 第一次发送时，producer 创建并发送 `flattened_tensor` IPC handle、metadata、event handle。
- 后续 bucket 复用同一块 buffer，只发送新的 metadata。
- consumer 缓存第一次打开的 IPC tensor。
- producer / consumer 通过 CUDA interprocess event 交接 buffer 读写权限。

关键概念：

- 预分配 IPC buffer。
- per-dtype buffer cache。
- `require_clone=False`。
- `event_ipc_handle`。
- 首次 / resize / dtype change 才重新发送 flattened tensor。
- 后续只发送 metadata。

真实源码映射：

- xtuner：
`/mnt/shared-storage-user/huanghaian/code/temp/xtuner/xtuner/v1/rl/trainer/update_weighter.py::_build_lmdeploy_flattened_tensor_data`。
- lmdeploy：
`/mnt/shared-storage-user/huanghaian/code/lmdeploy/lmdeploy/utils.py::FlattenedTensorBucket`。
`/mnt/shared-storage-user/huanghaian/code/lmdeploy/lmdeploy/pytorch/engine/model_agent/agent.py::update_params`。

完成标准：

- 能解释相比 03 基础方案减少了哪些重复开销。
- 能解释为什么需要 event 同步。
- 能解释为什么 `require_clone=False` 是 buffer 复用成立的关键。

### 5. Mock SGLang Receiver 和 Weight Loader

脚本：`06_mock_sglang_receiver.py`

目标：
用一个很小的模型或 fake loader 复现 SGLang 接收侧逻辑。

最小 case：

- 接收一个 serialized flattened bucket。
- 反序列化。
- reconstruct named tensors。
- 按参数名加载到一个小 `torch.nn.Module` 中。
- 验证模型参数确实被更新。

关键概念：

- per-TP payload 选择。
- `load_format="flattened_bucket"`。
- `model.load_weights(...)` 的等价行为。
- cache flush 和参数替换是两件不同的事。

真实源码映射：

- SGLang：
`../sglang/python/sglang/srt/managers/scheduler_update_weights_mixin.py::update_weights_from_tensor`。
`../sglang/python/sglang/srt/managers/tp_worker.py::update_weights_from_tensor`。
`../sglang/python/sglang/srt/model_executor/model_runner.py::update_weights_from_tensor`。
`../sglang/python/sglang/srt/model_executor/model_runner.py::_update_weights_from_flattened_bucket`。

完成标准：

- 能解释一个 serialized payload 如何变成模型里的 live weights。
- 能区分 tensor transport 和模型特定的 weight loading。

### 6. 生命周期和 Debug Cases

脚本：`07_lifecycle_debug.py`

目标：
理解 CUDA IPC 生命周期和清理相关的失败模式。

最小 cases：

- producer tensor 保持存活直到 consumer 完成。
- 过早删除 producer 引用，观察行为。
- 对比调用和不调用 `torch.cuda.ipc_collect()`。
- 可选：用 `torch.cuda.memory_allocated()` 观察显存变化。

关键概念：

- producer 侧生命周期。
- consumer handle release。
- CUDA IPC cache 清理。
- 为什么 slime 里会做：
`del long_lived_tensors, hf_named_tensors`
然后调用 `torch.cuda.ipc_collect()`。

真实源码映射：

- slime：
`slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py::update_weights`。

完成标准：

- 能解释 `_send_to_colocated_engine` 里的 `long_live_tensors`。
- 能解释为什么 cleanup 要发生在 `ray.get(refs)` 之后。

## 源码阅读索引

slime：

- `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
- `slime/backends/megatron_utils/update_weight/hf_weight_iterator_base.py`
- `slime/backends/megatron_utils/update_weight/hf_weight_iterator_direct.py`
- `slime/backends/megatron_utils/update_weight/hf_weight_iterator_bridge.py`
- `slime/backends/megatron_utils/sglang.py`

SGLang：

- `../sglang/python/sglang/srt/utils/common.py`
- `../sglang/python/sglang/srt/weight_sync/tensor_bucket.py`
- `../sglang/python/sglang/srt/managers/scheduler_update_weights_mixin.py`
- `../sglang/python/sglang/srt/managers/tp_worker.py`
- `../sglang/python/sglang/srt/model_executor/model_runner.py`

## 后续文档

- `05_slime_sglang_megatron_colocated_weight_sync.md`：slime + Megatron + SGLang 共卡权重同步完整主流程。
- `06_sglang_tp_colocated_ipc_weight_sync.md`：SGLang TP 情况下共卡 IPC 权重同步的 rank / payload / worker 对应关系。

## 进度

- [ ] 1. CUDA IPC 基本模型（进行中：最小 demo 已创建并跑通）
- [ ] 2. PyTorch CUDA Tensor 序列化（进行中：原理文档已创建）
- [ ] 3. FlattenedTensorBucket（进行中：原理文档已创建）
- [ ] 4. 预分配 IPC Buffer 和 Event 同步复用（进行中：原理文档已创建）
- [ ] 5. slime + Megatron + SGLang 共卡权重同步完整逻辑（进行中：原理文档已创建）
- [ ] 6. TP 情况下的共卡权重同步逻辑（进行中：原理文档已创建）
- [ ] 7. 生命周期和 Debug Cases
