# 06 SGLang TP 情况下的共卡 IPC 权重同步

本文基于 `05_slime_sglang_megatron_colocated_weight_sync.md`，只说明 SGLang rollout engine 开 TP 后，IPC 权重同步路径和单 worker 情况相比有什么变化。

按你的要求，本文不关心训练侧开了多少 TP，也不展开训练侧如何准备权重。本文只关心：

```text
slime 已经拿到要发给 SGLang 的 CUDA 权重后，
在 SGLang TP > 1 时，payload 如何组织、如何发送、SGLang 每个 worker 如何取自己的那份。
```

## 1. 单 worker 情况回顾

不考虑 TP 时，可以把一次同步理解成：

```text
slime rank
  hf_named_tensors
    -> FlattenedTensorBucket
    -> serialize
    -> engine.update_weights_from_tensor(serialized_named_tensors=[payload])

SGLang worker
  payload = serialized_named_tensors[0]
  deserialize
  reconstruct
  load_weights
```

也就是：

```text
一个 SGLang worker
对应一个 serialized payload
```

## 2. SGLang TP > 1 后的核心变化

SGLang TP > 1 后，一个 rollout engine 内部有多个 worker。

因此，Ray 调用传给 SGLang engine 的不再是“一个 payload”，而是：

```python
serialized_named_tensors = [
    payload_for_worker_0, # 意思是训练 rank0 有一份完整权重，然后 ipc 后得到 payload_for_worker_0
    payload_for_worker_1, # 意思是训练 rank1 有一份完整权重，然后 ipc 后得到 payload_for_worker_1，然后通过 gather 发送给 rank0
    ...
    payload_for_worker_n_minus_1,
]
```

SGLang 侧每个 worker 根据自己的 `tp_rank` 取对应元素：

```python
recv_req.serialized_named_tensors[self.tp_rank]
```

所以 TP 情况下最核心的规则是：

```text
serialized_named_tensors[i] 必须对应 SGLang tp_rank == i 的 worker
```

## 3. slime 如何为每个 engine 建 gather group

源码：

```text
slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py::connect_rollout_engines
```

核心逻辑：

```python
group_ranks = list(range(gpu_offset, gpu_offset + gpu_count))
new_group = dist.new_group(ranks=group_ranks, backend="gloo")
```

可以理解成：

```text
一个 SGLang engine 占 N 张 GPU
slime 就为这 N 个 colocated ranks 建一个 Gloo group
```

例如：

```text
engine_0:
  gpu_offset = 0
  gpu_count = 4

group_ranks = [0, 1, 2, 3]
ipc_gather_src = 0
```

这些 ranks 后续各自生成 serialized payload，然后 gather 到 `ipc_gather_src`。

## 4. 每个 rank 先各自 serialize

在 `_send_to_colocated_engine` 中，每个参与该 engine 的 rank 都会做：

```python
flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
flattened_tensor_data = {
    "flattened_tensor": flattened_tensor_bucket.get_flattened_tensor(),
    "metadata": metadata,
}
serialized_tensors.append(
    MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
)
```

这里得到的是当前 rank 准备给对应 SGLang worker 的 serialized payload。

如果只有一个 dtype bucket：

```text
serialized_tensors = [payload]
```

如果因为兼容旧 FlattenedTensorBucket，需要按 dtype 拆成多个 bucket：

```text
serialized_tensors = [payload_dtype_0, payload_dtype_1, ...]
```

当前 SGLang 版本支持 multi dtype bucket 时，通常就是一个 payload。

## 5. gather_object 把多个 rank 的 payload 收到 src rank

每个 rank 有自己的：

```python
serialized_tensors
```

然后执行：

```python
serialized_named_tensors = (
    [None] * dist.get_world_size(ipc_gather_group)
    if ipc_gather_src == dist.get_rank()
    else None
)

dist.gather_object(
    serialized_tensors,
    object_gather_list=serialized_named_tensors,
    dst=ipc_gather_src,
    group=ipc_gather_group,
)
```

gather 后，src rank 上得到：

```python
serialized_named_tensors = [
    rank0_serialized_tensors,
    rank1_serialized_tensors,
    rank2_serialized_tensors,
    ...
]
```

如果每个 rank 只有一个 payload，那么形态大致是：

```python
[
    [payload_from_rank0],
    [payload_from_rank1],
    [payload_from_rank2],
    ...
]
```

注意：这里还不是直接发给 SGLang 的最终 list，因为外面还有一层 dtype bucket 维度。

## 6. src rank 组装 SGLang 需要的 list

src rank 上：

```python
num_dtypes = len(serialized_named_tensors[0])
for i in range(num_dtypes):
    kwargs = {
        "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
        "load_format": "flattened_bucket",
        "weight_version": str(weight_version),
    }
    refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))
```

如果只有一个 dtype bucket，`i = 0`，最终发给 SGLang 的是：

```python
serialized_named_tensors = [
    payload_from_rank0,
    payload_from_rank1,
    payload_from_rank2,
    ...
]
```

这个 list 的长度就是该 engine 的 worker 数。

所以对 SGLang 来说，它收到的结构是：

```text
serialized_named_tensors[0] -> 给 tp_rank 0
serialized_named_tensors[1] -> 给 tp_rank 1
serialized_named_tensors[2] -> 给 tp_rank 2
...
```

## 7. SGLang worker 如何取自己的 payload

源码：

```text
../sglang/python/sglang/srt/managers/tp_worker.py::update_weights_from_tensor
```

关键代码：

```python
named_tensors = MultiprocessingSerializer.deserialize(
    recv_req.serialized_named_tensors[self.tp_rank]
)
```

这就是 TP 情况下最重要的一行。

每个 SGLang worker 都收到同一个 request，但是：

```text
tp_rank 0 取 serialized_named_tensors[0]
tp_rank 1 取 serialized_named_tensors[1]
tp_rank 2 取 serialized_named_tensors[2]
...
```

然后各自：

```text
deserialize
reconstruct flattened bucket
model.load_weights
```

## 8. 为什么 list 顺序很重要

由于 SGLang worker 是用：

```python
self.tp_rank
```

作为 index，所以 slime 侧组装 list 的顺序必须和 SGLang worker 的 rank 顺序一致。

slime 侧顺序来自：

```python
group_ranks = list(range(gpu_offset, gpu_offset + gpu_count))
```

以及 `dist.gather_object` 在 group 内的收集顺序。

因此隐含假设是：

```text
engine 的 GPU range 顺序
  == slime gather group rank 顺序
  == SGLang tp_rank 顺序
```

如果这个映射错了，可能导致：

```text
tp_rank 0 拿到本应给 tp_rank 1 的 payload
device / shard / 权重加载不匹配
权重更新错误或直接报错
```

所以 TP 情况下，最关键的不是 CUDA IPC 本身，而是：

```text
payload list index 和 SGLang tp_rank 的稳定对应关系
```

## 9. 一次 TP=4 的例子

假设一个 SGLang engine 使用 4 张 GPU：

```text
gpu_offset = 0
gpu_count = 4
group_ranks = [0, 1, 2, 3]
ipc_gather_src = 0
```

每个 rank 生成：

```text
rank 0 -> payload_0
rank 1 -> payload_1
rank 2 -> payload_2
rank 3 -> payload_3
```

gather 到 rank 0 后：

```python
serialized_named_tensors = [
    [payload_0],
    [payload_1],
    [payload_2],
    [payload_3],
]
```

rank 0 取 dtype bucket 0，组装：

```python
[
    payload_0,
    payload_1,
    payload_2,
    payload_3,
]
```

发给 SGLang：

```python
engine.update_weights_from_tensor.remote(
    serialized_named_tensors=[
        payload_0,
        payload_1,
        payload_2,
        payload_3,
    ],
    load_format="flattened_bucket",
)
```

SGLang 侧：

```text
worker tp_rank=0 -> payload_0
worker tp_rank=1 -> payload_1
worker tp_rank=2 -> payload_2
worker tp_rank=3 -> payload_3
```

然后每个 worker 独立完成：

```text
deserialize -> reconstruct -> load_weights
```

## 10. TP 情况下和单 worker 的区别


| 维度                 | 单 worker      | SGLang TP > 1                 |
| ------------------ | ------------- | ----------------------------- |
| SGLang worker 数    | 1             | N                             |
| slime gather group | size 1        | size N                        |
| 每次 Ray update 调用   | list 长度 1     | list 长度 N                     |
| SGLang 取 payload   | index 0       | index = `self.tp_rank`        |
| 关键风险               | producer 生命周期 | payload list 顺序必须和 tp_rank 对齐 |


## 11. 和训练侧 TP 的关系

本文故意不展开训练侧 TP。

对本文要理解的 IPC 发送逻辑来说，只需要看最终进入 `_send_to_colocated_engine` 的数据：

```python
hf_named_tensors
```

以及每个 colocated rank 生成的 serialized payload。

训练侧如何 all-gather、如何转换、是否和 SGLang TP 一样，不是本文重点。

本文关注的是：

```text
SGLang engine 有 N 个 worker
slime 侧收集 N 个 payload
SGLang worker i 取 payload i
```

## 12. 是否每个 worker 只收到自己的 1/N shard

当前 slime + SGLang colocated tensor path 通常不是：

```text
worker 0 只收到 shard 0
worker 1 只收到 shard 1
worker 2 只收到 shard 2
...
```

更接近：

```text
slime 侧准备 HF/SGLang 形态的 full weight bucket
每个 SGLang worker 收到一份可用于 load_weights 的 payload
SGLang worker 在 load_weights / weight_loader 内部根据自己的 rank 加载需要的部分
```

也就是说，SGLang worker 侧通常会经历：

```text
deserialize full-ish bucket
reconstruct named tensors
model.load_weights(...)
weight_loader 根据当前 worker rank / world size 切分或选择本地需要的 shard
```

所以如果 SGLang TP=4，常见心智模型是：

```text
worker0 收到 full bucket -> load shard0
worker1 收到 full bucket -> load shard1
worker2 收到 full bucket -> load shard2
worker3 收到 full bucket -> load shard3
```

而不是 slime 预先把每个权重切成 4 份，只给每个 worker 发 1/4。

这个设计的主要原因是复用 SGLang 已有的模型加载逻辑。

SGLang 的 `model.load_weights(...)` 和各类 `weight_loader` 已经知道：

```text
哪些权重需要按哪个维度切
哪些权重需要复制到所有 rank
哪些权重是 merged QKV / gate_up
哪些模型有特殊命名或特殊加载规则
量化权重、MoE 权重、embedding / lm_head 等如何处理
```

如果 slime 侧提前按 SGLang TP 预切 shard，就需要在 slime 里复刻这些模型特定规则：

```text
slime 必须知道 SGLang 每种模型的 weight_loader 逻辑
slime 必须知道每个权重应该切哪一维
slime 必须处理 merged weights / quantized weights / MoE 等特殊情况
SGLang loader 变了，slime 也要同步维护
```

当前设计选择：

```text
slime 只负责把 HF/SGLang named tensors 通过 CUDA IPC 发过去
SGLang 继续负责“如何把这些 named tensors 加载到本 rank 参数里”
```

代价是：

```text
多个 SGLang workers 可能重复接收 / reconstruct full-ish bucket
每个 worker 最终只使用其中一部分
GPU flatten buffer、IPC payload、reconstruct 阶段存在重复开销
```

收益是：

```text
权重加载规则集中在 SGLang 内部
slime 不需要复刻 SGLang loader
对不同模型和特殊权重更通用
实现路径更简单，出错面更小
```

所以这是一个明确的 tradeoff：

```text
牺牲一部分数据效率
换取 loader 逻辑复用、模型兼容性和实现简单性
```

## 13. 为什么不只发 payload0 给所有 worker

还有一个更具体的问题：

```text
既然每个 SGLang worker 最后都会在 load_weights 阶段切出自己需要的 shard，
为什么不只发送 payload0？sglang 其余 worker 可以采用广播或者重建方式来实现？
为什么要发送 [payload0, payload1, payload2, payload3]？
```

关键原因是：payload 不只是“权重内容描述”，它还绑定了某个 producer GPU 上的 CUDA allocation。

以 SGLang TP=4 为例，共卡映射通常希望是：

```text
slime rank0 / GPU0 -> SGLang worker0 / GPU0
slime rank1 / GPU1 -> SGLang worker1 / GPU1
slime rank2 / GPU2 -> SGLang worker2 / GPU2
slime rank3 / GPU3 -> SGLang worker3 / GPU3
```

当前设计下：

```text
payload0 指向 GPU0 上的 flattened_tensor
payload1 指向 GPU1 上的 flattened_tensor
payload2 指向 GPU2 上的 flattened_tensor
payload3 指向 GPU3 上的 flattened_tensor
```

所以 SGLang 侧是：

```text
worker0 打开 payload0 -> 读 GPU0 本地 IPC tensor
worker1 打开 payload1 -> 读 GPU1 本地 IPC tensor
worker2 打开 payload2 -> 读 GPU2 本地 IPC tensor
worker3 打开 payload3 -> 读 GPU3 本地 IPC tensor
```

如果只把 `payload0` 发给所有 worker，则会变成：

```text
worker0 打开 payload0 -> 读 GPU0 allocation
worker1 打开 payload0 -> 也读 GPU0 allocation
worker2 打开 payload0 -> 也读 GPU0 allocation
worker3 打开 payload0 -> 也读 GPU0 allocation
```

这时 worker1/2/3 的 `loaded_weight` 物理上都来自 GPU0，而它们自己的模型参数在 GPU1/2/3 上。后续加载会变成：

```text
GPU1 worker 从 GPU0 memory copy 自己需要的 shard 到 GPU1 参数
GPU2 worker 从 GPU0 memory copy 自己需要的 shard 到 GPU2 参数
GPU3 worker 从 GPU0 memory copy 自己需要的 shard 到 GPU3 参数
```

这会带来几个问题：

```text
1. GPU0 变成所有 worker 的数据源瓶颈
2. 依赖 GPU 间 P2P / peer access，拓扑不一定支持或性能稳定
3. 跨 GPU copy 比本地 GPU memory copy 更贵
4. device 语义更复杂，payload0 rebuild 出来的 tensor 本质属于 GPU0
```

所以发 TP 份 payload 的主要目的不是为了表达“这 TP 份逻辑内容一定完全不同”，而是为了保持共卡本地访问路径：

```text
SGLang worker i 打开 producer GPU i 上的 IPC tensor
worker i 从本地 GPU memory 读取并加载权重
```

即使训练侧没有按 SGLang TP 切 shard、每张 GPU 上准备的是重复 full-ish bucket，当前设计仍然倾向于：

```text
每张 GPU 各自准备一份本地 producer buffer
每个 SGLang worker 从自己同卡 producer buffer 读
```

这相当于用“每 GPU 一份 producer buffer”的成本，换取：

```text
本地 GPU IPC 访问
避免所有 worker 从 GPU0 集中读数据
避免 SGLang 内部二次 broadcast / scatter
避免复杂 P2P 依赖
```

一句话总结：

```text
不是不能只发 payload0，而是只发 payload0 会让所有 SGLang workers 都从 GPU0 那块 IPC allocation 读；
发 TP 份 payload 是为了让每个 worker 从自己同卡 producer 的 CUDA allocation 读，保持共卡本地路径。
```

## 14. 本节需要掌握的结论

- SGLang TP > 1 后，`serialized_named_tensors` 是一个 list，长度等于该 engine 的 worker 数。
- slime 为每个 colocated engine 建一个 Gloo gather group，group size 等于 engine 的 GPU count。
- 每个 group rank 先 serialize 自己的 payload。
- src rank gather 后，把同一个 dtype bucket 下的 payload 组装成 list 发给 SGLang。
- SGLang worker 用 `self.tp_rank` 从 list 中取自己的 payload。
- TP 情况下最重要的正确性条件是：payload list 顺序必须和 SGLang `tp_rank` 顺序一致。
- 当前路径通常不是 slime 预切 1/N shard 后分别发送，而是让 SGLang worker 在 `load_weights` 阶段按自身 rank 加载所需部分。
- 这样做的主要目的是复用 SGLang 模型加载和切分逻辑，代价是 TP workers 间可能有重复 payload / reconstruct 开销。
- 不只发 payload0 的关键原因是 payload 绑定 producer GPU allocation；发 TP 份 payload 可以让每个 SGLang worker 从同卡 GPU buffer 读取，避免所有 worker 跨 GPU 读 GPU0。

