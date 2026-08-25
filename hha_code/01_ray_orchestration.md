# Ray PG、Rollout Engine 与 SGLang 并行

---------------------------------------------------------------------
## GLM5.3 
所有这些工作都运行在 slime 之上。slime 是我们面向大规模强化学习的开源后训练框架，训练侧采用 Megatron，rollout 侧采用 SGLang。它将训练、rollout 和数据缓冲区组织在同一条数据流中，使数学、代码、沙箱、验证器以及长时程智能体环境都能够以“数据生成模块”的形式接入，而不需要修改训练循环。正因如此，从 GLM-5.2 到 GLM-5.3，我们可以持续加入新的环境，而无须每次都重新构建整套训练系统。

在 GLM-5.3 的研发过程中，我们继续从算法和系统两个方向完善 slime。在算法方面，我们增加了一系列面向强化学习研究的能力，包括 top-p mask、top-k OPD 和全词表 OPD，以及一些用于提升训练与 rollout 一致性的配置，例如 R3 风格的配置方案，以及训练
路径和 rollout 路径之间完整的数值对齐。这些能力使我们能够更精细地控制采样过程、训练过程和 teacher 信号，同时也能快速开展受控对比实验。在训练与 rollout 一致性评估中，二者 log probability（logprob）的平均差异被控制在 (10^{-7}) 量级，相较此前的配置降低了超过 99.99%。

我们也持续优化大规模强化学习的资源利用率和系统吞吐量。现在，本地存储可以作为额外的一层缓存，通过分层方式保存原本需要常驻主机内存的模型状态和数据。这项能力对多 teacher OPD 尤其重要：通过在训练侧动态切换并预取 teacher，可以使用多个teacher，而不必为每个 teacher 单独部署一个长期运行的推理服务；由此只会引入有限的额外开销，同时能显著降低资源消耗。

对于智能体和异步任务，我们改进了 router 与 slime 之间的联合调度与负载均衡，使长度和完成时间差异很大的 rollout 请求能够更加充分地利用推理资源。我们还加入了能够感知工作负载的启发式策略，根据不同 rollout 环境的特征，自动推导以吞吐量为目标的系统配置，包括 prefill 与 decode 的资源比例、并发设置，以及其他影响吞吐量的关键参数。

得益于这些系统级优化，在长时程代码强化学习任务中，端到端强化学习训练吞吐量提升了超过 2.3 倍，使我们能够以更高的效率，将训练扩展到更长的轨迹和更加复杂的环境。

综合来看，这些能力为我们带来了更高的实验灵活性、更低的资源成本和更高的系统吞吐量，而这正是持续扩大强化学习训练规模能够真正落地的基础。

---------------------------------------------------------------------------

这篇记录目前已经聊过的核心内容。它不是 Ray 或 SGLang 的入门笔记，重点是解释 slime 里资源编排相关的几个容易误解的问题。

相关代码：

- `slime/ray/placement_group.py`
- `slime/ray/rollout.py`
- `slime/utils/http_utils.py`
- `slime/backends/sglang_utils/arguments.py`
- `slime/backends/sglang_utils/sglang_engine.py`
- `../sglang/python/sglang/srt/server_args.py`
- `../sglang/python/sglang/srt/distributed/parallel_state.py`

## Placement Group 的实际语义

slime 在 `create_placement_groups()` 里创建 Ray placement group。核心逻辑是：

```python
bundles = [{"GPU": 1, "CPU": 1} for _ in range(num_gpus)]
pg = placement_group(bundles, strategy="PACK")
```

这里的 bundle 不是“独占容器”，而是 PG 里的一个资源池。一个 actor/task 绑定到某个 bundle 后，只表示它的资源需求要从这个 bundle 里扣。多个 actor 可以绑定到同一个 bundle，只要它们声明的资源加起来不超过 bundle 的资源容量。

因此，`placement_group_bundle_index=i` 不等价于“这个 actor 拥有整个 bundle”。它只是约束调度位置和资源扣减来源。

## 为什么 InfoActor 必须写 num_gpus=1

slime 里有这段：

```python
@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]
```

这个 `num_gpus=1` 是必要的。原因是：actor 绑定到包含 GPU 的 bundle，并不会自动获得 GPU。只有 actor 自己声明了 `num_gpus=1`，Ray 才会给它分配 GPU，并让 `ray.get_gpu_ids()` 返回非空结果。

我们用实验确认过：

- `PlainActor` 绑定到 `{"CPU": 1, "GPU": 1}` bundle，但不声明 GPU：`ray.get_gpu_ids()` 返回 `[]`。
- 同一个 bundle 里还能再启动一个 `GPUActor(num_gpus=1, num_cpus=0)`，并拿到 GPU ID。
- 第二个 `GPUActor(num_gpus=1)` 会 pending，因为该 bundle 只有 1 个 GPU。
- `ZeroResourceActor(num_cpus=0, num_gpus=0)` 也能和 GPU actor 共享同一个 bundle。

所以 slime 用 `InfoActor(num_gpus=1)` 的目的，是临时占住每个 bundle 的 GPU，从 Ray 那里问出实际的 `(node_ip, gpu_id)`，然后排序并记录：

```python
pg_reordered_bundle_indices
pg_reordered_gpu_ids
```

随后这些 InfoActor 会被 kill 掉，不参与后续训练或 rollout。

## bundle_group_*: 1000.0 是什么

Ray 在 PG 内部会生成一些 group resource，例如：

```text
bundle_group_0_<pg_id>: 1000.0
bundle_group_<pg_id>: 1000.0
CPU_group_0_<pg_id>: 1.0
GPU_group_0_<pg_id>: 1.0
```

`CPU_group_*`、`GPU_group_*` 比较直观，是 bundle 里真实资源的 group 版本。

`bundle_group_*` 是 Ray 内部的 bundle marker，不是硬件资源。Ray 源码里会给每个 bundle marker 一个 `1000` 的容量；绑定到 PG 的 task/actor 会额外带上 `0.001` 的 bundle constraint。这样 Ray 可以表达“这个任务必须属于这个 PG/bundle”，但这个 marker 基本不会成为真实瓶颈：

```text
1000 / 0.001 = 1,000,000
```

所以看到 `bundle_group_1_<pg_id>: 1000.0` 时，不要理解成 1000 个 CPU/GPU，也不要理解成 1000 个 actor 容量。它只是 Ray 的调度标记资源。

## PACK 是否重要

`placement_group(bundles, strategy="PACK")` 的 `PACK` 仍然有意义，但它不是“GPU=1 就自动分散”的反面。

`{"GPU": 1}` 只表示每个 bundle 需要 1 张 GPU。是否集中到少数节点，取决于 strategy 和集群剩余资源：

- `PACK`：尽量把 bundles 放到更少节点上。
- `SPREAD`：尽量分散。

如果集群刚好只有你要的那些 GPU，比如总共 16 张卡、正好创建 16 个 `{"GPU": 1}` bundle，那么无论 PACK/SPREAD，最终都必须占满这些卡，差异可能不明显。但如果集群 GPU 更多，PACK 会影响节点选择。

## slime 的 PG 排序

slime 创建 PG 后，会用 `InfoActor` 逐 bundle 获取 `(node_ip, gpu_id)`，再按 node IP 和 GPU ID 排序。

这一步主要是为了得到稳定的逻辑 GPU 顺序，后续 actor rank、rollout engine rank、base GPU ID 都依赖这个顺序。

我们也对比过 XTuner 的做法。XTuner 会按 Ray `node_id` 分组，再按 local rank 排序。它不会产生 `A0, B0, A1, B1` 这种交错 rank；更可能的问题只是节点 block 顺序不够显式稳定，比如这次是 A 节点在前，下次是 B 节点在前。这个问题更多是可观测性和复现层面的，不一定影响正确性。

## RAY_USE_UVLOOP=0

slime 在 `slime/ray/utils.py` 里默认设置：

```python
RAY_DEFAULT_ENV_VARS = {
    "RAY_USE_UVLOOP": "0",
}
```

这个注释说 Ray 的 uvloop integration 可能导致 intermittent async actor issues。我们查了 Ray 源码和 commit 记录，相关背景更具体：Ray worker 默认在安装了 uvloop 时可能使用 uvloop；某些高并发 async networking 场景里，uvloop 曾出现过 fd/socket 相关 race，例如 `File descriptor ... is used by transport`。

因此 slime 这里更像是保守稳定性开关。你没遇到问题也正常，因为它通常需要同时满足：

- 环境里装了 uvloop；
- 工作负载是高并发 async networking；
- 命中特定 socket/timeout/cancellation race；
- Ray/uvloop 版本组合正好有这个问题。

更准确的理解是：这不是普通 Ray async actor 必然有 bug，而是 slime 为 rollout/SGLang 这种高并发 HTTP 场景规避潜在事件循环问题。

## Rollout Engine 数量怎么算

`init_http_client(args)` 会先算：

```python
num_engines = get_rollout_num_engines(args)
```

默认计算逻辑在 `slime/utils/http_utils.py`：

```python
if args.rollout_num_engines is not None:
    return int(args.rollout_num_engines)

return max(1, args.rollout_num_gpus // args.rollout_num_gpus_per_engine)
```

也就是：

```text
num_engines = rollout_num_gpus / rollout_num_gpus_per_engine
```

这里的 engine 是 HTTP 层 router 后面的 SGLang engine endpoint，不是 GPU 数，也不是 EP rank 数。

### Qwen3-235B-A22B

`scripts/run-qwen3-235B-A22B.sh` 里：

```bash
--rollout-num-gpus 64
--rollout-num-gpus-per-engine 32
--sglang-enable-dp-attention
--sglang-dp-size 4
--sglang-ep-size 32
```

所以：

```text
num_engines = 64 // 32 = 2
```

含义是：2 个 SGLang rollout engine，每个 engine 占 32 张 GPU。`sglang_ep_size=32` 是每个 engine 内部的 MoE expert parallel 配置，不会让 engine 数变成 32。

### Qwen3-30B-A3B

`scripts/run-qwen3-30B-A3B.sh` 里：

```bash
--actor-num-nodes 1
--actor-num-gpus-per-node 8
--colocate
--rollout-num-gpus-per-engine 8
```

脚本没有显式写 `--rollout-num-gpus`。但参数后处理里，如果开启 `--colocate` 且 `rollout_num_gpus is None`，会设置：

```text
rollout_num_gpus = actor_num_nodes * actor_num_gpus_per_node = 1 * 8 = 8
```

所以：

```text
num_engines = 8 // 8 = 1
```

即 1 个 8-GPU rollout engine。

## Qwen3-30B 的 SGLang 内部并行

30B 脚本不写 TP/EP，但并不是没有并行。slime 会把：

```bash
--rollout-num-gpus-per-engine 8
```

转换成 SGLang 的：

```text
tp_size = 8
```

因为 `slime/backends/sglang_utils/sglang_engine.py` 启动 SGLang 时传入：

```python
"tp_size": _gpus_per_engine // args.sglang_pp_size,
"dp_size": args.sglang_dp_size,
"pp_size": args.sglang_pp_size,
"ep_size": args.sglang_ep_size,
```

SGLang 源码 `ServerArgs` 的默认值是：

```python
tp_size = 1
pp_size = 1
dp_size = 1
ep_size = 1
```

但 slime 已经显式传了 `tp_size=8`。因此 30B 的有效并行是：

```text
num_engines = 1
tp_size = 8
pp_size = 1
dp_size = 1
ep_size = 1
moe_dp_size = 1
moe_tp_size = 8
```

SGLang 的 MoE 并行组里有：

```python
moe_tp_size = tensor_model_parallel_size // moe_ep_size // moe_dp_size
```

所以 30B 默认不是 expert parallel，而是 MoE MLP 也走 TP 切分。可以理解为：每个 rank 都覆盖所有 experts 的 shard；不是每个 rank 只负责一部分 experts。

如果开 EP，例如 `ep_size=8`，则更接近：

```text
moe_ep_size = 8
moe_tp_size = 1
```

每张卡负责一部分 experts，需要 token dispatch/all-to-all。30B 这个脚本选择 TP-only，更简单，也足够跑。

235B 脚本开了 `--sglang-moe-a2a-backend deepep`。SGLang 源码里当 `moe_a2a_backend == "deepep"` 时，会把：

```python
ep_size = tp_size
```

所以 235B 是更复杂的 DP attention + EP/DeepEP 路线。

## use_distributed_post 是什么

`init_http_client()` 里还有：

```python
if args.use_distributed_post:
    _init_ray_distributed_post(args)
    _distributed_post_enabled = True
```

这个功能默认关闭。当前 repo 里没有任何脚本显式开启 `--use-distributed-post`。

不开时：

```text
RolloutManager 本进程的 httpx.AsyncClient 直接发 /generate POST
```

开启时：

```text
RolloutManager 把 POST 请求 round-robin 派给一批 Ray _HttpPosterActor
这些 actor 再用自己的 httpx.AsyncClient 发请求
```

它会在每个 alive Ray node 上创建 `args.num_gpus_per_node` 个 poster actor，并用 `NodeAffinitySchedulingStrategy` 固定到对应节点。每个 actor 只申请 `num_cpus=0.001`，不占 GPU。

它不改变 SGLang engine 数量，不改变 router 策略，也不是分布式推理。它只是把客户端侧 HTTP POST 压力从 RolloutManager 单点分散到多个 Ray actor 上。

适用场景：

- rollout 请求并发很高；
- 单个 RolloutManager 进程维护太多 HTTP 连接；
- 单节点网络出口或 event loop 成为瓶颈；
- 多节点场景下希望 POST 请求从多个节点发出。

普通单机或并发不高时，不开也没问题。

## 当前结论

目前这部分可以先形成几个稳定心智模型：

- PG bundle 是资源池，不是 actor 独占容器。
- 绑定 GPU bundle 不等于获得 GPU，actor 必须声明 `num_gpus`。
- PG ready 后普通 CPU/GPU 资源减少，是因为 PG reserve 了资源。
- `bundle_group_*: 1000.0` 是 Ray 内部 marker，不是真实硬件资源。
- slime 的 `rollout_num_gpus_per_engine` 在没有 PP 时基本等价于 SGLang `tp_size`。
- EP 是 SGLang engine 内部 MoE 并行策略，不影响 slime 的 engine 数量。
- `use_distributed_post` 是 HTTP client 侧的高并发优化，当前默认没启用。

