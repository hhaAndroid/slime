# PR #2262：GLM-5 Megatron 训练与 SGLang Rollout 数值对齐总览

> 本文分析 THUDM/slime PR [#2262](https://github.com/THUDM/slime/pull/2262)：
> `feat(glm5): align Megatron DeepEP training with SGLang rollout`。
>
> 这是一份总览和后续阅读索引。本文先建立整体认知，不试图一次讲透所有 kernel；
> DeepGEMM、DeepEP/MoE、DSA 和 FP8 KV cache 可以在后续文档中分别深入。

## 1. 实现 GLM-5 训推一致，需要对齐哪些核心模块

先给出整篇文档最重要的总表。要让：

```text
SGLang rollout logprob
          ≈
Megatron train logprob
```

不能只修改某一个 GEMM，也不能只开启 deterministic mode。需要同时对齐下面六个核心模块。

| 核心模块 | 必须对齐的内容 | 不对齐会发生什么 |
|---|---|---|
| 1. 全局确定性与 batch invariant | GEMM、RMS reduction、BMM、FP32 matmul、log-softmax 的 batch-independent 执行 | SGLang decode 和 Megatron packed training 因 batch shape 不同产生不同舍入结果 |
| 2. Dense/MLP/Router DeepGEMM | block-FP8 量化、scale、padding、RMSNorm、residual、SwiGLU、FP32 router GEMM | 即使权重相同，每层 hidden state 仍会逐步漂移 |
| 3. Routed MoE 与 DeepEP | top-k expert ID、slot 顺序、dispatch layout、expert padding、route weight、owner gather/combine 顺序 | 微小误差跨过 top-k 边界，或相同 experts 因归并顺序不同得到不同输出 |
| 4. GLM-5 DSA sparse attention | fused Q-RMSNorm、indexer projection、FP8 logits、RoPE、DSA top-k、FlashMLA、head padding | attention 选中的 KV page 或 attention 输出不同，后续所有层随之发散 |
| 5. FP8 KV cache | prefill/decode 的 KV 量化误差、packed cache layout、选中 page 的 gather/dequantize | prefill 使用新鲜 BF16 KV、decode 使用 FP8 历史 KV，导致同一上下文的 attention 不一致 |
| 6. 训练集成与验证 | aligned forward hook、可训练 backward、权重同步、expert 边界、逐层和 logprob gate | 单算子看似一致，但真实 rollout/train 流程未实际使用对齐路径，或无法证明最终结果 |

完整依赖关系是：

```text
共享 deterministic / batch-invariant 环境
                     │
                     ▼
residual + RMSNorm + dense/indexer FP8 Linear
                     │
                     ▼
DSA indexer + RoPE + sparse attention + FP8 KV cache
                     │
                     ▼
FP32 router + deterministic top-k
                     │
                     ▼
DeepEP route-preserving dispatch
                     │
                     ▼
routed/shared expert DeepGEMM
                     │
                     ▼
ordered FP32 probability multiply / owner combine
                     │
                     ▼
final RMSNorm + LM head + log-softmax
                     │
                     ▼
train logprob ≈ rollout logprob
```

### 1.1 全局 deterministic 与 batch-invariant 执行环境

首先要让两侧共享确定性数值基础：

- 关闭不确定性 Transformer Engine/CUDA 算法；
- 固定可能影响 reduction/collective 顺序的运行配置；
- 开启 DeepGEMM batch invariant；
- 对 RMS reduction、BMM、FP32 matmul、log-softmax 等启用 SGLang batch-invariant ops；
- 让 SGLang engine 与 Megatron actor 继承同一组 alignment 环境变量。

这一层只解决“同一逻辑 token 不应因为 batch 大小、pack 或调度形态而改变结果”，
它是必要条件，但还没有对齐模型计算图的具体舍入边界。

### 1.2 DeepGEMM、RMSNorm、residual、SwiGLU 与 router

第二层对齐 Transformer block 中的大部分连续计算：

- dense、attention、indexer 和 shared expert Linear 使用 SGLang-compatible block-FP8 forward；
- routed expert 使用 SGLang-compatible grouped block-FP8 forward；
- 对齐 activation/weight quantization、block scale 和 expert `M` padding；
- 对齐 Hopper FP32 scale 和 Blackwell UE8M0 scale；
- 保留 SGLang 相同的 FP32 residual/RMSNorm 舍入边界；
- 使用相同的 fused SwiGLU forward；
- router 使用相同的 batch-invariant FP32 GEMM；
- shared expert 和 residual 按相同顺序累加。

训练不能直接使用纯 inference kernel 后就结束，所以还要为这些 forward 提供：

```text
BF16 GEMM dgrad / wgrad
+ RMSNorm、SwiGLU、RoPE 等解析 backward
```

这形成了该 PR 的基本模式：`SGLang forward semantics + Megatron trainable backward`。

### 1.3 Routed MoE 与 DeepEP 通信/归并

MoE 是整个对齐链路最敏感的离散部分，需要同时解决两类问题。

Router 侧：

- router 输入 hidden state 必须先对齐；
- router score、sigmoid/scaling/expert bias 必须一致；
- top-k expert 集合必须一致；
- `sorted=False` 的原始 top-k slot 顺序也必须一致。

DeepEP 侧：

- Megatron normal DeepEP 与 SGLang low-latency DeepEP 的 route 表达要建立映射；
- 每个 `(token, top-k slot)` route 不能因 rank 去重而丢失；
- received expert ID、FP32 weight、source token 和 receive order 需要验证；
- expert-major padding/layout 必须与 SGLang 相同；
- expert 输出必须按 SGLang top-k 顺序做 FP32 owner gather/combine；
- backward 也要固定累加顺序，不能依赖 nondeterministic atomics。

这部分不依赖 rollout routing replay 强制 expert ID；目标是让 Megatron 自己重算 router 后，
自然得到与 SGLang 一致的 route 和输出。

### 1.4 GLM-5 DSA indexer 与 sparse attention

DSA 决定 attention 实际读取哪些 KV page，因此它也是一个离散分支。需要对齐：

- fused Q-RMSNorm 的真实输入；
- indexer Q/K projections；
- indexer activation FP8 quantization 和 logits；
- head weight 与 RoPE；
- deterministic sparse top-k；
- FlashMLA sparse attention forward；
- Hopper/Blackwell 不同 head padding 行为。

如果 DSA top-k 不同，即使后面的 sparse attention kernel 完全相同，两边读取的 K/V 已经不是同一批数据。

### 1.5 FP8 KV cache 的 prefill/decode 数值语义

SGLang rollout 的 decode 会读取 FP8 paged KV cache，而 Megatron teacher-forcing 通常直接使用
新鲜 BF16 KV。为对齐这两条路径，需要：

- 让训练 forward 中的 KV 也经过 FP8-E4M3 quantize/dequantize；
- 使用 straight-through estimator 保留训练梯度；
- 保证 FP8 cache block size、scale 和 packed layout 与 SGLang 相同；
- sparse attention 只 gather 并反量化 DSA top-k 选中的 pages；
- 让 SGLang prefill 的新鲜 KV 也体现与 decode cache 相同的量化误差。

因此“开启 batch invariant”不能单独保证 prefill/decode 一致，KV cache 数值表示也必须对齐。

### 1.6 训练接入、在线权重同步和验证闭环

最后要保证上述对齐路径真正进入 slime 的训练流程：

- 在 Megatron 计算 train logprob 前安装 aligned forward；
- 在正式 train step/recompute 前再次保证 hook 生效；
- 主模型参数执行真实 backward，只有辅助 DSA indexer 按 gate 配置冻结；
- Megatron 权重通过 NCCL full update 同步给 SGLang；
- 权重传输计划保留单个 expert 的参数边界；
- Pipeline Parallel 和 routing capture 不完整时 fail fast；
- 通过单算子、route/layout、逐层 hidden state 和端到端 logprob 四级测试验证。

这六块需要共同成立。它们不是六种可任选的优化，而是从输入 hidden 到最终 logprob 的一条连续正确性链路。

## 2. 这个 PR 的文件级改动地图

先不进入实现细节，PR #2262 的整体改动可以归纳成七组：

```text
1. 建立 Megatron ↔ SGLang 共享的确定性运行环境
2. 用 SGLang-compatible DeepGEMM 替换 Megatron 关键 forward
3. 对齐 routed/shared expert、router 和 DeepEP route/combine
4. 对齐 GLM-5 DSA indexer 与 sparse attention
5. 让训练侧模拟 SGLang FP8 KV cache 数值误差
6. 把对齐能力接进训练、权重同步和参数系统
7. 增加单算子、逐层和真实 train/rollout 端到端 gate
```

合入提交为 `a74ae3a0`，规模是：

```text
45 个文件变化
新增 12114 行
删除 42 行
```

这解释了为什么它很难一次读完：它不是给某个 kernel 打一个补丁，而是同时修改
Megatron、SGLang、DeepGEMM、DeepEP、GLM-5 model plugin、训练集成和 CI。

### 2.1 新增统一的 alignment 模块

新增目录：

```text
slime/backends/megatron_utils/alignment/
├── env.py
├── deepgemm_forward.py
├── deepgemm_moe_forward.py
├── deterministic_route_kernels.py
└── layerwise_alignment.py
```

职责分别是：

| 文件 | 主要改动 |
|---|---|
| `env.py` | 集中管理 Megatron 与 SGLang 必须共享的确定性和数值对齐环境变量 |
| `deepgemm_forward.py` | 对齐 dense、attention、indexer、shared expert Linear，以及 RMSNorm、SwiGLU、router forward |
| `deepgemm_moe_forward.py` | 对齐 routed MoE 的 FP8 forward、DeepEP route layout、combine 顺序和训练 backward |
| `deterministic_route_kernels.py` | 提供固定顺序的 route scatter/gather backward，避免原子累加不确定性 |
| `layerwise_alignment.py` | dump Megatron 逐层 hidden state，与 SGLang 输出做精确比较 |

这是该 PR 的主体，其中 `deepgemm_moe_forward.py` 单文件新增约 3184 行。

### 2.2 给 Megatron 增加 SGLang-compatible forward

PR 把选定的 Megatron forward 替换成：

```text
SGLang-compatible forward
+ Megatron 可训练 backward
```

覆盖：

- dense/attention/indexer Linear 的 block-FP8 DeepGEMM；
- routed expert 和 shared expert 的 grouped DeepGEMM；
- FP32 batch-invariant router GEMM；
- SGLang-compatible RMSNorm、residual 和 SwiGLU；
- activation quantization、expert padding 和 router probability multiply 顺序；
- Hopper FP32 scale 与 Blackwell UE8M0 scale。

### 2.3 建立 Megatron normal DeepEP 到 SGLang low-latency DeepEP 的对齐桥

PR 没有把 Megatron training 直接改成 low-latency DeepEP，而是保留：

```text
Megatron training：normal DeepEP
SGLang rollout：low-latency DeepEP
```

然后新增一层 route-preserving bridge：

- 捕获原始 top-k slot 顺序；
- 通过额外小 metadata dispatch 保存每个 `(token, slot)` route；
- 校验 expert ID、route weight、source token 和 receive order；
- 构造 SGLang-compatible expert-major layout；
- 使用固定顺序的 FP32 token-owner gather；
- 为 scatter/gather 补充确定性 backward；
- 对过大的 low-latency prefill dispatch 做 staging；
- 统一 DeepEP 两侧 FP8 quantization。

普通 Megatron all-to-all 的对齐桥被移除，当前 MoE alignment 只支持 DeepEP。

### 2.4 修改 GLM-5 DSA 和 FP8 KV cache 路径

GLM-5 plugin 和 SGLang patch 中增加了：

- fused Q-RMSNorm 输入对齐；
- indexer Q/K projection 和 FP8 logits 对齐；
- RoPE、head weight、deterministic top-k 对齐；
- FlashMLA sparse forward；
- Blackwell attention head padding；
- FP8-E4M3 KV cache 的 straight-through QAT；
- sparse attention 只 gather/dequantize 被选中的 FP8 pages；
- `--freeze-indexer` 和未知 indexer layout 的 fail-fast 检查。

主要修改文件：

- [`slime_plugins/models/glm5/glm5.py`](../slime_plugins/models/glm5/glm5.py)
- [`slime_plugins/models/glm5/ops/indexer.py`](../slime_plugins/models/glm5/ops/indexer.py)
- [`slime_plugins/models/glm5/ops/sparse_mla.py`](../slime_plugins/models/glm5/ops/sparse_mla.py)
- [`docker/patch/latest/sglang-deterministic.patch`](../docker/patch/latest/sglang-deterministic.patch)

### 2.5 修改训练、路由和权重同步集成

PR 还修改了外围控制流：

- 新增选择 aligned dense/MoE layer 和 module 的参数；
- 在计算 train logprob 和 train step 前安装 alignment hooks；
- 区分“捕获当前 forward 的 ordered top-k”与 rollout routing replay；
- 增强 R3 shape 和 Pipeline Parallel 缺层校验；
- 在线权重同步计划按 expert 保持边界，避免不同 expert 参数混装；
- 规范化 SGLang MoE-DP 参数；
- 增加 train/rollout logprob diff CI 阈值；
- 将 GLM-5 FP32 router GEMM JIT kernel 放入 slime 仓库。

主要入口：

- [`slime/utils/routing_replay.py`](../slime/utils/routing_replay.py)
- [`slime/backends/megatron_utils/update_weight/expert_routing.py`](../slime/backends/megatron_utils/update_weight/expert_routing.py)
- [`slime/backends/megatron_utils/model_provider.py`](../slime/backends/megatron_utils/model_provider.py)
- [`slime/backends/sglang_utils/jit_kernels`](../slime/backends/sglang_utils/jit_kernels)

### 2.6 增加 Megatron 和 SGLang patch

PR 新增两组关键 patch：

| Patch | 作用 |
|---|---|
| [`megatron-sglang-aligned.patch`](../docker/patch/latest/megatron-sglang-aligned.patch) | 修改 Megatron residual/RMSNorm 等边界，使训练 forward 能复刻 SGLang 舍入行为 |
| [`sglang-deterministic.patch`](../docker/patch/latest/sglang-deterministic.patch) | 增加 batch-invariant router、DeepGEMM 开关、DSA/FP8 KV cache 和确定性执行路径 |

Dockerfile 同时调整了这些 patch 和依赖的应用顺序，确保对应 DeepGEMM、DeepEP、Megatron 和
SGLang 版本组合能工作。

### 2.7 增加三层验证体系

PR 增加约 100 个 focused tests，整体分成三层：

```text
第一层：单算子和辅助逻辑
  DeepGEMM、MoE、route kernel、indexer、freeze、参数校验

第二层：逐层 hidden state
  dump SGLang/Megatron layer outputs，要求指定 gate 中 max hidden diff = 0

第三层：真实端到端
  6-layer GLM-5 + 8 GPU + SGLang rollout + NCCL weight update
  + Megatron train forward/backward + logprob threshold
```

主要测试：

- [`tests/test_deepgemm_forward.py`](../tests/test_deepgemm_forward.py)
- [`tests/test_deepgemm_moe_forward.py`](../tests/test_deepgemm_moe_forward.py)
- [`tests/test_glm52_6layer_deterministic_e2e.py`](../tests/test_glm52_6layer_deterministic_e2e.py)
- [`tests/test_glm52_layerwise_zero_e2e.py`](../tests/test_glm52_layerwise_zero_e2e.py)

完成这张改动地图后，下面再解释这些改动为什么必要以及如何协作。

## 3. 核心结论

PR #2262 建立了一条 GLM-5 专用的数值对齐路径：

```text
Megatron 训练 forward
        尽可能复刻
SGLang rollout forward

同时：
Megatron 仍然执行可训练的 backward 和 optimizer step
```

它解决的不是普通的随机确定性问题，而是：

```text
相同权重 + 相同 token

SGLang 在动态 prefill/decode 环境计算的 rollout logprob
                          与
Megatron 在 packed teacher-forcing 环境计算的 train logprob

如何做到极其接近？
```

PR 的核心设计可以概括为：

```text
forward：尽可能采用 SGLang 的数值语义和 batch-invariant 算子
backward：为这些 forward 补充 BF16 GEMM 或解析梯度
```

## 4. 为什么训练侧和 rollout 侧会不一致

slime 的 RL 主流程是：

```text
Megatron 当前权重
       │
       │ 在线同步
       ▼
SGLang rollout
       │
       ├─ 生成 response tokens
       └─ 记录 rollout logprobs
                    │
                    ▼
Megatron 对同一串 tokens 做 teacher-forcing forward
                    │
                    ├─ 重新计算 train logprobs
                    ├─ 计算 RL loss
                    └─ backward + optimizer step
```

从数学定义看，两边对同一 token 的条件概率应该相同：

```text
log P(x_t | x_0, ..., x_{t-1})
```

但实际计算形态非常不同。

### 4.1 SGLang rollout

```text
动态 continuous batching
prefill + 逐 token decode
paged KV cache
可能使用 FP8 KV cache
SGLang DeepGEMM
DeepEP low-latency 模式
DSA sparse attention
```

### 4.2 Megatron training

```text
多个 sequence 被 pack
一次 teacher-forcing forward
训练用 attention 和 Transformer Engine 算子
Megatron DeepEP normal 模式
需要保留完整 backward graph
```

即使数学公式相同，下列差异也会改变浮点结果：

- GEMM 的 batch `M` 大小、tile 和 reduction 顺序；
- FP8 activation/weight 的 scale、padding 和量化边界；
- residual add 与 RMSNorm 之间的 cast 位置；
- SwiGLU 是否融合以及中间值精度；
- router GEMM 的精度和 top-k 顺序；
- DeepEP dispatch、接收和 combine 顺序；
- prefill 的新鲜 KV 与 decode 的 paged FP8 KV cache；
- DSA top-k、RoPE 和 sparse attention kernel；
- LM head 与 log-softmax 的 batch shape。

这些误差还会逐层传播。对于 MoE，微小连续误差甚至可能跨过 top-k 边界，突然选中另一个 expert。

## 5. 三个容易混淆的概念

理解该 PR 最重要的是区分以下三层。

### 5.1 Deterministic：重复执行稳定

普通确定性只保证：

```text
同一套实现 + 相同输入 shape + 相同调度条件
重复执行得到相同结果
```

它不能保证两个不同实现相等：

```text
SGLang 每次稳定得到 A
Megatron 每次稳定得到 B

A 和 B 都 deterministic，但 A != B
```

因此只设置随机种子、关闭非确定性算法远远不够。

### 5.2 Batch invariant：不因 batch 形态变化而改变单 token 结果

同一个 token 可能分别处于：

```text
SGLang decode：M = 32
Megatron packed training：M = 4096
```

普通 FP8 GEMM 可能因为 `M` 不同选择不同 kernel 配置、padding 和累加方式。
batch-invariant kernel 的目标是让相同逻辑行不受这些 batch 变化影响。

这里的 batch invariant 不是只有 DeepGEMM。Megatron 训练进程还会开启 SGLang 的全局
batch-invariant 模式，使 RMS reduction、BMM、FP32 matmul、log-softmax 等也采用兼容路径。

入口：

- [`alignment/env.py`](../slime/backends/megatron_utils/alignment/env.py)
- [`alignment/deepgemm_forward.py`](../slime/backends/megatron_utils/alignment/deepgemm_forward.py)

### 5.3 整图数值对齐：所有关键舍入边界一致

batch invariant 仍然不够。

假如 DeepGEMM 已经完全一致，但输入它的 RMSNorm 结果不同，那么 GEMM 输出仍然不同；
如果 GEMM 输出一致，但后面的 residual、SwiGLU 或 MoE combine 顺序不同，下一层仍会分叉。

因此 PR 实际对齐的是一条完整链路：

```text
residual / RMSNorm
→ FP8 activation quantization
→ dense/indexer DeepGEMM
→ RoPE / DSA top-k / sparse attention
→ FP8 KV cache
→ FP32 router
→ MoE top-k slot
→ DeepEP dispatch layout
→ expert DeepGEMM
→ router probability multiply
→ ordered combine
→ shared expert accumulation
→ final RMSNorm / LM head / log-softmax
```

所以更准确的说法是：

> 该 PR 让 packed Megatron training 和动态 SGLang rollout 对每个逻辑 token 遵守同一套
> forward 数值契约。

## 6. 端到端测试是否真的执行 rollout

测试入口：

- [`tests/test_glm52_6layer_deterministic_e2e.py`](../tests/test_glm52_6layer_deterministic_e2e.py)

它不是加载一份预存 logprob 做离线比较，而是真正调用 `train.py`，完整执行一个 rollout/train step：

```text
启动 Ray
  ↓
启动 8-GPU SGLang engine
  ↓
启动 Megatron actor
  ↓
Megatron → SGLang：NCCL full weight update
  ↓
SGLang 对真实 prompt 做 autoregressive rollout
  ↓
保存生成 token 和 rollout logprob
  ↓
Megatron 对完全相同的 token 做 packed teacher-forcing
  ↓
比较 train logprob 和 rollout logprob
  ↓
完整主模型 backward + stateless Adam step
  ↓
Megatron → SGLang：再次更新权重
```

主循环可以参见 [`train.py`](../train.py)。

### 6.1 为什么假模型和乱码 token 也能测对齐

测试使用 6 层裁剪模型，它不需要具备正常语言能力。假设 SGLang 生成一串无意义 token：

```text
[苹果, 火箭, 9273, <奇怪 token>, ...]
```

Megatron 不会再独立采样，而是对同一串 token 重算：

```text
SGLang：  log P_sglang(x_t | x_<t)
Megatron：log P_megatron(x_t | x_<t)
```

语义是否合理不影响数值一致性测试。该 gate 验证的是计算图，不是语言能力。

真实 rollout 仍有必要，因为它会覆盖：

- 逐 token decode；
- paged FP8 KV cache 的写入和读取；
- 每一步动态变化的 DSA top-k；
- 每一步动态变化的 MoE route；
- DeepEP low-latency dispatch；
- 不同 decode batch 状态下的 DeepGEMM。

### 6.2 这是完整流程，但不是完整生产训练

该测试覆盖真实的单轮端到端数据流，但有意缩小规模：

```text
模型：6 层 GLM-5.2 结构
层分布：3 层 dense + 3 层 MoE
rollout batch：8
n_samples_per_prompt：1
num_rollout：1
并行：EP=8，TP=PP=CP=ETP=1
优化器：stateless Adam
```

因此它能验证当前配置的一次 rollout、训练重算和 backward，不能证明：

- 完整 750B 模型所有层都完全一致；
- 任意 token 序列都满足相同阈值；
- 其他 TP、PP、CP、ETP 组合同样成立；
- 长期多轮训练一定保持相同误差；
- 所有硬件和 kernel 版本都有相同行为。

## 7. DeepGEMM：复刻 SGLang forward，补充训练 backward

主要实现：

- [`deepgemm_forward.py`](../slime/backends/megatron_utils/alignment/deepgemm_forward.py)
- [`deepgemm_moe_forward.py`](../slime/backends/megatron_utils/alignment/deepgemm_moe_forward.py)

选定的 Transformer Engine Linear 会被替换成自定义 autograd：

```text
forward：SGLang 风格 block-FP8 DeepGEMM
backward：显式 BF16 dgrad / wgrad
```

对齐对象包括：

- attention Q/K/V 和 output projection；
- DSA indexer projection；
- dense MLP；
- shared expert；
- routed experts；
- FP32 router GEMM；
- RMSNorm；
- SwiGLU；
- residual accumulation 边界。

MoE forward 被明确调整为 SGLang 的执行顺序：

```text
block-FP8 fc1
→ SwiGLU
→ block-FP8 fc2
→ FP32 router-probability multiply
```

这里不能只替换一个 GEMM。weight/activation FP8 scale、expert `M` padding、Blackwell 的
UE8M0 scale、Hopper 的 FP32 scale以及 shared expert 的相加顺序都属于数值契约的一部分。

该实现还通过 expert grouping、chunked probability multiply 和 dead-buffer reuse 控制峰值显存，
否则显式训练 backward 和中间 FP32 Tensor 会非常昂贵。

## 8. MoE routed experts：为什么是最难的部分

该测试确实使用 routed experts：

```text
后 3 层为 MoE
每层 256 experts
每 token top-k = 8
EP = 8
```

困难在于 top-k 是离散分支。假设第 8、9 名非常接近：

```text
SGLang：  expert 37 = 1.000001，expert 91 = 1.000000
Megatron：expert 37 = 1.000000，expert 91 = 1.000001
```

router 前仅有很小误差，选中的 expert 就可能不同。之后使用的是完全不同的 expert 权重，
hidden state 差异会迅速放大。

要让 MoE 输出一致，至少要连续通过三道门。

### 8.1 Expert ID 一致

需要对齐：

```text
router 输入 hidden
+ FP32 router GEMM
+ sigmoid / scaling / expert bias
+ top-k 实现和 tie-breaking
```

### 8.2 Top-k slot 顺序一致

即使 expert 集合相同，slot 顺序不同也可能改变 BF16 累加结果：

```text
SGLang：  [expert 3, expert 8, expert 12]
Megatron：[expert 8, expert 3, expert 12]
```

SGLang 的确定性 GLM 路径使用 `torch.topk(..., sorted=False)`；Megatron 原路径默认排序。
PR 会捕获与 SGLang 兼容的原始 top-k slot 顺序。

相关入口：[`slime/utils/routing_replay.py`](../slime/utils/routing_replay.py)。

### 8.3 Dispatch、expert forward 和 combine 一致

即使 ID 和 slot 一样，还需要对齐：

- token 发往哪个 EP rank；
- expert-major layout 和 padding；
- FP8 activation/weight quantization；
- expert DeepGEMM；
- router probability 的乘法精度；
- token owner gather 和 combine 顺序；
- shared expert 的相加顺序。

这也是为什么该 PR 的 DeepEP 部分非常大。

## 9. Routed expert 不等于 Routing Replay（R3）

这两个概念必须分开。

### 9.1 当前 gate 使用 routed experts

SGLang 和 Megatron 都各自执行 router：

```text
SGLang hidden  → SGLang router  → top-k experts
Megatron hidden → Megatron router → top-k experts
```

### 9.2 当前 gate 没有使用 rollout routing replay

R3 的做法是：

```text
SGLang rollout 选中 expert IDs
              ↓ 保存
Megatron 训练直接强制复用同一组 IDs
```

该 gate 明确断言没有启用 `--use-rollout-routing-replay`。Megatron 自己重算 router，并依靠前向数值对齐
自然得到一致的 expert selection。

当前 PR 中的 ordered top-k capture 只是保存“本次 Megatron forward 自己算出的 top-k slot 顺序”，
供同一次 MoE dispatch/combine 使用；它不是复用 SGLang rollout 的 expert IDs。

因此，该 gate 无 R3 仍获得极低 logprob diff 是一个较强结果：它意味着测试样本上 router 之前的 hidden、
router、top-k、DeepEP、expert forward 和 combine 整条路径都高度一致。

但 logprob MAE 很小不能形式化证明所有 token、所有层的 expert IDs 逐元素一致。严格验证仍应额外 dump：

```text
[layer, token, topk_slot] expert_id
[layer, token, topk_slot] route_weight
```

然后在 SGLang 与 Megatron 间直接逐元素比较。

## 10. DeepEP：对齐 normal training 与 low-latency rollout

两侧 DeepEP 模式不同：

```text
Megatron training：normal DeepEP
SGLang rollout：low-latency DeepEP
```

它们对 token route 的表达、去重、接收 layout 和 owner combine 顺序不天然一致。

PR 的处理方式是：

1. 主 normal DeepEP dispatch 继续高效传输真实 hidden states；
2. 额外发送很小的 route metadata，把每个 `(token, top-k slot)` 视为独立逻辑 route；
3. 恢复精确的 route-level handle 和 receive order；
4. 校验 expert ID、FP32 route weight、source-token fingerprint 和数量；
5. 构造与 SGLang low-latency 路径兼容的 expert-major layout；
6. 使用 SGLang `ep_gather` 语义按固定 top-k 顺序归并；
7. 为 scatter/gather 编写确定性 backward，避免原子加造成顺序不稳定。

主要实现：

- [`deepgemm_moe_forward.py`](../slime/backends/megatron_utils/alignment/deepgemm_moe_forward.py)
- [`deterministic_route_kernels.py`](../slime/backends/megatron_utils/alignment/deterministic_route_kernels.py)

这里的额外 metadata dispatch 更像一个 correctness bridge。代码注释也将其描述为当前正确性方案，
未来可以考虑把 metadata 融入主 dispatch，减少重复通信和复杂度。

当前 GLM-5 对齐路径只支持 DeepEP MoE backend；普通 Megatron all-to-all 对齐桥已被移除。

## 11. DSA sparse attention 对齐

GLM-5 使用 DSA，attention 侧还需要对齐：

- fused Q RMSNorm 的真实输入；
- indexer Q/K projection；
- indexer activation FP8 quantization；
- FP8 logits；
- head weights；
- RoPE；
- deterministic top-k；
- FlashMLA sparse forward；
- Blackwell attention head padding。

Blackwell head padding 是容易忽略的一点。padding 即使填零，也可能改变 kernel shape 和 BF16 舍入顺序，
所以 Megatron 训练侧必须采用相同 padding 才能复刻 SGLang forward。

训练还提供 `--freeze-indexer`：

```text
冻结辅助 DSA indexer
主模型其余参数正常训练
```

模型结构识别和 fail-fast 检查位于：

- [`model_provider.py`](../slime/backends/megatron_utils/model_provider.py)

## 12. FP8 KV cache：prefill/decode 对齐的额外条件

只开启 batch invariant，不能自动保证 SGLang prefill 和 decode 一致。

两种路径的 KV 可能是：

```text
Prefill 新鲜 KV：BF16
Decode 历史 KV：BF16 → FP8 cache → BF16
```

如果 prefill 直接使用新鲜 BF16 KV，而 decode 使用反量化 KV，即使其他 GEMM 完全一致，attention 输入仍不同。

PR 为训练侧加入 straight-through FP8 KV cache QAT：

```text
forward：BF16 KV → FP8 E4M3 → 反量化 BF16 → sparse attention
backward：梯度穿过量化/反量化节点
```

SGLang sparse attention 侧也支持从 packed FP8 cache 中只 gather DSA 选中的 page，再反量化为 BF16，
避免反量化完整 cache。

相关补丁：

- [`docker/patch/latest/sglang-deterministic.patch`](../docker/patch/latest/sglang-deterministic.patch)
- [`docker/patch/latest/megatron-sglang-aligned.patch`](../docker/patch/latest/megatron-sglang-aligned.patch)

因此，prefill/decode 极其接近需要同时满足：

```text
batch-invariant token-wise ops
+ 相同 attention 语义
+ 相同 KV 量化误差
+ 相同 RoPE / DSA top-k
+ 相同 MoE 路由和 combine
```

不能简化成“只要打开 batch invariant 就必然一致”。

## 13. 训练集成和工程配套

该 PR 不只增加 kernel，还补齐了运行和验证边界：

- 统一 Megatron/SGLang 的 deterministic alignment 环境；
- 提供训练前 logprob hook 和 train-step hook；
- 在线权重传输计划保留 expert 边界；
- 规范化 SGLang MoE-DP 参数；
- Pipeline Parallel 下检查缺失 routing capture；
- 将 GLM-5 router JIT kernel 放在 slime 内维护；
- Docker 按依赖顺序应用 Megatron/SGLang patches；
- 支持 train/rollout logprob diff CI 阈值；
- 增加逐层 hidden state dump 和比较工具。

主要入口：

- [`alignment/env.py`](../slime/backends/megatron_utils/alignment/env.py)
- [`alignment/layerwise_alignment.py`](../slime/backends/megatron_utils/alignment/layerwise_alignment.py)
- [`update_weight/expert_routing.py`](../slime/backends/megatron_utils/update_weight/expert_routing.py)
- [`sglang_utils/jit_kernels`](../slime/backends/sglang_utils/jit_kernels)
- [`compare_glm52_layerwise.py`](../slime/utils/compare_glm52_layerwise.py)

## 14. PR 报告的验证结果及正确解读

PR message 报告：

```text
4096-token train/rollout logprob MAE：约 1.89e-7
decoder layer 0～5：匹配 token 的 hidden state 最大差异为 0
```

这些结果表明指定 6 层、EP=8 对齐配置下，跨框架 forward 已经高度一致。

但需要注意 metric 的边界：

```text
logprob gate：聚合 MAE，不等于所有 token 的 max diff
layerwise zero gate：当前测试的 6 层和匹配 token
```

因此不能直接外推为完整模型、任意并行配置和任意硬件上都 bitwise identical。

## 15. 当前方案的适用边界

当前实现是明确的 GLM-5 专用路径，不是所有模型都能直接开启的通用功能。重点约束包括：

- 依赖 PR 指定的 patched Megatron 和 SGLang；
- 依赖支持 batch-invariant 的 DeepGEMM；
- 依赖支持对齐 FP8 quantization 的 DeepEP；
- 当前端到端 gate 使用 EP=8，TP=PP=CP=ETP=1；
- 部分 dense/indexer 对齐路径当前明确要求 TP=1；
- dropout 为 0，相关 Linear/MLP 为 bias-free；
- MoE alignment backend 为 DeepEP；
- FP8 cache、GPU 架构和 kernel scale 格式需要匹配；
- DSA indexer 在 gate 中冻结，主模型参数仍完整训练。

## 16. 推荐的后续拆解顺序

后续可以按以下顺序逐项研究，每一项单独形成文档。

### 第一项：DeepGEMM batch invariant

目标问题：

- 为什么普通 FP8 GEMM 会随 `M` 改变？
- block-FP8 activation/weight 如何量化？
- Hopper FP32 scale 与 Blackwell UE8M0 scale 有何区别？
- 为什么 aligned forward 可以配 BF16 backward？
- batch-invariant 是否真的支持 token permutation？

建议入口：

- `alignment/deepgemm_forward.py`
- `tests/test_deepgemm_forward.py`

### 第二项：RMSNorm、residual 和 SwiGLU 边界

目标问题：

- FP32 residual 为什么必须跨层保留？
- SGLang fused add+RMSNorm 与 Megatron TE 路径差在哪里？
- 哪些 cast 点会导致下一层放大误差？
- 解析 backward 是否严格对应训练目标？

### 第三项：MoE router 和 top-k

目标问题：

- SGLang 与 Megatron router score 的完整公式是否相同？
- `sorted=False` 为什么影响 combine？
- tie-breaking 是否有普遍保证？
- 如何直接 dump 并比较两边 expert IDs、weights 和 margin？
- 何时仍应该使用 R3？

建议入口：

- `slime/utils/routing_replay.py`
- `alignment/deepgemm_forward.py` 中 router wrapper

### 第四项：DeepEP normal 与 low-latency bridge

目标问题：

- normal dispatch 为什么会丢失 route-level slot 信息？
- 主 hidden dispatch 与 metadata dispatch 分别传什么？
- route fingerprint 如何检查 receive order？
- `ep_gather` 为什么需要固定 top-k 顺序？
- backward 如何避免 nondeterministic atomic accumulation？

建议入口：

- `alignment/deepgemm_moe_forward.py`
- `alignment/deterministic_route_kernels.py`

### 第五项：Grouped MoE DeepGEMM 与显存复用

目标问题：

- expert `M` padding 如何与 SGLang 对齐？
- routed expert 和 shared expert 的 forward 顺序是什么？
- 为什么 FP32 probability multiply 会产生巨大临时 Tensor？
- expert grouping 和 dead-buffer reuse 如何控制峰值显存？

### 第六项：DSA indexer 和 sparse attention

目标问题：

- fused Q RMSNorm 输入如何重建？
- DSA indexer 如何生成 sparse top-k？
- FlashMLA sparse forward 和训练 backward 如何组合？
- Blackwell head padding 为什么改变 BF16 结果？
- 为什么当前 gate 冻结 indexer？

### 第七项：FP8 KV cache QAT

目标问题：

- FP8 E4M3 cache 的打包格式是什么？
- prefill 新鲜 KV 如何模拟 decode cache 误差？
- straight-through backward 的梯度假设是什么？
- 为什么只反量化被 sparse top-k 选中的 page？

### 第八项：端到端 gate 和可验证性

目标问题：

- 4096-token MAE 的精确聚合方式是什么？
- 如何增加 per-token max/p99 diff？
- 如何增加 route-ID/route-weight 精确 gate？
- 如何扩展到更多层和其他 TP/ETP 配置？
- layerwise hidden dump 在 prefill/decode/packed token 间如何匹配？

## 17. 阅读时应始终记住的主线

如果后续细节太多，可以一直用下面这条主线检查每项修改：

```text
它是在解决哪一种差异？

1. 重复运行不确定？
2. batch shape / pack 方式改变结果？
3. SGLang 与 Megatron 的算子数值语义不同？
4. prefill 与 decode 的 KV/attention 路径不同？
5. MoE expert ID、slot 或 combine 顺序不同？
6. forward 对齐后，backward 如何保持可训练？
```

PR #2262 的复杂度来自：这六类问题同时存在，而且任意一项都可能破坏最终 logprob 对齐。
