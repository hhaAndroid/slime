# slime RolloutManager 训练数据调度与负载均衡分析

本文分析 slime 通用的 rollout → train 数据调度、micro-batch packing、DP 分配与负载均衡流程。这些机制可服务 PPO、GRPO 等不同训练算法，不属于某一种 RL 算法本身。

这个过程和 PPO/GRPO 算法本身无关，是 slime 通用的 rollout -> train 数据调度逻辑。核心入口是：

```text
RolloutManager._split_train_data_by_dp(...)
  -> build_dp_schedule(...)
  -> partitions[dp_rank]
  -> micro_batch_indices[dp_rank]
  -> ray.put(per_rank_rollout_data)
```

调度策略可以概括为：

> pack first, distribute second：先把变长样本组成 micro-batch，再把 micro-batch 分配给 DP rank。

它不是简单地让每个 rank 拿相同数量的 token，也不是直接轮询每条 sample。因为 Transformer 的计算量不只与 token 数成正比，attention 还包含与序列长度平方相关的部分；同时 Megatron PP 又要求各 rank 执行相同数量的 micro-batch。

## 1. 负载均衡的单位是 DP replica

这里说的 train worker 需要先区分两类并行：

- TP/PP/CP rank：共同完成一个模型 replica 的一次 forward/backward。
- DP rank：持有一份模型 replica，负责训练不同的数据分片。

RolloutManager 真正做数据负载均衡的单位是 **DP rank / DP replica**，而不是把不同 sample 随意分给每个 TP/PP/CP 进程。

```text
DP replica 0
  ├─ TP/PP/CP worker ... 共同处理 partition[0]

DP replica 1
  ├─ TP/PP/CP worker ... 共同处理 partition[1]

DP replica 2
  ├─ TP/PP/CP worker ... 共同处理 partition[2]
```

同一 DP replica 内的模型并行 ranks 必须看到一致的数据调度；不同 DP replica 才处理互不重叠的 sample partition。

训练 worker 初始化后会把下面这些并行信息告诉 RolloutManager：

```text
dp_size
cp_size
vpp_size
microbatch_group_size_per_vp_stage
```

RolloutManager 据此在 rollout 全部生成完成后，统一计算全局 DP/micro-batch schedule。

## 2. 调度器的输入和输出

RolloutManager 首先取得每个训练 sample 的完整 token 长度：

```text
total_lengths[i] = len(data["tokens"][i])
```

然后调用 `build_dp_schedule`，主要输入是：


| 输入                  | 作用                                     |
| ------------------- | -------------------------------------- |
| `total_lengths`     | 估算每个 sample 的 token 数和 FLOPs           |
| `rollout_indices`   | 标识 sample 属于哪个 rollout                 |
| `global_batch_size` | 每个 optimizer step 包含多少个 rollout        |
| `dp_size`           | 要分给多少个 DP replica                      |
| `cp_size`           | 调整每个 dynamic micro-batch 可容纳的总 token 数 |
| `vpp_size/mb_group` | 约束 micro-batch 数，保证 VPP schedule 合法    |
| batching/balance 参数 | 选择 static、dynamic、token 或 FLOPs 调度方式   |


输出是四组 schedule：

```text
partitions[rank]
  = 这个 DP rank 最终拥有的全局 sample indices

micro_batch_indices[rank]
  = 这个 rank 每个 micro-batch 使用哪些本地 sample indices

num_microbatches[step]
  = 每个 DP rank 在这个 step 都要执行多少个 micro-batch

global_batch_sizes[step]
  = 这个 optimizer step 实际包含的 rollout 数
```

### 2.1 调度全流程速览（下次先看这里）

一句话概括：

> RolloutManager 先决定一次参数更新使用哪些 rollout，再把其中的 samples 打包成全局 micro-batches，对齐数量后分给各 DP ranks；每个 rank 依次执行自己分到的 micro-batches、累积梯度，最后共同更新一次参数。

完整流程是：

```text
rollout samples
  │
  ├─ 1. 按 global_batch_size 划分 optimizer step
  │     同一个 rollout 的 sibling samples(一拆多样本) 留在同一个 step
  │
  ├─ 2. 把当前 step 的 samples 打包成全局 K 个 micro-batches
  │     static  ：每个 micro-batch 固定 sample 数
  │     dynamic ：按 token cap 进行 first-fit packing
  │
  ├─ 3. 对齐全局 micro-batch 数 K
  │     要求 K 能被 dp_size（以及 VPP group）整除
  │     dynamic 不整除时拆分多样本 micro-batch
  │     static 不整除时直接报配置错误
  │
  ├─ 4. 把 K 个 micro-batches 分给 DP ranks
  │     每个 rank 固定拿 n = K / dp_size 个
  │     默认 round-robin；balance_data 用 KK 近似平衡每个 rank 的总 FLOPs
  │
  └─ 5. 开始训练
        每个 rank 做 n 次 forward/backward，梯度累计 n 次
        所有 micro-batches 完成后共同执行 1 次 optimizer.step()
```

最重要的数量关系是：

```text
K = 当前 optimizer step 全局共有多少个 micro-batch
n = K / dp_size = 每个 DP rank 的 num_microbatches
n 也是每个 DP rank 在本 step 的梯度累计次数
```

例如 `dp_size=4`，全局 packing 并对齐后 `K=12`：

```text
每个 rank 拿 12 / 4 = 3 个 micro-batches
每个 rank 做 3 次 forward/backward并累计梯度
最后共同执行 1 次 optimizer.step()
```

记住下面三个限制即可，后文只是展开实现细节：

- 普通 dynamic first-fit 能限制每个 micro-batch 的 token 总量；单条 sample 自身超限除外。
- `balance_by_flops` 只用 token cap 计算分组数，KK 分组本身不遵守 token cap，因此可能 OOM。
- `balance_data` 的 KK 只平衡每个 rank 在整个 optimizer step 中的总 FLOPs，不保证每个 accumulation slot 都均衡；EP 每个 micro-batch 内都有 All-to-All，因此当前调度在 EP 下可能产生明显的逐 slot 等待。

## 3. 第一步：先按 rollout 划分 optimizer step

调度器先按 `rollout_indices` 聚合 sample。同一个 rollout 可能因为 compact、subagent 等机制产生多个训练 sample，这些 sibling sample 会留在同一个 optimizer step 中：

```text
rollout 0 -> sample [0, 1, 2]
rollout 1 -> sample [3]
rollout 2 -> sample [4, 5]
```

如果 `global_batch_size=2`，第一个训练 step 会包含 rollout 0 和 rollout 1，也就是 sample `[0,1,2,3]`，而不是简单拿前两个 sample。

这一层本身不是计算量负载均衡，只是按 rollout 数量做确定性的连续分组。实现等价于：

```text
rollout_ids = 按第一次出现顺序去重后的 rollout id

step_0 = rollout_ids[0 : global_batch_size]
step_1 = rollout_ids[global_batch_size : 2 * global_batch_size]
...
```

这里不会：

- 按 sequence length 对 rollout 排序。
- 按 estimated FLOPs 在不同 optimizer steps 之间配平。
- 根据一个 rollout 展开出的 sample 数调整 step 边界。
- 为了平衡计算而把同一个 rollout 的 sibling samples 拆到不同 steps。

因此它只能保证每个完整 step 拥有相同数量的 distinct rollouts，不能保证每个 step 的 sample 数、token 数或总 FLOPs 相同。例如：

```text
global_batch_size = 2

rollout 0 -> 1 个短 sample
rollout 1 -> 1 个短 sample
rollout 2 -> 4 个长 sample
rollout 3 -> 3 个长 sample

step 0 = rollout [0, 1] -> 很轻
step 1 = rollout [2, 3] -> 很重
```

`--balance-data` 也不会改变这两个 step 的成员；它只会在每个 step 内部，把已经组成的 micro-batches 尽量均衡地分给 DP ranks。所以要区分两类目标：

```text
第 3 节 step grouping:
  固定 optimizer step 的 rollout 语义和 global batch size

第 4/6 节 packing + DP assignment:
  在一个已经确定的 step 内平衡显存和各 DP rank 计算量
```

输入 rollout 的先后顺序可能已经受上游 dataset shuffle / rollout shuffle 影响，但 `build_dp_schedule` 自己不会再次做 workload-aware shuffle。因而随机 shuffle 可以在统计上缓解连续重样本聚集，却不是严格的跨-step FLOPs balancing。

这样做保证：

- 同一个 rollout 拆出的 sibling 不会跨 optimizer step。
- per-rollout loss reducer 的 denominator 语义保持完整。
- 一个 rollout 产生多少训练 sample，不会改变以 rollout 数定义的 step 边界。

当前实现只构造完整的 step。末尾不足一个 `global_batch_size` 的 trailing rollout 不会进入本轮 schedule。这一步决定了 optimzer.step() 次数

## 4. 第二步：把 step 内样本打包成全局 micro-batch

调度器对一个 step 内的 sample 做一次全局 packing，得到 `K` 个 micro-batch。这里有三种主要路径。

### Static micro-batch

不开启 dynamic batch 时，按固定 `micro_batch_size` 连续切分：

```text
samples = [0,1,2,3,4,5,6,7]
micro_batch_size = 2

micro-batches = [[0,1], [2,3], [4,5], [6,7]]
```

这种方式保证每个 micro-batch 的 sample 数固定，但长短差异很大时，token 数和计算量可能很不均衡。

需要区分这里的两种“pack”：

```text
schedule-level packing：
  决定哪些完整 samples 属于同一个 micro-batch

tensor-level packed sequence：
  训练 forward 前，把已经确定的 samples concat/pad 成 Megatron 输入，
  并用 cu_seqlens 保留每条 sequence 的 attention 边界
```

static 路径在 schedule-level 只按固定 sample 数切分。后面的 `get_batch()` 确实还会执行 tensor-level packed sequence，但它不会：

- 重新调整这个 micro-batch 包含哪些 samples。
- 因为总 token 太多而再拆成两个 micro-batches。
- 检查或应用 `max_tokens_per_gpu`。
- 截断某条 sequence 来满足显存上限。

它只会：

1. 根据 CP 对每条 sequence 做 slice。
2. concat 当前 micro-batch 中的 token tensors。
3. 构造 `cu_seqlens` / `PackedSeqParams`，保证 packed samples 的 attention 仍相互隔离。
4. pad 到 `tp_size * data_pad_size_multiplier` 等对齐要求的倍数。

例如 static `micro_batch_size=2` 时，如果两条 sample 长度分别是 `1000` 和 `8000`：

```text
schedule 仍会得到同一个 micro-batch：[1000, 8000]
get_batch 会把它们变成约 9000 tokens 的 packed stream（再加对齐 padding）
```

即使该 packed stream 超过 GPU 能力，也不会自动重新切分，而是可能在 forward 时 OOM。`--max-tokens-per-gpu` 只有开启 `--use-dynamic-batch-size` 时才参与 schedule。

因此 static 模式的显存安全依赖：

- 上游 rollout 的单样本最大 context/response 长度约束。
- 用户根据最坏情况选择足够小的 `micro_batch_size`。
- 必要时使用 CP 分摊长 sequence。

另外，即使 dynamic first-fit 满足未 padding 的 token cap，`get_batch()` 后续仍可能加入少量对齐 padding，所以 `max_tokens_per_gpu` 也应保留一定显存余量，不能正好卡在理论 OOM 边界。

因此 static 并不是不能用。把 `micro_batch_size` 设小，就能通过梯度累积支持较大的 global batch。不过，梯度累积不能解决单个 micro-batch 已经太大导致的 OOM。

在这段训练流程里，static 和 dynamic 的核心区别是“如何切 micro-batch”：

- static：每个 micro-batch 固定放几条样本。
- dynamic：根据 token/FLOPs 决定每个 micro-batch 放哪些样本。

切完以后，两者使用的是同一套梯度累积和参数更新流程。

### Dynamic micro-batch：按 token first-fit packing

开启：

```bash
--use-dynamic-batch-size
--max-tokens-per-gpu N
```

后，固定 `micro_batch_size` 会被忽略。调度器按照 first-fit bin packing，把 sample 尽量装入 token budget：

```text
max_per_micro_batch = max_tokens_per_gpu * cp_size
```

例如长度为 `[100, 200, 300]`，上限为 `300`：

```text
micro-batch 0 = [100, 200]
micro-batch 1 = [300]
```

默认 dynamic 路径保证每个 micro-batch 的总 token 不超过上限。唯一例外是某个单独 sample 本身就超过上限：它不会被截断，而是独占一个超限 micro-batch，因此仍可能 OOM。

乘上 `cp_size` 是因为一个 CP group 会共同处理序列；例如 `cp_size=2`、`max_tokens_per_gpu=4096` 时，一个 micro-batch 的 group-level token budget 是约 `8192`。

### Dynamic + `balance_by_flops`

开启：

```bash
--use-dynamic-batch-size
--balance-by-flops
```

时，micro-batch 本身不再按 first-fit token cap 组织，而是用 FLOPs 权重通过 Karmarkar-Karp 做均衡分组。它更关注长序列 attention 的二次计算量，但有一个重要代价：

> `balance_by_flops` 不保证每个 micro-batch 仍低于 `max_tokens_per_gpu * cp_size`，可能产生超 token budget 的 micro-batch 并导致 OOM。

原因是当前代码只用 token cap 计算需要多少个 micro-batch：

```text
num_mbs = ceil(total_tokens / max_per_micro_batch)
```

随后调用 KK 时只传入“每条 sample 的 FLOPs 权重”和 `num_mbs`，没有把每条 sample 的 token 长度及 `max_per_micro_batch` 传进去。因此 KK 只优化各组 FLOPs 尽量接近，并不知道某一组的 token 总量是否已经越界。

例如，假设 attention FLOPs 简化为 `length²`： 它不是完全不使用最大长度，而是只用最大长度计算“分成几组”，不限制每组长度

```text
sample lengths = [4, 2, 2, 2, 2]
token cap = 6
total tokens = 12，因此 num_mbs = 2 # K = ceil(总 token 数 / max_tokens_per_micro_batch)

对应 FLOPs 权重 = [16, 4, 4, 4, 4]
```

KK 会认为下面是完美均衡：

```text
mb0 = [length 4]          FLOPs = 16，tokens = 4
mb1 = [2, 2, 2, 2]       FLOPs = 16，tokens = 8  # 超过 token cap 6
```

其实存在满足 token cap 的分法 `[4, 2]` 和 `[2, 2, 2]`，但它们的估算 FLOPs 分别是 20 和 12，不如 16/16 均衡。当前 KK 的目标只有 FLOPs 均衡，所以会选择前者。

这不是理论上无法同时约束，而是当前复用的 KK 实现属于无容量上限的 k-way partition。若要同时保证 token cap 和 FLOPs 均衡，需要实现 capacity-constrained packing：每次只能把 sample 放进仍有 token 容量的 bin，必要时还要增加 bin 数；这已经不是给当前 KK 多传一个参数就能完成的逻辑。

因此，显存安全优先时，更稳妥的组合是 `dynamic + balance_data`：先用 first-fit 保证每个 micro-batch 的 token cap，再把这些安全的 micro-batch 按 FLOPs 分配给 DP ranks。`balance_by_flops` 则是在组成 micro-batch 时优先追求 FLOPs 均衡，属于明确接受 OOM 风险的可选策略。

`balance_by_flops` 会自动开启 `balance_data`，并且要求 dynamic batch 已开启。

## 5. 第三步：对齐所有 DP rank 的 micro-batch 数

初次 packing 后的 micro-batch 数 `K` 不一定适合 Megatron 调度。slime 要求：

```text
K % [dp_size * (vpp_mb_group if vpp_size > 1 else 1)] == 0
```

原因是：

- 每个 DP rank 必须拿到相同数量的 micro-batch。
- PP ranks 才能执行相同次数的 pipeline schedule，避免 collective/通信失配。
- 使用 VPP 时，每个 rank 的 micro-batch 数还要满足 interleaved pipeline 的 group 要求。

dynamic 路径如果不整除，会反复拆分 token 总量最大的多样本 bin，直到 `K` 对齐。拆分只会让 bin 变小，不会破坏原 first-fit 的 token 上限。

以 `dp_size=4`、first-fit 后只有 `K=3` 个 micro-batch 为例：

```text
first-fit：[sample_a, sample_b]、[sample_c]、[sample_d]  # K=3
数量对齐：[sample_a]、[sample_b]、[sample_c]、[sample_d] # K=4
最终分配：4 个 DP rank 各拿 1 个 micro-batch
```

因此 4 个 rank 都执行 1 次 forward/backward，然后共同执行 1 次 optimizer step。

但如果不是“pack 后只有 3 个 micro-batch”，而是这个 optimizer step **总共只有 3 条原始 sample**，那么没有东西可以继续拆。`dp_size=4` 时这种 schedule 非法，代码会报错；必须增加该 step 的 sample 数，或者减小 DP size。也就是说，slime 不会让某个 DP rank 拿空数据继续训练。

这里的“反复拆分”不是重新执行 `balance_by_flops`，也不是不断降低 `max_tokens_per_gpu`，更不会把一条长 sequence 从 token 中间切开。它操作的是 micro-batch 中的 **sample index 集合**： 假设 dp_size=2，那么下面操作就是全局 micro-batch 数从 1 变成 2

```text
拆分前：micro-batch = [sample_a, sample_b, sample_c]

拆分后：
  micro-batch_left  = [sample_a]
  micro-batch_right = [sample_b, sample_c]
```

每次拆分的具体过程是：

1. 从当前 bins 中选择 token 总量最大的、且包含至少两条 sample 的 bin。
2. 把该 bin 内的 samples 按长度从大到小排序。
3. 依次把完整 sample 放入当前 token 总量较小的左半或右半。
4. 用这两个新 bins 替换原来的一个 bin，因此 `K` 每次只增加 `1`。
5. 重复直到 `K == target_K`。

例如：

```text
原 micro-batch sample lengths = [200, 150, 100]

token-balanced split：
  left  = [200]
  right = [150, 100]  # total = 250
```

所以被缩小的是“一个 micro-batch 中包含的完整 samples 集合及其 token 总量”，不是任何单条 sample 的长度。若某条 sample 自己有 `8000` tokens，拆分后它仍然是 `8000` tokens，只可能变成独占一个 micro-batch；调度器不会把它切成两条 `4000-token` sample。

这个 alignment splitting 和 FLOPs balancing 是两个独立阶段：

```text
默认 dynamic：
  first-fit token packing
    -> 为 K 对齐做 token-based bin splitting
    -> 可选 balance_data：按 FLOPs 把最终 bins 分给 DP ranks

balance_by_flops dynamic：
  按 FLOPs 组成初始 bins
    -> 为 K 对齐仍做 token-based bin splitting
    -> balance_data：按 FLOPs 把最终 bins 分给 DP ranks
```

最终“每个 DP rank 有相同数量的 micro-batch”来自两个约束的组合：

```text
target_K 是 dp_size 的整数倍
每个 rank 固定分到 target_K / dp_size 个 bins
```

它只硬保证 micro-batch **数量**相同，不硬保证每个 rank 的 token/FLOPs 完全相同；后面的 `balance_data` 才负责在“每个 rank bin 数相同”的前提下，让估算 FLOPs 尽量接近。

static 路径不能随意拆分，否则会破坏固定 `micro_batch_size` 语义；配置不整除时会直接报错，要求调整 `global_batch_size`、`micro_batch_size`、DP 或 VPP 配置。

## 6. 第四步：把 micro-batch 分给 DP ranks

到目前为止，已经得到合法的 `K` 个 micro-batch 后，每个 DP rank 一定拿到：

```text
num_microbatches_per_rank = K / dp_size
```

具体如何分有两种路径。

### 未开启 `balance_data`

使用 strided round-robin：

```text
rank 0 <- micro-batch [0, dp_size, 2*dp_size, ...]
rank 1 <- micro-batch [1, dp_size+1, ...]
...
```

它保证 micro-batch 数相同，但不保证总 token 或总 FLOPs 相同。

### 开启 `balance_data`

```bash
--balance-data
```

调度器先估算每个 sample 的 forward FLOPs，再求出每个 micro-batch 的权重：

```text
mbs_flops[k] = sum(sample_flops[i] for i in micro_batch[k])
```

随后用 Karmarkar-Karp largest differencing，把这 `K` 个带权 micro-batch 分给 `dp_size` 个 rank。调用中的 `equal_size=True` 同时约束：

1. 每个 rank 获得相同数量的 micro-batch。
2. 各 rank 的估算总 FLOPs 尽量接近。

```text
rank_flops[r] = sum(mbs_flops[k] for k assigned to rank r)
```

所以 `balance_data` 不是改变 micro-batch 内部 composition，而是在已有 micro-batch 之间做 DP rank 负载均衡；只有 `balance_by_flops` 才会连 micro-batch composition 也按 FLOPs 重排。

### 当前 KK 只平衡总体，不保证每一步均衡

这是当前第 6 节最重要的限制。KK 的输入只是：

```text
K 个 micro-batch 的权重
dp_size 个目标分组
每组必须包含相同数量的 micro-batch
```

它优化的是每个 rank 最终拿到的**权重总和**：

```text
rank_total_flops[r] = sum(该 rank 的所有 micro-batch FLOPs)
```

KK 不知道某个 micro-batch 会成为该 rank 的第几个 accumulation slot，也没有优化同一个 slot 上不同 ranks 的负载差。因此，它可能得到：

```text
                  slot 0    slot 1    总 FLOPs
rank 0              10         1        11
rank 1               2         9        11
```

从当前 KK 的目标看，这是完美均衡，因为两个 rank 的总 FLOPs 都是 11；但逐 slot 看，`10 vs 2` 和 `1 vs 9` 都很不均衡。也就是说，当前实现保证的是：

```text
整个 optimizer step 的 per-rank 总负载近似相等
```

而不是：

```text
每个 forward/backward slot 的跨-rank 负载近似相等
```

对于没有逐 micro-batch 跨-rank 同步的普通 dense DP，这个缺点未必造成明显等待，因为 ranks 可以独立推进，最终总时间主要取决于总体负载。但只要训练拓扑或功能在 micro-batch 内引入跨这些 ranks 的 collective，这个缺点就会直接变成逐 slot straggler；EP 是最典型的场景。即使没有 EP，逐 slot 顺序也可能影响 pipeline 行为、最后一个 micro-batch 的梯度通信 overlap 和实际 wall-clock，因此“总 FLOPs 均衡”仍不等价于“每一步执行时间均衡”。

### EP 场景的关键限制：只平衡整个 step 的总 FLOPs 不够

对于普通 dense DP，前几个 micro-batch 通常在 `no_sync` 中只做本地梯度累积，DP 梯度同步推迟到最后一个 micro-batch。此时不同 DP rank 可以独立推进，主要目标确实是让它们在整个 optimizer step 内的总计算量接近。

但开启 Expert Parallelism（EP）后，这个假设不再完整。EP ranks 会在**每个 micro-batch 的每个 MoE 层**执行 token dispatch/combine All-to-All。`no_sync` 只能推迟 DP 梯度同步，不能取消这些 EP collectives。因此，同一个 EP group 内某个 rank 的当前 micro-batch 更重时，其他 ranks 会在本次 EP 通信处等待。

例如，同一个 EP group 中有两个 ranks，并且各自分到两个 micro-batch：

```text
rank 0：mb0=10，mb1=1     # 总 FLOPs = 11
rank 1：mb0=2， mb1=9     # 总 FLOPs = 11
```

当前 `balance_data` 会认为二者已经完全均衡，因为每个 rank 的总量都是 11。但每个 micro-batch 内存在 EP 同步时，耗时更接近：

```text
slot 0：max(10, 2) = 10
slot 1：max(1, 9)  = 9
总耗时近似 = 19
```

如果进一步对齐同一个 EP group 内相同 accumulation slot 的负载：

```text
rank 0：mb0=10，mb1=1
rank 1：mb0=9， mb1=2

slot 0：max(10, 9) = 10
slot 1：max(1, 2)  = 2
总耗时近似 = 12
```

两种分法的 per-rank 总 FLOPs 都是 11，但第二种明显减少了逐 micro-batch 的 EP 等待。因此 EP 下需要同时优化两个层次：

1. 每个 DP rank 在整个 optimizer step 中的总负载接近。
2. 同一个 EP group、同一个 accumulation slot 中各 rank 的 micro-batch 负载也接近。

当前第 6 节只完成了第 1 项。这不只是 EP 参数缺失的问题，也是上述 KK 目标本身只看总体、不看 slot 的结果。此外，`train_parallel_config` 只向调度器提供 `dp_size`、`cp_size`、`vpp_size` 和 VPP micro-batch group 信息，没有提供 `ep_size` 或 DP-rank 到 EP-group 的映射，因此当前实现也无法进一步构造或优化逐 slot 的 EP-aware schedule。

CP 的情况不同：调度器使用的 `dp_size` 排除了 CP，同一个 CP group 处理的是同一个 micro-batch 的不同 sequence shards，不会像 EP 一样给 CP ranks 分配不同 micro-batch。EP 是这里需要额外处理的特殊情况。

这个限制通常不会造成训练结果错误或 collective 次数不匹配，因为每个 rank 仍执行相同数量的 micro-batch 和 MoE 层；但它可能造成严重的逐 micro-batch straggler，因而当前的“负载均衡”不能视为 EP 场景下完整的 wall-clock 均衡。

更合理的 EP-aware 调度应把最终分配显式组织为 `[accumulation_slot][dp_rank]`，并同时满足：

- 每个 DP rank 的 micro-batch 数相同。
- 同一 EP group 在每个 slot 的估算 FLOPs/token 近似接近。
- 每个 DP rank 跨所有 slots 的总负载近似接近。
- 每个 micro-batch 仍满足 token cap。

## 7. 为什么使用 FLOPs，而不是只看 token 数

长度为 `L` 的 Transformer sample，计算量可以粗略写成：

```text
F(L) = linear_projection_and_MLP_cost * L
     + attention_cost * L^2
     + LM_head_cost * L
```

两个 `1000-token` sample 的 token 总数与一个 `2000-token` sample 相同，但 attention 部分近似为：

```text
2 * 1000^2 < 2000^2
```

因此只平衡 token 总数仍可能让长序列所在 rank 更慢。slime 的 `calculate_fwd_flops` 会根据模型配置估算：

- Q/K/V projection。
- causal attention 的 `L^2` 成本。
- output projection。
- dense 或 MoE FFN。
- LM head。
- MLA、query groups、MoE top-k/shared expert 等模型差异。

训练 backward 的成本通常也与 forward workload 强相关，所以 forward FLOPs 可以作为整个 train step 负载的近似权重。

## 8. 一个简化例子

假设：

```text
dp_size = 2
cp_size = 1
sample lengths = [100, 200, 300, 400]
max_tokens_per_gpu = 500
```

first-fit dynamic packing 先得到：

```text
[100, 200], [300], [400]
```

当前 `K=3` 不能被 `dp_size=2` 整除，因此调度器拆分多样本 bin：

```text
[200], [300], [400], [100]
```

现在 `K=4`，每个 DP rank 必须拿两个 micro-batch。如果开启 `balance_data`，调度器会根据 FLOPs 倾向于形成类似：

```text
rank 0 <- [400] + [100]
rank 1 <- [300] + [200]
```

而不是让某个 rank 集中拿到所有长序列。真实分配由 Karmarkar-Karp 根据完整模型 FLOPs 估算决定，不是简单按长度排序配对。

## 9. 最终如何送到 train workers

调度结束后，RolloutManager 为每个 DP rank 构造一份 `rollout_data`：

```text
rollout_data_rank_r = {
    "partition": partitions[r],
    "tokens": samples selected by partitions[r],
    ...,
    "micro_batch_indices": micro_batch_indices[r],
    "num_microbatches": num_microbatches,
    "global_batch_sizes": global_batch_sizes,
}
```

然后分别通过 Ray Object Store 或 NIXL tensor transport 保存，返回 `rollout_data_refs[r]`。训练 worker 使用自己的 DP rank 拉取对应 ref；`DataIterator` 再严格按照 `micro_batch_indices[r]` 逐个产出 micro-batch。

PPO 下按当前支持约束，actor 和 critic 的并行拓扑应当相同，它们才可以使用同一套 DP partition 和 micro-batch schedule。这样同 rank 的 critic values 才能直接传给同 rank actor，不需要再次做 sample redistribution；当前框架尚未完整硬校验这一约束，详见 [PPO / GRPO 分析文档的 3.7 节](08_slime_ppo_grpo_algorithm_analysis.md)。

## 10. 它保证了什么，又没有保证什么

硬保证包括：

- 每个 DP rank 在每个 step 执行相同数量的 micro-batch。
- VPP 下 micro-batch 数满足 group 对齐要求。
- 每个保留的 sample 只分配一次，所有 DP partitions 互不重叠。
- 默认 dynamic 路径遵守 token cap，单样本本身超限除外。
- 同一个 rollout 产生的 sibling samples 不跨 optimizer step。

但“计算量差不多”是启发式目标，不是严格的 wall-clock 保证：

- Karmarkar-Karp 近似平衡 estimated FLOPs，不保证数学最优。
- `balance_data` 的 KK 只平衡每个 rank 在整个 optimizer step 中的总 FLOPs，不感知 accumulation slot，也不保证每一步 forward/backward 的跨-rank 负载接近。
- 当前调度不感知 EP group，也不平衡同一 EP group 在每个 accumulation slot 的负载；即使 per-rank 总 FLOPs 相同，也可能在每个 MoE All-to-All 处产生等待。
- MoE 实际 routing imbalance、节点性能差异和通信等待没有完全进入估算。
- kernel efficiency、padding、通信 overlap、显存带宽等不只由 FLOPs 决定。
- `balance_by_flops` 可能牺牲 token cap，带来 OOM 风险。
- PPO actor 和 critic 共用 schedule，但 actor vocabulary head 与 critic scalar head 的实际成本不同；同一 FLOPs 权重不可能同时精确拟合两者。

常见的安全配置组合是：

```bash
--use-dynamic-batch-size \
--max-tokens-per-gpu <safe_token_budget> \
--balance-data
```

它保留 first-fit 的 token 上限，再在 DP rank 之间按估算 FLOPs 平衡已有 micro-batch。只有在长短差异极大、确认显存有余量时，才进一步考虑 `--balance-by-flops`。
