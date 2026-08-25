# Rollout Top-p Replay 笔记

## 背景

PR: [https://github.com/THUDM/slime/pull/2102](https://github.com/THUDM/slime/pull/2102)

标题：`Support top_p mask`

合入时间：2026-06-19

这个 PR 不只是加了一个 `top_p` 采样参数。它的核心目的是：当 rollout 使用
`--rollout-top-p < 1.0` 时，让下面几类 logprob 处在同一个概率分布上：

- `rollout_log_probs`：rollout engine 生成 token 当时返回的 logprob。
- `old_log_probs`：训练端用 old actor 或 actor 在 rollout 样本上重算的 logprob。
- `current_log_probs`：训练 step 中当前 actor 前向得到的 logprob。

只要这些 logprob 后续会相减、取指数变成 ratio，或者作为 KL/mismatch 指标使用，
就必须保证它们来自同一种分布。这个 PR 做的就是把训练端重算 logprob 时的 support
改成 rollout 当时的 top-p nucleus，而不是完整 vocab。

## 为什么需要这个功能

top-p sampling 不是从完整 vocab softmax 里采样，而是先选出 nucleus，再在这个
nucleus 里重新归一化：

```text
q(a | s) = p(a | s) / sum_{x in nucleus(s)} p(x | s)
```

如果 rollout engine 给出的 logprob 是这个截断分布 `q` 下的 logprob，而训练端
Megatron 重算的是 full vocab softmax `p` 下的 logprob，那么两者不能直接相减
或相除。

这会影响所有依赖两个 logprob 对比的逻辑，例如：

- importance ratio
- PPO clip
- KL
- GSPO 的 sequence-level KL
- CISPO/GRPO 里复用的 PPO-style policy loss
- TIS/off-policy correction 里 `old_log_probs` 和 `rollout_log_probs` 的 ratio
- mismatch metrics 里 rollout engine logprob 和训练端重算 logprob 的差异

所以它不是严格意义上“只有 PPO 才需要”。更准确地说：凡是 policy loss 里用
两个 logprob 的差或其指数 ratio 的路径，都需要保证两边的 logprob 来自同一个
分布。

不需要它的场景是那些不比较 rollout 采样分布 logprob 和当前 policy logprob 的
路径，比如纯 SFT、value、reward 逻辑。

## 三种 logprob 和两类 ratio

这里最容易混淆的是三种 logprob：

```text
rollout_log_probs  = SGLang 生成 token 当时的 logprob
old_log_probs      = 训练端 old actor/actor 在同一批 token 上重算的 logprob
current_log_probs  = 当前正在更新的 actor logprob
```

slime 里至少有两类重要 ratio。

### Policy loss ratio

PPO-style policy loss 用的是：

```text
ratio = exp(current_log_probs - old_log_probs)
```

代码里通常先写成：

```text
ppo_kl = old_log_probs - current_log_probs
ratio = exp(-ppo_kl)
```

如果 rollout 当时是 top-p 截断分布，但训练端 old/current 都用 full vocab softmax
重算，那么这个 ratio 仍然能算出来，但它不是 rollout 真实采样策略下的 ratio。

有 top-p replay 时，训练端会在 rollout 记录的 nucleus 上重算：

```text
old     = log p_old(y) - log sum_{x in K_rollout} p_old(x)
current = log p_cur(y) - log sum_{x in K_rollout} p_cur(x)
```

其中 `K_rollout` 是 rollout 生成 token `y` 时 SGLang 实际保留的 top-p token set。

没有 top-p replay 时，训练端算的是：

```text
old     = log p_old(y)
current = log p_cur(y)
```

两者差了一个 normalization factor：

```text
log Z_old(K_rollout) - log Z_cur(K_rollout)
```

如果 old/current 很接近，这个差可能比较小，所以训练不一定立刻坏掉；但它在语义上
不是同一个采样分布下的 policy ratio。

### TIS / off-policy correction ratio

TIS 或类似 off-policy correction 用的是：

```text
tis = exp(old_log_probs - rollout_log_probs)
```

这里 top-p replay 更直接重要。因为 `rollout_log_probs` 来自 SGLang，如果它是按
top-p 截断分布返回的，而训练端 `old_log_probs` 是 full vocab softmax，那么
`exp(old - rollout)` 会混用两个不同分布。

所以 top-p replay 不只是服务 `old-current` 的 PPO ratio，也服务 `old-rollout`
的 TIS/mismatch 计算。

## “都重算了”为啥还需要

如果不使用 `--use-rollout-logprobs`，并且 old/current 都由训练引擎重算，那么确实
可以做到两边都用 full vocab softmax。这样 policy ratio 在数学上是自洽的：

```text
ratio_full = p_cur(y) / p_old(y)
```

但 rollout 真正采样用的是 top-p 分布：

```text
q(y) = p(y) / Z(K)
```

所以更严格的 rollout-policy ratio 应该是：

```text
ratio_top_p = [p_cur(y) / Z_cur(K)] / [p_old(y) / Z_old(K)]
```

也就是说，即使 old/current 都重算，top-p replay 的作用也不是“让 logprob 能算”，
而是“让 logprob 算在 rollout 真实采样 support 上”。

影响大小取决于：

- `top_p` 越低，normalization factor 越明显。
- old/current 差距越大，`Z_old(K)` 和 `Z_cur(K)` 的差越明显。
- response 越长，单 token 的差异越容易积累到 sequence-level KL/metrics。
- 如果启用 `--use-rollout-logprobs`，old/current/rollout 的分布错配会更直接。
- 如果还有 MoE routing replay 问题，top-p mismatch 和 routing mismatch 会叠加，让
  logprob 差异更难分析。

## 数据流

当 `args.rollout_top_p != 1.0` 时，slime 会在发给 SGLang 的 sampling params 里
加：

```python
custom_params = {"return_top_p_token_ids": True}
```

这要求 SGLang 返回每个生成 token 当时实际保留下来的 top-p token ids。

因为每个 response token 的 nucleus 大小不一样，所以 slime 用 ragged array
表示：

```text
rollout_top_p_token_ids     = 所有 token 的 kept ids 拼平
rollout_top_p_token_offsets = 每个 token 在 ids 里的起止位置，长度为 response_length + 1
```

第 `i` 个 response token 对应的 kept ids 是：

```python
ids[offsets[i]:offsets[i + 1]]
```

这些字段会被加入 `Sample`，经过 rollout data 传给训练端。`RolloutManager` 会校验：

- `offsets` 长度等于 `response_length + 1`
- `offsets[-1]` 等于 flattened ids 的长度

## 训练阶段怎么 replay

Megatron 训练端会根据这些 ids 构造一个 boolean keep mask：

```text
[T, vocab_local]
```

对于有 top-p replay 数据的 response 行，mask 外的 logits 会被置成 `-inf`，然后
再算 logprob。这样训练端重算的是同一个 top-p nucleus 上的归一化 logprob。

这个 replay mask 不只用于最终 policy loss 里的 current logprob。只要训练端调用
`compute_log_prob` 去补 `log_probs`、`ref_log_probs`、`teacher_log_probs` 或用于
mismatch/TIS 相关统计，开启 top-p replay 后都会沿同一套 logprob 计算路径走。
真正需要它的是后续拿这些 logprob 做差的地方，而不是某一个变量名本身。

这个 mask 还需要处理 CP/TP 对齐：

- CP=1 的普通 packed row
- zigzag CP
- `allgather_cp`
- tensor parallel 下的 vocab shard

实现里还会强制 keep 目标 token 所在 shard 上的 id，避免 SGLang 和 PyTorch 在
top-p 边界上有细微差异时导致 sampled token 被 mask 掉，从而产生 `-inf` 或 NaN。

注意：这个 PR 只对 logprob 使用 top-p mask；entropy 仍然按 unmasked logits 算。

## 存储和传输开销

这个方案的主要风险是 payload 可能很大。粗略估算：

```text
response_tokens * avg_kept_ids_per_token * 4 bytes
```

例如：

```text
1000 response tokens * 990 kept ids/token * 4 bytes ~= 3.96 MB/sample
```

如果经过 HTTP/base64 传输，还会大约膨胀 33%。`offsets` 本身很小，主要开销在
flattened `token_ids`。

top-p 不是“保留 p% 的 vocab”。它是按概率从高到低累计，直到累计概率达到 `p`。
如果分布很平，比如接近均匀，那么 `top_p=0.99` 可能会保留接近 99% 的 vocab。

PR 里加了一个指标：

```text
top_p_kept_vocab_per_token
```

开 `--rollout-top-p < 1.0` 时应该重点看这个指标。如果它到了几百甚至几千，说明
网络传输、Ray object store、训练 batch 内存都会有明显压力。

## PD 路径里的 4096 cap

SGLang patch 里有：

```python
MAX_PD_TOP_P_TOKEN_IDS = 4096
```

这个 cap 出现在 PD/disaggregation metadata buffer 路径里，因为那里用的是固定
shape 的 buffer：

```text
(size, MAX_PD_TOP_P_TOKEN_IDS)
```

它不是所有 SGLang 返回路径的通用 cap。普通 tokenizer manager 路径看起来是把
ragged arrays 编成 base64 int32 payload，没有这个 4096 限制。

当 PD 路径某一行 top-p ids 超过 4096 时，代码不是“截断到前 4096 个”，而是直接
退化成只保留实际采样出来的 token：

```python
output_top_p_token_ids = [int(req.output_ids[0])]
```

这个 token 不一定是最高概率 token。它只是当时实际生成出来的 sampled token。

## len=1 fallback 对训练的影响

一个 response 里每个 token 的 kept set 长度可以不同：

```text
token 0: kept ids len = 1
token 1: kept ids len = 990
token 2: kept ids len = 37
```

代码上不需要为 `len=1` 做特殊结构处理。ragged `ids + offsets` 可以统一表达所有
行。

但语义上，`len=1` 的行很特殊：

```text
keep set = {sampled_token}
softmax over keep set = 1
logprob(sampled_token) = 0
```

此时 current policy logprob 对 logits 的有效梯度基本没有了。目标 token 的 logit
同时出现在分子和分母中，会抵消；其他 token 被 mask 成 `-inf`，也没有
policy-gradient 信号。

如果 old/current logprob 都在训练端用同一个 `len=1` mask 重算，那么这个 token
的 ratio 基本是 1，KL 基本是 0。

如果使用 `--use-rollout-logprobs`，old logprob 可能来自 SGLang 原本按真实 top-p
分布算出的值，而 current logprob 在训练端因为 `len=1` 变成 0。这样 ratio/metric
可能不一致；但 current logprob 对 policy logits 的梯度仍然基本为 0。

所以这个 fallback 的作用是避免 PD buffer 溢出和数值问题，但代价是这部分 token
基本失去了有效 policy-gradient 信息。

## 实用结论

- 这个 PR 解决的是 top-p rollout 下 logprob 分布不一致的问题，不只是
  `old-current`，也包括 `old-rollout`。
- 它不是严格 PPO-only；所有 PPO-style old/current logprob 比较、TIS/off-policy
  correction、mismatch metrics 都会受影响。
- 如果 old/current 都由训练端 full softmax 重算，训练仍然能跑，但 ratio 对应的是
  full-softmax policy，不是 rollout 实际 top-p sampling policy。
- `--use-rollout-logprobs` 场景更需要小心，因为 rollout logprob 直接来自 SGLang
  的 top-p 分布，训练端如果不 replay top-p support，会直接混用两个分布。
- 最大风险是每 token 返回的 top-p ids 太多，导致 payload 很大。
- 应该重点监控 `top_p_kept_vocab_per_token`。
- PD 路径超过 4096 ids 时不会截断前 4096 个，而是 fallback 到 sampled-token only。
- sampled-token only 的行代码上不用特殊处理，但 policy-gradient 信号基本会消失。
