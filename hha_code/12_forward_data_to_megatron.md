# Forward 流程：从 slime 数据准备到 Megatron 内部

这篇从训练 actor 收到 `rollout_data` 开始，顺着一次 microbatch 的 forward 走到 Megatron 内部。例子仍然以 Qwen3.5 MoE + `--use-rollout-routing-replay` 为主。

CP / all-gather CP 的展开细节已经单独放到 [13_full_attention_allgather_cp.md](/mnt/shared-storage-user/huanghaian/code/slime_package/slime/hha_code/13_full_attention_allgather_cp.md)。本文只保留 forward 数据主线里必要的本地 shape 说明。

核心入口：

```text
MegatronTrainRayActor.train_actor
  -> get_data_iterator(rollout_data)
  -> train(...)
  -> train_one_step(...)
  -> forward_step(...)
  -> get_batch(...)
  -> model(**forward_kwargs)
  -> GPTModel.forward
```



## 1. rollout_data 里最初是什么形态

rollout 侧把一组 `Sample` 转成训练用的 `train_data` / `rollout_data`。关键字段大多是 list，每个元素对应一个样本：

```text
rollout_data["tokens"]:
  list[Tensor[L_i]]
  每个样本的完整 token 序列，prompt + response。

rollout_data["response_lengths"]:
  list[int]
  每个样本 response token 数，记作 R_i。

rollout_data["total_lengths"]:
  list[int]
  每个样本总 token 数，L_i = prompt_len_i + R_i。

rollout_data["loss_masks"]:
  list[Tensor[R_i]]
  只覆盖 response 段。一般 shape 是 [R_i]。

rollout_data["log_probs"] / "rollout_log_probs" / "ref_log_probs":
  list[Tensor[R_i]]
  每个样本 response token 上的 logprob。

rollout_data["advantages"] / "returns" / "values":
  list[Tensor[R_i]]
  PPO / GRPO 训练用的 response-token 级别数据。

rollout_data["rollout_routed_experts"]:
  list[Tensor[L_i - 1, num_layers, topk]]
  R3 用。SGLang rollout 侧返回每个“预测位置”的 routed experts。
```

为什么 routed experts 是 `L_i - 1` 行？因为模型在 token position `j` 的 logits 预测 `tokens[j + 1]`，所以一个长度 `L_i` 的序列有 `L_i - 1` 个预测位置。

Qwen3.5-35B-A3B 这个例子里：

```text
num_layers = 40
topk = 8
rollout_routed_experts[i].shape = [L_i - 1, 40, 8]
```

如果 microbatch 里有 `B_mb` 个样本，可以先记：

```text
tokens:
  [Tensor[L_0], Tensor[L_1], ..., Tensor[L_{B_mb-1}]]

sum_len = Σ_i L_i
```



## 2. DataIterator：按 micro_batch_indices 取样本

`train_actor` 先做：

```python
data_iterator = get_data_iterator(rollout_data)
num_microbatches = rollout_data["num_microbatches"] # 梯度累加次数
global_batch_sizes = rollout_data["global_batch_sizes"] # 一步完整训练的总样本条数
```

`get_data_iterator` 会根据 VPP 数量创建一个或多个 `DataIterator`。每个 `DataIterator.get_next(keys)` 根据：

```text
rollout_data["micro_batch_indices"]
```

取出当前 microbatch 的样本子集。

所以 `get_next(["tokens", "loss_masks", ...])` 返回的是：

```text
batch["tokens"]:
  list[Tensor[L_i]]

batch["loss_masks"]:
  list[Tensor[R_i]]

batch["total_lengths"]:
  list[int]

batch["response_lengths"]:
  list[int]
```

这一步还没有 pack，还没有变成 Megatron 输入。

### 2.1 num_microbatches / global_batch_sizes 是怎么来的

这两个字段不是 Megatron 的 `get_data_iterator` 算出来的，而是在 rollout manager 把一轮 rollout 数据切给训练 DP ranks 时写进 `rollout_data` 的：

```python
partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
    args,
    train_parallel_config,
    total_lengths,
    global_batch_size=args.global_batch_size,
    rollout_indices=data["rollout_ids"],
)
```

以 `scripts/run-qwen3.5-35B-rl-eagle-hha.sh` 为例：

```text
rollout_batch_size = 32
n_samples_per_prompt = 8

一轮 rollout 生成的训练 sample 数:
  32 * 8 = 256

actor GPUs = 8
TP = 2
PP = 1
CP = 1

训练 DP size:
  8 / (2 * 1 * 1) = 4
```

如果设置：

```text
--global-batch-size 256
```

那么一轮 rollout 只会拆成 1 个 Megatron train step：

```text
num_steps = 256 // 256 = 1
global_batch_sizes = [256]
```

这个 step 内的 256 条样本再按实际 token 长度做 dynamic microbatch packing。脚本开启了：

```text
--use-dynamic-batch-size
--max-tokens-per-gpu 9216
```

所以 `--micro-batch-size` 不决定 microbatch 数。假设 dynamic packing 之后全局得到 `K` 个 microbatch，Slime 会把 `K` 对齐到 `dp_size=4` 的倍数，然后：

```text
num_microbatches = [K / 4]
```

如果设置：

```text
--global-batch-size 128
```

那么一轮 rollout 会拆成 2 个 Megatron train step：

```text
num_steps = 256 // 128 = 2
global_batch_sizes = [128, 128]
num_microbatches = [K0 / 4, K1 / 4]
```

训练循环会执行两次 `train_one_step(...)`，也就是一轮 rollout 做两次 optimizer step，每次 step 使用 128 条训练 sample。

如果设置：

```text
--global-batch-size 512
```

会报错，因为一轮 rollout 只有 256 条训练 sample：

```text
num_steps = 256 // 512 = 0
```

`build_dp_schedule(...)` 里要求 `num_steps >= 1`，否则会触发类似：

```text
AssertionError: num_rollouts (256) < global_batch_size (512); need at least one rollout per step.
```

所以这个配置下，`global_batch_size` 最大只能到 256；如果要用 512，需要把 `rollout_batch_size * n_samples_per_prompt` 提高到至少 512。

### 2.2 VPP 下为什么要多个 DataIterator

`get_data_iterator(...)` 会按 VPP 数量创建多个 iterator：

```python
vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
return [DataIterator(rollout_data, micro_batch_indices) for _ in range(vpp_size)]
```

核心不是复制多份训练数据，而是给每个 virtual model chunk 一个独立 offset，让它们都能按自己的调度节奏读到同一个 microbatch 的 metadata。

以 `VPP=2` 为例：

```text
model[0] = virtual chunk 0
model[1] = virtual chunk 1

data_iterator[0] -> 给 chunk 0 用
data_iterator[1] -> 给 chunk 1 用
```

Megatron schedule 会按 `model_chunk_id` 调用：

```python
forward_step(
    forward_step_func,
    data_iterator[model_chunk_id],
    model[model_chunk_id],
    ...
)
```

chunk 1 的主输入确实不是原始 tokens，而是 chunk 0 传来的 hidden states；但是 slime 的 `forward_step` 每次都会先 `get_batch(...)`，这个 batch 里有当前 microbatch 的 metadata：

```text
packed_seq_params:
  attention 需要的 packed sequence 边界。

full_loss_masks / total_lengths / response_lengths / advantages / log_probs:
  最后一个 chunk 算 loss、对齐 response token 时需要。
```

所以 chunk 1 “读 MB0”的准确含义是：它需要 MB0 的 metadata，不是用 MB0 的 tokens 重新做 embedding。

如果 VPP=2 但只有一个共享 iterator，就可能出现 hidden state 和 metadata 错位：

```text
chunk0 跑 MB0:
  hidden/input 是 MB0
  metadata 读到 MB0

chunk1 跑 MB0:
  hidden state 是 chunk0 的 MB0
  但共享 iterator 已经前进，metadata 可能读到 MB1
```

因此需要两个独立 offset：

```text
data_iterator[0]:
  MB0 -> MB1 -> ...

data_iterator[1]:
  MB0 -> MB1 -> ...
```

这样 chunk 1 在处理 MB0 的 hidden state 时，也能拿到 MB0 对应的 `packed_seq_params` 和 loss metadata。

## 3. get_batch：把样本 pack 成 Megatron THD 输入

真正把 list 样本变成 Megatron forward 输入的是：

```python
batch = get_batch(
    data_iterator,
    keys,
    args.data_pad_size_multiplier,
    args.allgather_cp,
)
```

`get_batch` 做几件事：

1. 保存原始样本 token：

```python
batch["unconcat_tokens"] = tokens
```

shape：

```text
unconcat_tokens:
  list[Tensor[L_i]]
```

这个字段后面算 logprob/loss 时还要用，因为需要知道每个样本 response 段在哪。

1. 处理 token、concat 并 padding。

`get_batch` 会按当前并行配置把样本转换成本 rank 的 token 流。CP layout、`allgather_cp` 和 `PackedSeqParams` 坐标系的细节不在本文展开，见 [13_full_attention_allgather_cp.md](/mnt/shared-storage-user/huanghaian/code/slime_package/slime/hha_code/13_full_attention_allgather_cp.md)。

最后本 rank 拿到的 token 会 pad 到合适倍数：

```text
pad_size = tp_size * data_pad_size_multiplier
T_local = padded local token length
```

1. 构造 Megatron 输入：

```python
tokens = tokens.unsqueeze(0)
batch["tokens"] = tokens
```

shape：

```text
batch["tokens"]:
  Tensor[1, T_local]
```

这里 batch 维是 1，不代表 microbatch 只有一个样本，而是 slime 把多个样本 pack 到一个长序列里了。

1. 构造 `PackedSeqParams`：

```python
packed_seq_params = PackedSeqParams(
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_kv=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_kv=max_seqlen,
    qkv_format="thd",
)
```

常见 shape：

```text
cu_seqlens_q:
  Tensor[int32, B_mb + 1]               # 无 pad 时
  或 Tensor[int32, B_mb + 2]            # 有时末尾追加一个 pad 段

qkv_format:
  "thd"
```

`thd` 可以理解成 Megatron/TE 用 packed sequence 格式处理：

```text
T: packed token 维
H: head 维
D: head_dim
```

1. 构造 full loss mask。

原始：

```text
loss_mask_i:
  Tensor[R_i]
```

`get_batch` 会把它 pad 到完整 token 序列位置：

```python
prompt_length = total_length - response_length
loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
```

shape 从：

```text
[R_i]
```

变成：

```text
[L_i]
```

为什么左边是 `prompt_length - 1`，右边是 `1`？

因为 logits 位置 `j` 预测 token `j + 1`。response 第一个 token 在 `prompt_length`，对应的预测 logits 位置是 `prompt_length - 1`。最后一个 token 没有下一个 response target，所以右侧补 1 个 0。

最后和 tokens 一样映射到本 rank 的 packed token layout：

```text
batch["full_loss_masks"]:
  Tensor[1, T_local]
```

并且代码断言：

```python
loss_masks.shape == tokens.shape
```



## 4. forward_step 传给模型什么

训练时 `train_one_step.forward_step` 会准备：

```python
forward_kwargs = {
    "input_ids": batch["tokens"],
    "position_ids": None,
    "attention_mask": None,
    "labels": None,
    "packed_seq_params": batch["packed_seq_params"],
    "loss_mask": batch["full_loss_masks"],
}

output_tensor = model(**forward_kwargs)
```

shape 总结：

```text
input_ids:
  Tensor[1, T_local]

position_ids:
  None

attention_mask:
  None

labels:
  None
  slime 不让 GPTModel 内部直接算 CE loss，而是拿 logits 出来后自己算 PPO/SFT loss。

packed_seq_params.cu_seqlens_q:
  Tensor[int32, num_packed_sequences + 1]

loss_mask:
  Tensor[1, T_local]
```

`labels=None` 很重要。因为 GPTModel `_postprocess` 看到 `labels is None` 时会返回 logits：

```python
return logits.transpose(0, 1).contiguous()
```

所以输出到 Slime loss 函数的是：

```text
policy actor:
  logits: Tensor[1, T_local, V_local]

critic:
  values/logits: Tensor[1, T_local, 1]
```

这里 `V_local` 是 TP 切分后的本地 vocab 大小。因为 slime provider 创建 GPTModel 时 `parallel_output=True`，Megatron output layer 不 gather 全 vocab。

后面的 logprob kernel 会用 TP group 做 all-reduce：

```text
max over vocab: all_reduce(MAX)
sum exp vocab: all_reduce(SUM)
target logit: all_reduce(SUM)
```

所以即使本地 logits 是 `[T_local, V_local]`，最后算的是全 vocab 上的 logprob。

### 4.1 开了 sequence parallel，为什么 logits 的 T 没有除以 TP

这里要区分 Transformer 层内部的 SP layout 和最终 lm head 的输出 layout。

开启 sequence parallel 后，Transformer 层内一些位置的 hidden states 确实按 TP rank 切 sequence 维：

```text
T_sp = T_local / tp_size

SP hidden states on each TP rank:
  [T_sp, 1, H]
```

但是 GPTModel 的 `output_layer` 是 `ColumnParallelLinear`。当它的 `sequence_parallel=True` 时，linear forward 会先在 TP group 内把输入 hidden states 沿 sequence 维 all-gather：

```text
[T_sp, 1, H]
  -- TP all-gather sequence -->
[T_local, 1, H]
```

然后每个 TP rank 再用自己持有的 vocab weight shard 做 projection：

```text
local output weight:
  [V_local, H]

local logits:
  [T_local, 1, V_local]
```

所以 Slime 最终拿到的是：

```text
[1, T_local, V_local]
```

而不是：

```text
[1, T_local / tp_size, V_local]
```

这是因为 vocab-parallel softmax 要让所有 TP ranks 持有相同的 token 行，只在 vocab 列上分片，才能对每个 token 做 `MAX` / `SUM` all-reduce：

```text
TP rank 0: all T_local tokens x first vocab shard
TP rank 1: all T_local tokens x second vocab shard
```

当前实现不是 sequence 和 vocab 同时二维切分的 output layout。SP 会显著减少 Transformer 主体内的部分激活，但不会把最终 logits 的 token 维再除以 TP。

### 4.2 full-vocab logits 的显存成本

这块输出很大，而且在训练中属于需要参与 autograd 的激活。

当前 actor 使用 BF16 模型，但 Megatron 的 `Float16Module` 默认会把最后一个 PP stage 的模型输出转成 FP32。Slime 的 `get_log_probs_and_entropy(...)` 也明确要求：

```python
assert logits.dtype == torch.float32
```

因此单个 TP rank 上，仅 logits 本身的显存约为：

```text
M_logits = T_local * V_local * 4 bytes
         = T_local * (padded_vocab_size / tp_size) * 4 bytes
```

对本文 Qwen3.5-35B-A3B 配置：

```text
T_local = 9216                  # 取 dynamic microbatch 上限附近
padded_vocab_size = 248320
tp_size = 2
V_local = 124160

M_logits = 9216 * 124160 * 4 bytes
         ~= 4.26 GiB / GPU
```

随后 Slime 的 `_VocabParallelLogProbEntropy` 会从 logits 计算 vocab-parallel softmax，并为 backward 保存一个同样是 `[T_local, V_local]` 的 FP32 `log_prob_softmax`。在没有 top-p mask、entropy 不参与梯度的常见路径上，loss forward 峰值附近至少会同时看到：

```text
FP32 logits:                  ~= 4.26 GiB
FP32 softmax/autograd state: ~= 4.26 GiB
------------------------------------------------
仅这两项:                    ~= 8.52 GiB / GPU
```

这还没有计算 output projection 临时 buffer、hidden states、attention/MoE 激活等。`--entropy-coef 0` 会避免额外保存 entropy backward 需要的 full-vocab 张量，但不会消除 logits 和 logprob softmax 这两个主要张量。

`--log-probs-chunk-size` 只是把已有 logits 上的 logprob/entropy 计算按 token 分块。完整 logits 已经由 lm head 产生，而且各 chunk 为 backward 保存的 softmax 总元素数仍约为 `T_local * V_local`，所以它能降低部分临时峰值，但不能从根本上消除这块 full-vocab 激活。

### 4.3 当前 fused 路径的边界

这里有两类容易混淆的 fusion：

```text
Slime 当前已有:
  logits + vocab-parallel softmax/target gather/entropy
  -> fused logprob/entropy autograd kernel
  -> 输入仍然是完整 [T_local, V_local] logits

当前 actor/SFT 主线没有:
  hidden states + lm head weight + target tokens
  -> fused per-token logprob/loss
  -> 不物化完整 logits
```

当前 Megatron 底层已经有 `LinearCrossEntropyModule`。当以下条件成立时，它可以融合 lm head projection 和 CE：

```text
labels is not None
cross_entropy_loss_fusion = True
cross_entropy_fusion_impl = "linear"
```

但是 Slime 的训练 forward 固定传 `labels=None`，所以 GPTModel 必然先执行 output layer 并返回 logits。即使 `loss_type=sft_loss`，当前也是在 Slime 中从完整 logits 重新取目标 token logprob，再计算 NLL，没有接入 Megatron 的 fused linear CE。

对 SFT，这条路径原则上可以比较直接地改成 fused linear CE。对 PPO/GRPO，per-token CE 等于目标 token 的 `-logprob`，所以一个支持 `reduction="none"` 和任意上游梯度的 fused linear CE/logprob 算子原则上也能接 policy loss；但还要处理 Slime 当前支持的 entropy、rollout top-p replay、temperature、CP/SP packed layout、GSPO/OPSM/TIS/CISPO 和 custom loss 等分支。

特别是普通 fused CE 只返回目标 token 的 logprob，不能直接提供完整分布 entropy。本文脚本的 `entropy_coef=0`，entropy 不参与梯度但仍会作为 metric 计算；若要完全消除 full logits，需要同时决定 entropy metric 是关闭、降频计算，还是实现 tiled/fused entropy。

另外，本文当前环境安装的 Megatron `cross_entropy_fusion_impl="linear"` 实现只支持 compute capability major 10（Blackwell）。因此现状可以总结为：Megatron 已有可复用的 fused linear CE 基础能力，但 Slime 当前 actor/SFT 主线尚未接入能消除 `[T_local, V_local]` logits 的 fused 路径。

### 4.4 Megatron fused linear CE 内部怎么避免 full logits

当前环境安装的 Megatron 里，这不是只有接口的占位实现，而是一条完整的 forward/backward 路径。公开入口是：

```text
megatron.core.fusions.fused_linear_cross_entropy.linear_cross_entropy
```

Megatron GPTModel 的 output layer 则包装成：

```text
LinearCrossEntropyModule(ColumnParallelLinear)
```

普通路径是：

```text
hidden [T, H]
  -> lm head GEMM
  -> logits [T, V_local]
  -> vocab-parallel softmax / target gather
  -> per-token CE or logprob
```

fused linear CE 接口直接接收：

```text
hidden:
  [T, H]
  SP 下调用前可以是 [T_sp, H]

local lm head weight:
  [V_local, H]

labels:
  [T]

reduction:
  "none" / "sum" / "mean"
```

当 `reduction="none"` 时输出是 `[T]`。实现内部变量名叫 `logprobs`，但实际计算的是：

```text
maximum + log(sum_exp) - target_logit
  = logsumexp(logits) - target_logit
  = -log p(target)
  = per-token cross entropy
```

所以如果 PPO 需要目标 token logprob，语义上只要：

```python
token_logprob = -token_ce
```



#### 4.4.1 forward：按 vocab chunk 在线归约

实现不会把 `[T, V_local]` 写到 HBM，而是沿本地 vocab 分块。默认配置是：

```text
vocab_per_split = 512 * 6 = 3072
```

也可以用环境变量调整：

```text
LCE_FWD_VOCAB_SPLIT_SIZE
LCE_BWD_VOCAB_SPLIT_SIZE
```

对本文配置：

```text
V_local = 124160
num_splits = ceil(124160 / 3072) = 41
```

每个 split 的主流程是：

```text
hidden @ weight_chunk.T
  -> logits_chunk [T, <= 3072]
  -> 在线更新每个 token 的:
       local maximum
       local sum(exp(logit - maximum))
       target_logit
  -> 不把这个 logits_chunk 保存成完整输出
```

为了稳定地合并不同 vocab chunk，使用的是 online log-sum-exp：

```text
new_max = max(old_max, chunk_max)

new_accu = exp(old_max - new_max) * old_accu
         + sum(exp(chunk_logits - new_max))
```

TP 下每个 rank 只持有自己的 vocab shard，因此还会对 maximum、accumulate 和目标 token logit 做 TP 通信，最后得到全词表上的：

```text
CE = global_max + log(global_accumulate) - global_target_logit
```

SP 下，这个实现仍然会先在 TP group 内 all-gather hidden：

```text
per-rank SP hidden:
  [T_sp, H]

all-gather 后:
  global_hidden [T_local, H]
```

然后再进行本地 vocab 分块 projection。也就是说它没有同时保留 sequence 和 vocab 的二维输出切分；它和普通 vocab-parallel lm head 一样需要完整 token 行，但只保存很小的 hidden，而不保存巨大 logits。

#### 4.4.2 forward 为 backward 保存什么

自定义 autograd 不保存 full logits 或 full softmax，只保存：

```text
global_hidden:       [T_local, H]
weight:              [V_local, H]   # 参数引用，不复制一份 weight
labels:              [T_local]
maximum:             [T_local]      # FP32
accumulate:          [T_local]      # FP32, 即稳定化后的 sum-exp
num_valid_tokens:    scalar
```

forward 计算期间还有：

```text
_max / _accu:
  [T_local, num_splits]
```

但对本文例子 `num_splits=41`，每个 FP32 tensor 约为：

```text
9216 * 41 * 4 bytes ~= 1.44 MiB
```

远小于 `[9216, 124160]` 的 full-vocab tensor。

#### 4.4.3 backward：按 vocab chunk 重算 logits

backward 从上层收到每个 token 的梯度：

```text
d_token_ce:
  [T_local]           # reduction="none"
```

然后逐个 vocab split 重算：

```text
logits_chunk = hidden @ weight_chunk.T

p_chunk = exp(logits_chunk - maximum) / accumulate

dlogits_chunk = d_token_ce * (p_chunk - one_hot(target)_chunk)
```

再用这个局部 `dlogits_chunk` 累计：

```text
dHidden += dlogits_chunk @ weight_chunk

dWeight_chunk = dlogits_chunk.T @ hidden
```

当前 backward 默认只申请：

```text
_d_logits:
  [T_local, vocab_per_split]
  即 [T_local, 3072]
```

对本文 BF16 配置：

```text
9216 * 3072 * 2 bytes ~= 54 MiB
```

`dHidden` 在累计时使用 FP32。对 `[9216, 2048]`：

```text
9216 * 2048 * 4 bytes ~= 72 MiB
```

TP 下各 rank 计算本地 vocab shard 对 `dHidden` 的贡献，然后 all-reduce；如果开启 SP，最后只把属于当前 TP rank 的 `[T_sp, H]` slice 返回给上游。

所以这条路径本质上是：

```text
不保存 logits/softmax
  + backward 重算一次 lm head projection
  = 用额外计算换大幅显存下降
```



#### 4.4.4 和当前 Slime 路径的量级对比

仍取：

```text
T_local = 9216
H = 2048
V_local = 124160
```

当前 Slime policy loss 的核心 full-vocab 激活：

```text
FP32 logits:                   ~= 4.26 GiB
FP32 saved logprob softmax:   ~= 4.26 GiB
合计:                          ~= 8.52 GiB / GPU
```

fused linear CE 中与这部分对应的主要 tensor：

```text
saved BF16 global_hidden:      ~= 36 MiB
saved maximum + accumulate:    ~= 72 KiB
backward BF16 dlogits chunk:   ~= 54 MiB, 临时
backward FP32 dHidden:         ~= 72 MiB, 临时
```

`dWeight`、lm head weight 和 optimizer/main-grad buffer 两条训练路径都需要，不应算成 fused 路径独有的成本。精确峰值还会受 allocator、TP 通信 buffer 和 kernel workspace 影响，但这里已经能看出：核心差异是 GiB 级 full-vocab 激活变成百 MiB 量级的 hidden/chunk 临时量。

#### 4.4.5 为什么它在数学上可以接 PPO

PPO/GRPO 最终只需要采样 target token 的当前策略 logprob：

```text
logprob = -CE(target)
```

而 fused operator 的 `reduction="none"` 支持每个 token 独立的任意上游梯度。PPO 的 ratio、clip、advantage 等先从 `logprob` 算出：

```text
g_i = d L_policy / d logprob_i
```

再通过：

```text
d logprob_i / d logits_ij
  = one_hot(target_i)_j - softmax(logits_i)_j
```

传回 lm head。所以 PPO 的非线性 clipping 并不要求 loss kernel 直接输出完整 logits；per-token fused CE/logprob 已经足够承载这个梯度。

但当前 Megatron fused linear CE 还不能直接覆盖 Slime 的全部能力：

```text
1. 不输出 entropy。
2. 不支持 rollout top-p replay 后的稀疏 support 归一化。
3. 没有显式 temperature 参数。
4. 不理解 Slime 的 sample/response 边界、CP layout 和 packed loss mask。
5. custom loss 可能确实要求原始 logits。
```

temperature 在纯线性无 bias lm head 下可以通过缩放 hidden 等价实现；sample/response/CP 对齐则需要 Slime 在 forward 前构造与 logits position 对齐的 shifted labels，并把 prompt/pad 位置设为 `ignore_index`。

entropy 需要额外设计。普通 CE 只累计：

```text
sum(exp(z))
```

如果 fused forward 再在线累计：

```text
sum(exp(z) * z)
```

就可以在不物化 full logits 的前提下得到 entropy metric。若 entropy 需要梯度，backward 也可以利用重算出来的 `logits_chunk`、softmax chunk 和保存的逐 token 统计量计算。

#### 4.4.6 如何真正接进 Slime

仅在启动参数里增加：

```bash
--cross-entropy-loss-fusion
--cross-entropy-fusion-impl linear
```

还不够，因为 Slime 当前传给 GPTModel 的仍是 `labels=None`。一条可行的 common fast path 至少需要：

```text
1. 在 get_batch/forward_step 构造 shifted labels:
   response target 使用真实 token id；
   prompt、最后一个无 target 位置和 padding 使用 ignore_index。

2. 传 labels 给 GPTModel，让 output layer 返回 per-token CE，而不是 logits。

3. Slime loss callback 使用 logprob = -CE，继续计算 PPO/GRPO loss。

4. 只在不要求原始 logits 的配置启用；例如先覆盖:
   CP=1、无 top-p replay、entropy_coef=0、非 custom loss。

5. 为 entropy metric 选择关闭、降频，或扩展 fused kernel。
```

Megatron 的 GPTModel 已经在有 labels 且 fusion flags 开启时走这条 output layer 分支；MTP loss 也已经有对应调用。因此主要缺口不在 fused CE 数学或底层 kernel，而在 Slime 的 batch/forward/loss 接口目前围绕“模型返回 logits”设计。

最后还有硬件和依赖限制。当前环境这份实现要求：

```text
GPU compute capability major == 10
hidden/weight dtype 为 FP16 或 BF16且相同
hidden size 满足 kernel 对齐要求
CUTLASS/CuTe、CUDA Python bindings、Triton 可用
```

其他 GPU 架构会在 platform dispatch 阶段报 `Unsupported architecture`。`cross_entropy_fusion_impl="native"` 或 `"te"` 可能支持更多硬件，但它们是 logits 产生之后的 CE fusion，不能像 `"linear"` 这样消除 lm head 的 full-vocab logits。

## 5. Megatron GPTModel.forward 主干

进入 Megatron 后，主干是：

```text
GPTModel.forward
  -> _preprocess(...)
  -> self.decoder(...)       # TransformerBlock
  -> _postprocess(...)
```

大致 shape：

```text
input_ids:
  [1, T_local]

embedding / decoder_input:
  逻辑 shape [T_local, 1, H]
  SP-sharded 位置的单 rank 物理 shape 可为 [T_sp, 1, H]

decoder hidden_states:
  逻辑 shape [T_local, 1, H]
  SP-sharded 位置的单 rank 物理 shape 可为 [T_sp, 1, H]

output_layer logits:
  [T_local, 1, V_local]

return to slime:
  [1, T_local, V_local]
```

这里的 batch size 仍然是 1，因为 microbatch 内样本被 pack 到 `T_local` 维。

## 6. TransformerBlock.forward

`GPTModel.forward` 里的：

```python
hidden_states = self.decoder(...)
```

会进入 `TransformerBlock.forward`。

它做的核心事情是遍历当前 PP/VPP stage 上的 layer：

```python
for layer in self.layers:
    hidden_states, context = layer(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=packed_seq_params,
        ...
    )
```

shape：

```text
hidden_states:
  逻辑 shape Tensor[T_local, 1, H]
  SP-sharded 位置的单 rank 物理 shape 可为 Tensor[T_sp, 1, H]
```

如果有 pipeline parallel：

```text
非 first PP stage:
  hidden_states 来自上一个 PP stage 的 p2p recv

非 last PP stage:
  输出 hidden_states 发给下一个 PP stage
```

这里的 PP 通信不改变上述逻辑 shape；如果同时开启 SP，当前 TP rank 在 SP-sharded 位置仍可能只持有 `[T_sp, 1, H]`。

## 7. TransformerLayer.forward

Megatron 每层的结构是：

```text
TransformerLayer.forward
  -> _forward_attention
  -> _forward_mlp
```

attention 段：

```text
residual = hidden_states
input_layernorm
self_attention(...)
post_self_attn_layernorm
self_attn_bda
```

MLP/MoE 段：

```text
residual = hidden_states
pre_mlp_layernorm
self.mlp(...)
post_mlp_layernorm
mlp_bda
```

其中：

```python
self.self_attention(...)
self.mlp(...)
```

这两个对象是什么，取决于建模阶段传入的 `transformer_layer_spec`。

## 8. Qwen3.5 attention forward

Qwen3.5 的 `get_qwen3_5_spec` 会先让 Megatron 生成普通 GPT decoder block spec，然后检查 HF config：

```text
text_config.layer_types[layer_id]
```

如果某层是：

```text
linear_attention
```

就把该层：

```text
layer_specs.submodules.self_attention
```

替换成：

```python
ModuleSpec(
    module=slime_plugins.models.qwen3_5.Attention,
    params={"args": args},
)
```

所以 forward 到这一行时：

```python
attention_output_with_bias = self.self_attention(...)
```

如果是 full attention 层：

```text
self.self_attention = Megatron / TE 默认 attention
```

如果是 linear attention 层：

```text
self.self_attention = slime_plugins.models.qwen3_5.Attention
```

Qwen3.5 的 `Attention` 继承 `HuggingfaceAttention`。它的 forward 通用流程是：

```text
输入:
  hidden_states [T_local, 1, H]

permute:
  [T, 1, H] -> [1, T, H]

调用 hf_forward(...)

permute 回:
  [1, T, H] -> [T, 1, H]

输出:
  (output [T_local, 1, H], bias=None)
```

这里不展开 `HuggingfaceAttention` 里对 sequence parallel / context parallel 的 gather 和 scatter 处理；Qwen3.5 CP 的单独说明见 [13_full_attention_allgather_cp.md](/mnt/shared-storage-user/huanghaian/code/slime_package/slime/hha_code/13_full_attention_allgather_cp.md)。

Qwen3.5 自己的 `hf_forward` 是：

```python
hidden_states = self.input_layernorm(hidden_states)
hidden_states = self.linear_attn(
    hidden_states=hidden_states,
    cu_seqlens=packed_seq_params.cu_seqlens_q,
)
```

其中：

```text
hidden_states:
  [1, T, H]

cu_seqlens:
  packed sequence 边界，用来让 GatedDeltaNet 知道每个样本的序列范围
```

`self.linear_attn` 是：

```text
Qwen3_5GatedDeltaNet
```

它内部主要 shape：

```text
输入 hidden_states:
  [B_pack, S_pack, H]
  在 slime packed 情况下通常是 [1, T, H]

in_proj_qkv(hidden_states):
  [1, T, 2 * key_dim + value_dim]

in_proj_z(hidden_states):
  [1, T, value_dim]

query:
  [1, T, num_k_heads or num_v_heads, head_k_dim]

key:
  [1, T, num_k_heads or num_v_heads, head_k_dim]

value:
  [1, T, num_v_heads, head_v_dim]

chunk_gated_delta_rule(..., cu_seqlens):
  [1, T, num_v_heads, head_v_dim]

out_proj:
  [1, T, H]
```

所以 Qwen3.5 的自定义 forward 只替换 attention 子模块；TransformerLayer 外壳、residual、BDA、MLP/MoE 仍是 Megatron。

## 9. MoE forward

Qwen3.5 MoE 层的 `self.mlp` 是 Megatron `MoELayer`。

`MoELayer.forward` 主流程：

```text
hidden_states [T_local, 1, H]
  |
  | route
  v
probs, routing_map
  |
  | preprocess / dispatch
  v
dispatched_input
  |
  | experts(dispatched_input)
  v
expert_output
  |
  | combine
  v
output [T_local, 1, H]
```

更具体：

```python
shared_expert_output = self.shared_experts_compute(hidden_states)
probs, routing_map = self.route(hidden_states)
hidden_states, probs, residual = self.preprocess(hidden_states, probs, routing_map)
dispatched_input, probs = self.dispatch(hidden_states, probs)
output, mlp_bias = self.routed_experts_compute(dispatched_input, probs, residual)
output = self.combine(output, shared_expert_output)
```

router 输出 shape：

```text
logits:
  [T_local * 1, num_experts]

top_indices:
  [T_local * 1, topk]

probs:
  [T_local * 1, topk] 或等价表示

routing_map:
  [T_local * 1, num_experts]
  bool/multihot，每个 token topk 个 True
```

对 Qwen3.5-35B-A3B：

```text
num_experts = 256
topk = 8

top_indices:
  [T_local, 8]

routing_map:
  [T_local, 256]
```



## 10. R3：rollout routed experts 如何影响 forward

开启：

```bash
--use-rollout-routing-replay
```

会隐式启用：

```bash
--use-routing-replay
```

并让 SGLang 返回 routed experts：

```text
payload["return_routed_experts"] = True
```

SGLang engine 启动时也会打开：

```text
enable_return_routed_experts = True
```



### 10.1 rollout 侧 shape

每个样本：

```text
rollout_routed_experts:
  Tensor[L_i - 1, num_layers, topk]
```

Qwen3.5：

```text
[L_i - 1, 40, 8]
```



### 10.2 训练前填充 RoutingReplay

`train_actor` 一开始：

```python
if args.use_rollout_routing_replay:
    self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)
```

它会按 microbatch 读取：

```python
batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
```

然后调用：

```python
prepare_routed_experts_for_routing_replay(...)
```

这个函数会检查：

```python
experts.shape[0] == token_ids.shape[0] - 1
```

然后给每个样本补 1 行 dummy experts，使 routed experts 长度和 token stream 对齐：

```text
原始:
  [L_i - 1, num_layers, topk]

补 1 行:
  [L_i, num_layers, topk]
```

之后和 tokens 一样对齐到本 rank 的 packed token layout，得到：

```text
routed_experts_local:
  Tensor[T_local_or_sp_local, num_layers, topk]
```

如果 sequence parallel 开启，最终还会按 TP rank 切 sequence 维：

```text
T_sp = T_local / tp_size
routed_experts_local:
  [T_sp, num_layers, topk]
```



### 10.3 每个 MoE router 是什么时候注册 RoutingReplay 的

`register_routing_replay(...)` 不是从 `train_actor` 显式调用的，而是在模型构造期间由每个 Megatron `TopKRouter.__init__` 自动调用：

```python
from slime.utils.routing_replay import register_routing_replay
register_routing_replay(self) # 也就是每一层都会调用一次
```

完整启动顺序是：

```text
--use-rollout-routing-replay
  -> 参数处理阶段令 args.use_routing_replay = True
  -> ActorGroup 创建 actor worker 前设置 ENABLE_ROUTING_REPLAY=1
  -> worker 内构造 Megatron model
  -> 每个 MoE layer 构造自己的 TopKRouter
  -> TopKRouter.__init__ 调 register_routing_replay(self)
```

`register_routing_replay(module)` 只有在环境变量打开时才工作：

```python
if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
    module.routing_replay = RoutingReplay()

    def pre_forward_hook(*args, **kwargs):
        set_routing_replay(module.routing_replay)

    module.register_forward_pre_hook(pre_forward_hook)
```

这里做了两件事。

第一，每个 MoE router 都得到独立的 replay 对象：

```text
MoE layer 0 router.routing_replay -> RoutingReplay 0
MoE layer 1 router.routing_replay -> RoutingReplay 1
...
```

假设当前进程实际构造了 30 个 MoE router，这里就会创建 30 个完全独立的
`RoutingReplay` 实例。每个实例分别持有：

```python
self.top_indices_list # 每个里面都是 list，表示 micro batch
self.forward_index  # forward 用
self.backward_index # 不是说 backward 时候用的，实际上是 backward 时候重计算用的
```

其中两个 index 都不是 layer id，而是“当前这一层已经消费到第几个
microbatch/第几次调用”：

```text
layer 7 RoutingReplay:
  top_indices_list[0] -> layer 7 / MB0
  top_indices_list[1] -> layer 7 / MB1
  top_indices_list[2] -> layer 7 / MB2

  forward_index  -> layer 7 的正常 forward 消费游标
  backward_index -> layer 7 的 checkpoint 重算消费游标
```

`RoutingReplay.__init__` 同时把对象追加到进程内的注册表：

```python
RoutingReplay.all_routing_replays.append(self) # 就这个代码比较绕，这个对象是类变量，所有 RoutingReplay 共享的
```

这个类变量只是保存所有实例引用的全局注册表，并不保存一份被所有层共享的
route 数据：

```text
RoutingReplay.all_routing_replays
├── [0] -> layer 0 自己的 RoutingReplay
├── [1] -> layer 1 自己的 RoutingReplay
└── ...
```

它有三个用途：

1. `clear_all()` / `clear_all_forward()` 能遍历所有实例统一重置，从而不需要遍历每个对象去 clear
2. `fill_routing_replay()` 能按注册顺序把 rollout 的逐层 expert id 填到对应实例；
3. 用列表长度校验 rollout 遍历到的 MoE 层数是否与注册数一致。

因此，调用 `clear_all()` 并不是不清理每个实例，而是由注册表代替调用方逐个执行：

```python
for replay in RoutingReplay.all_routing_replays:
    replay.clear()
```

第二，每个 router 都注册一个 forward pre-hook。后面某层 router 真正 forward 前，这个 hook 会把模块级变量：

```python
ROUTING_REPLAY # 这个也很关键，用于切换到每一层的 routing_replay
```

切换为当前层的 `module.routing_replay`。

所以 `ROUTING_REPLAY` 不是所有层共享的一份数据，而是一个“当前正在执行哪一层”的进程内指针。真正的数据仍分别保存在每层自己的 `RoutingReplay.top_indices_list` 中。

这里三种引用各自负责不同的事情：

```text
module.routing_replay:
  router 和本层 replay 的稳定绑定，供运行时 pre-hook 使用。

RoutingReplay.all_routing_replays:
  所有本地 MoE 层的注册表，供批量管理和 rollout 数据填充使用。

模块级 ROUTING_REPLAY:
  当前正在 forward 的那一层，供拿不到 TopKRouter/self 的 compute_topk wrapper 使用。
```

从数据结构上说，`all_routing_replays` 完全可以改写成全局
`list[LayerReplayState]` 或 `dict[layer_id, LayerReplayState]`；当前使用小对象只是把
`record/pop/clear` 和每层状态封装在一起。真正不能轻易合并的是每层独立的
buffer 和游标。如果只用一条全局 `top_indices` 队列，就会要求 rollout 写入、
训练 forward、PP/VPP microbatch 调度以及 checkpoint 重算的跨层调用顺序完全一致。
每层独立维护游标后，只要求同一层内部的 microbatch 顺序一致，层与层之间的
执行交错不会让某层误取另一层的数据。

所以这套结构可以概括为：

```text
必要的部分：每个本地 MoE 层有独立的 replay buffer 和消费游标。
实现得较绕的部分：通过类变量注册表 + 全局当前指针 + pre-hook 接入 compute_topk。
```

后半部分主要是为了少改 Megatron 的 routing API：`compute_topk(scores, topk, ...)`
拿不到 router 对象或 layer id，只能由 pre-hook 在调用前设置当前层。

### 10.4 `fill_routing_replay` 如何给注册好的对象填数据

模型构造完成以后，`RoutingReplay.all_routing_replays` 已经包含当前 PP/VPP rank 上所有 MoE router 的 replay 对象。

`fill_routing_replay` 再按相同的 model chunk / layer 顺序遍历 MoE 层：

```python
routing_replay_offset = 0
for vp_stage, model in enumerate(self.model):
    for layer_id in range(offset, offset + num_layers_to_build):
        if this layer is dense:
            continue

        layer_routed_experts = rollout_routed_experts[:, layer_id]
        RoutingReplay.all_routing_replays[routing_replay_offset].record(
            layer_routed_experts
        )
        routing_replay_offset += 1
```

每次 `record(...)` 会把 GPU tensor 复制到 CPU pinned memory：

```text
layer_routed_experts:
  [T_local_or_sp_local, topk]

Qwen3.5:
  [T_local_or_sp_local, 8]
```

如果有多个 microbatch，每层 replay 最终是：

```text
layer 0 replay.top_indices_list:
  [MB0 top_indices, MB1 top_indices, ...]

layer 1 replay.top_indices_list:
  [MB0 top_indices, MB1 top_indices, ...]
```

这段代码依赖以下顺序一致：

```text
TopKRouter 的构造/注册顺序
==
fill_routing_replay 的 model chunk / MoE layer 遍历顺序
```

代码最后只检查：

```python
assert routing_replay_offset == len(RoutingReplay.all_routing_replays)
```

也就是检查 MoE router 数量一致，没有给每个 replay 额外保存 layer id 再逐层校验。

### 10.5 forward pre-hook 和 compute_topk wrapper 怎么连起来

某个 MoE 层实际执行 router 时，链路是：

```text
TopKRouter.forward(input)
  -> PyTorch 先执行 Slime 注册的 forward_pre_hook
     -> set_routing_replay(module.routing_replay)
     -> ROUTING_REPLAY 指向当前 MoE layer
  -> self.gating(input)
     -> router logits
  -> self.routing(logits)
  -> topk_routing_with_score_function(...)
```

`get_routing_replay_compute_topk(...)` 也不是启动时全局 monkey patch 一次，而是在每次进入非 fused 的 `topk_routing_with_score_function(...)` 时，动态包住当前调用里的局部函数：

```python
def compute_topk(scores, topk, num_groups=None, group_topk=None):
    # Megatron 原始 topk / 原生 RouterReplay 逻辑
    ...

compute_topk = get_routing_replay_compute_topk(compute_topk)
```

随后 softmax 或 sigmoid routing 分支执行：

```python
probs, top_indices = compute_topk(...)
```

这里实际调用的已经是 Slime wrapper。wrapper 读取三份状态：

```text
ENABLE_ROUTING_REPLAY:
  是否打开整个 Slime routing replay 机制。

ROUTING_REPLAY_STAGE:
  当前是 fallthrough / record / replay_forward / replay_backward。

ROUTING_REPLAY:
  forward pre-hook 刚设置的当前层 replay 对象。
```

完整分支是：

```text
fallthrough:
  调原始 compute_topk，正常选择专家。

record:
  调原始 compute_topk；
  把得到的 top_indices 写入当前层 ROUTING_REPLAY。

replay_forward:
  top_indices = 当前层 ROUTING_REPLAY.pop_forward()
  probs = scores.gather(1, top_indices)

replay_backward:
  top_indices = 当前层 ROUTING_REPLAY.pop_backward()
  probs = scores.gather(1, top_indices)
```

所以 R3 replay 时：

```text
router logits / scores:
  仍由当前训练模型计算。

topk expert id:
  来自 rollout/record 阶段保存的数据。

topk probs:
  用当前训练模型的 scores 在 replay expert id 上 gather。
```

得到 `top_indices` 和 `probs` 后，Megatron 再 scatter 成：

```text
routing_probs [num_tokens, num_experts]
routing_map   [num_tokens, num_experts]
```

后续 token dispatch、expert 和 combine 不需要知道这些 expert id 是正常 topk 还是 replay 得到的。

### 10.6 训练 forward 和 backward recompute 实际用哪个 index

`RoutingReplay` 给每层维护两个独立游标：

```text
forward_index
backward_index
```

actor 各阶段先设置外层 stage：

```text
ref / teacher logprob:
  ROUTING_REPLAY_STAGE = fallthrough

独立 actor logprob:
  rollout routing replay -> replay_forward
  普通 routing replay  -> record  # 这个就是 r2 的用法

进入 actor train 前:
  ROUTING_REPLAY_STAGE = replay_backward
```

四个 stage 的准确含义和使用位置是：


| stage             | top-k 行为                           | 使用位置                                                      |
| ----------------- | ---------------------------------- | --------------------------------------------------------- |
| `fallthrough`     | 调原始 `compute_topk`，不记录也不回放         | ref/teacher logprob forward                               |
| `record`          | 正常计算 top-k，并把 `top_indices` 保存到当前层 | 普通 `--use-routing-replay` 的 actor logprob forward         |
| `replay_forward`  | 调当前层 `pop_forward()`               | rollout actor logprob forward、正式 training forward         |
| `replay_backward` | 调当前层 `pop_backward()`              | activation checkpoint 在 autograd backward 中触发的 forward 重算 |


但是“actor train 使用 replay_backward”这个说法不够准确。Slime 的训练 `forward_step` 在调用 `model(...)` 前会临时执行：

```python
old_stage = os.environ["ROUTING_REPLAY_STAGE"]       # replay_backward
os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
output_tensor = model(...)
os.environ["ROUTING_REPLAY_STAGE"] = old_stage      # 恢复 replay_backward
```

因此真实时间线是：

```text
训练的原始 forward:
  replay_forward
  -> 每层 pop_forward()
  -> 消费 forward_index

forward 结束:
  stage 恢复成 replay_backward

activation checkpoint 在 backward 中重算 layer forward:
  replay_backward
  -> 每层 pop_backward()
  -> 消费 backward_index
```

checkpoint recompute 会再次执行 router forward，所以它必须再次使用同一批专家 id；两个独立 index 让原始 forward 和 backward recompute 可以各自从 MB0 开始消费同一份 `top_indices_list`。

这里需要特别区分两件事：普通的梯度 backward 只沿已有计算图传播，不会再次调用
`TopKRouter.forward`；只有 activation checkpoint 为恢复未保存的激活而重算 forward
时，才会再次进入 router 并真正消费 `backward_index`。如果没有 activation
checkpoint，训练外层虽然仍把 stage 设成 `replay_backward`，但通常没有 router
调用会读取这个 index。

如果训练前已经单独跑过 actor logprob，`forward_index` 已经走到列表末尾。rollout routing replay 会在进入训练前调用：

```python
RoutingReplay.clear_all_forward()
```

只把所有层的 `forward_index` 重置为 0，保留数据和 `backward_index`。训练完成后再调用：

```python
RoutingReplay.clear_all()
```

同时清空每层数据以及两个 index。

把两种数据来源和正式训练串起来，完整时间线是：

```text
普通 routing replay:
  actor logprob forward --record--> 保存 top_indices
  training forward     --replay_forward--> 固定专家选择
  checkpoint recompute --replay_backward--> 再次固定相同专家

rollout routing replay:
  SGLang routed experts --fill_routing_replay--> 保存 top_indices
  可选 actor logprob    --replay_forward--> 固定 rollout 专家选择
  clear_all_forward()  --只重置 forward_index-->
  training forward     --replay_forward--> 再从 MB0 固定专家选择
  checkpoint recompute --replay_backward--> 再次固定相同专家

ref/teacher logprob:
  --fallthrough--> 使用各自模型正常计算的专家选择
```

因此 `replay_backward` 更准确的含义其实是
`replay_checkpoint_recompute`，而不是“整个训练阶段”或“梯度 backward 本身”。

### 10.7 普通 routing replay 和 rollout routing replay 的记录来源

只开启：

```bash
--use-routing-replay
```

时，没有 SGLang routed experts 可提前填充。Slime 会先跑一次 actor logprob forward，并设置：

```text
ROUTING_REPLAY_STAGE=record
```

每层 wrapper 正常执行当前模型 topk，再把结果写进自己的 replay。后面的训练 forward/recompute 重放这次 actor logprob 的路径。

开启：

```bash
--use-rollout-routing-replay
```

时，`fill_routing_replay(...)` 已经直接把 SGLang rollout expert ids 填进每层 replay，不需要 `record` forward。后面直接使用 `replay_forward` / `replay_backward`。

### 10.8 两个容易混淆或漏掉的限制

第一，Megatron 自己也有一个原生调试机制：

```python
self.router_replay       # Megatron RouterReplay
```

Slime 注册的是：

```python
self.routing_replay      # Slime RoutingReplay
```

一个是 `router_replay`，一个是 `routing_replay`，不是同一个对象。`topk_routing_with_score_function(...)` 先定义包含 Megatron 原生 replay 的 `compute_topk`，Slime 再把自己的 wrapper 包在它外面：

```text
Slime wrapper
  -> fallthrough/record 时可调用 Megatron 原始 compute_topk
  -> replay_forward/replay_backward 时直接忽略原始 compute_topk
```

第二，如果配置：

```text
moe_router_fusion = True
```

`topk_routing_with_score_function(...)` 会在前面直接调用 TE fused router 并返回，不会执行：

```python
compute_topk = get_routing_replay_compute_topk(compute_topk)
```

所以当前 Slime routing replay hook 只覆盖非 fused router 路径。本文脚本没有开启 `--moe-router-fusion`，默认值是 `False`；如果以后打开，需要先给 fused 分支接 replay，或者显式禁止它与 routing replay 同时启用。

### 10.9 最新 Megatron 原生 RouterReplay：可以去掉哪些 patch

截至 2026-08-04，Megatron Core 官方最新 release 是 `0.18.2`。本文环境安装的
`0.16.0rc0` 已经包含原生 `RouterReplay`，当前官方文档也继续提供该能力：

- [Megatron Core releases](https://github.com/NVIDIA/Megatron-LM/releases)
- [官方 Router Replay 设计文档](https://docs.nvidia.com/megatron-core/developer-guide/0.17.1/api-guide/router_replay.html)
- [最新 TransformerConfig API](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.transformer_config.html)

它不是 Slime patch。打开：

```python
config.moe_enable_routing_replay = True
```

后，每个 `TopKRouter` 原生构造自己的对象：

```python
self.router_replay = RouterReplay()
```

随后 router 把它显式传入 routing 函数：

```text
TopKRouter.routing(...)
  -> topk_routing_with_score_function(
       ...,
       router_replay=self.router_replay,
     )
  -> router_replay.get_replay_topk(...)
```

因此 Megatron 原生实现不需要 Slime 的 forward pre-hook 和模块级
`ROUTING_REPLAY` 当前层指针；routing 函数已经直接拿到了当前层对象。

原生对象提供三种 action：

```python
RouterReplayAction.RECORD
RouterReplayAction.REPLAY_FORWARD
RouterReplayAction.REPLAY_BACKWARD
```

以及模型级管理接口：

```python
RouterReplay.set_replay_data(all_layers_topk_indices)
RouterReplay.get_recorded_data()
RouterReplay.set_global_router_replay_action(action)
RouterReplay.clear_global_indices()
RouterReplay.clear_global_router_replay_instances()
```

它和当前 Slime 实现的对应关系是：


| 当前 Slime patch                      | Megatron 原生能力                                                   |
| ----------------------------------- | --------------------------------------------------------------- |
| `RoutingReplay.all_routing_replays` | `RouterReplay.global_router_replay_instances`                   |
| `module.routing_replay`             | `TopKRouter.router_replay`                                      |
| `record()`                          | `RouterReplayAction.RECORD`                                     |
| `pop_forward()`                     | `RouterReplayAction.REPLAY_FORWARD`                             |
| `pop_backward()`                    | `RouterReplayAction.REPLAY_BACKWARD`                            |
| `fill_routing_replay()` 最终逐层填数据     | `RouterReplay.set_replay_data(...)` / `set_target_indices(...)` |


所以如果后续切换到新版 Megatron，理论上可以删除两处对 Megatron 源码的 Slime
patch：

```text
TopKRouter.__init__ 中：
  register_routing_replay(self)

moe_utils.compute_topk 外：
  get_routing_replay_compute_topk(compute_topk)
```

但不能仅升级 Megatron 后直接删掉所有 Slime routing replay 代码。Megatron 只提供
底层 replay 机制，不知道 SGLang 和 Slime 的训练生命周期；Slime 仍需要负责：

```text
rollout_routed_experts 的接收
  -> packed token、CP、SP 对齐
  -> 按 PP/VPP layer 和 microbatch 准备 replay 数据
  -> ref/teacher/actor/training/recompute 阶段切换 action
  -> 一轮训练结束后清理 replay 数据/action
  -> 仅在模型销毁或重新构造前清理全局 instance registry
```

这里最需要验证的是多 microbatch 语义。当前 Slime 为每层预存：

```text
top_indices_list = [MB0, MB1, MB2, ...]
```

而本文环境中的 Megatron 原生对象使用当前 `target_topk_idx`，并用
`replay_backward_list` 为 checkpoint 重算保存队列。因此迁移时需要在正确的
microbatch 时机调用 `set_target_indices()`，或者采用目标 Megatron 版本提供的
批量/static buffer API，不能机械地把现有列表直接替换成一次
`set_replay_data(...)`。

最终结论是：

```text
新版 Megatron 已原生提供 R3 所需的 Router Replay 内核；
Slime 可以改成只做数据对齐和训练阶段编排，从而不再 patch Megatron 源码；
但这需要一次有针对性的集成迁移，并不是升级依赖后自动生效。
```

另外，本文环境的原生 replay 与 Slime replay 一样位于非 fused routing 分支；迁移到
具体新版后仍应单独验证 `moe_router_fusion=True` 是否已经支持 replay。

## 11. logits 如何变成 PPO/SFT loss

因为 `labels=None`，Megatron 返回 logits 给 Slime：

```text
logits:
  [1, T_local, V_local]
```

`loss_function` 再按 `args.loss_type` 分发：

```text
policy_loss:
  policy_loss_function

value_loss:
  value_loss_function

sft_loss:
  sft_loss_function
```

### 11.1 policy logprob

`policy_loss_function` 会调用：

```python
get_log_probs_and_entropy(
    logits,
    unconcat_tokens=batch["unconcat_tokens"],
    total_lengths=batch["total_lengths"],
    response_lengths=batch["response_lengths"],
)
```

里面先 squeeze：

```text
logits:
  [1, T_local, V_local] -> [T_local, V_local]
```

然后构造 shifted target token：

```text
full_tokens:
  [T_local]
```

语义：

```text
logits[j] 预测 full_tokens[j]
full_tokens[j] = original_tokens[j + 1]
```

再调用：

```python
calculate_log_probs_and_entropy(logits, full_tokens, tp_group, ...)
```

这个 kernel 在 TP vocab parallel 下做：

```text
local logits:
  [T_local, V_local]

target:
  [T_local]

输出:
  log_prob_full [T_local, 1]
  entropy_full  [T_local] 或 None
```

最后只取 response 位置：

```text
log_probs_list:
  list[Tensor[R_i_local]]
```

对最简单的单 rank token layout，第 i 个样本：

```text
end = previous_offset + total_length_i
start = end - response_length_i

logits response slice:
  logits[start - 1 : end - 1]

target response tokens:
  tokens[-response_length_i:]
```

也就是：

```text
response token k 的 logprob
  来自它前一个位置的 logits
```

### 11.2 policy loss shape

`policy_loss_function` 里：

```python
advantages = torch.cat(batch["advantages"], dim=0)
old_log_probs = torch.cat(old_log_probs, dim=0)
log_probs = torch.cat(log_probs, dim=0)
```

shape：

```text
advantages:
  [R_total_local]

old_log_probs:
  [R_total_local]

log_probs:
  [R_total_local]

ppo_kl = old_log_probs - log_probs:
  [R_total_local]

pg_loss:
  [R_total_local]
```

再用：

```python
sum_of_sample_mean(...)
```

按样本/rollout 维度做归一化。

### 11.3 value loss

critic 的输出不是 vocab logits，而是：

```text
values/logits:
  [1, T_local, 1]
```

`get_values` 会按 response 段切出：

```text
values_list:
  list[Tensor[R_i_local]]
```

然后和：

```text
old_values / returns:
  [R_total_local]
```

计算 clipped value loss。

## 12. 一次 microbatch 的完整 shape 总图

设：

```text
B_mb: microbatch 样本数
L_i: 第 i 个样本 prompt+response 总长度
R_i: 第 i 个样本 response 长度
T_raw = Σ_i L_i
T_local: pack/pad 后，本 rank 实际 token 长度
T_sp = T_local / tp_size: SP-sharded 模块在单个 TP rank 上看到的 token 数
H: hidden_size
V_local: TP 本地 vocab size
E: num_experts
K: moe_router_topk
```

其中 `T_local` 在不同上下文里有一个细微差别：`batch["tokens"]` 的 shape 是 `[1, T_local]`，但如果 sequence parallel 开启，部分 layer 内部模块看到的 sequence 维可能已经按 TP rank 再切成 `T_sp = T_local / tp_size`。下面 MoE/R3 里用 `T_moe` 表示 router 实际处理的本地 token 数。

数据准备：

```text
rollout_data["tokens"]:
  list[Tensor[L_i]]

rollout_data["loss_masks"]:
  list[Tensor[R_i]]

get_batch 后:
  batch["tokens"]          Tensor[1, T_local]
  batch["full_loss_masks"] Tensor[1, T_local]
  packed_seq_params.cu_seqlens_q Tensor[num_segments + 1]
```

Megatron forward：

```text
input_ids:
  [1, T_local]

embedding output:
  [T_local, 1, H]

TransformerBlock/Layer hidden_states:
  逻辑 shape [T_local, 1, H]
  SP-sharded 位置的单 rank 物理 shape 可为 [T_sp, 1, H]

Qwen3.5 linear attention 内部:
  [T_local, 1, H] -> [1, T, H] -> GatedDeltaNet -> [1, T, H] -> [T_local, 1, H]

MoE router:
  logits      [T_moe, E]    # T_moe 是 router 实际看到的本地 token 数
  top_indices [T_moe, K]
  routing_map [T_moe, E]

output_layer:
  SP input on each TP rank [T_sp, 1, H]
  -> TP all-gather sequence [T_local, 1, H]
  -> local vocab projection [T_local, 1, V_local]

return to slime:
  [1, T_local, V_local]
```

loss：

```text
logits squeeze:
  [1, T_local, V_local] -> [T_local, V_local]

shifted target:
  [T_local]

log_prob_full:
  [T_local, 1] -> [T_local]

response log_probs:
  list[Tensor[R_i_local]]

cat response log_probs:
  Tensor[R_total_local]
```

R3：

```text
rollout_routed_experts per sample:
  [L_i - 1, num_layers, K]

after pack/pad/local layout:
  [T_moe, num_layers, K]

per MoE layer replay:
  [T_moe, K]
```

## 13. forward_backward_func 里 loss 在哪里算

`train_one_step` 里看不到显式的 `loss = ...; loss.backward()`，因为 Megatron pipeline schedule 用的是 callback 协议：

```python
forward_backward_func = get_forward_backward_func()
losses_reduced = forward_backward_func(
    forward_step_func=_wrap_forward_step_with_microbatch_pbar(forward_step, microbatch_pbar),
    data_iterator=data_iterator,
    model=model,
    num_microbatches=num_microbatches,
    seq_length=args.seq_length,
    micro_batch_size=args.micro_batch_size,
    decoder_seq_length=args.decoder_seq_length,
    forward_only=False,
)
```

Slime 传进去的 `forward_step` 返回两个东西：

```python
output_tensor = model(**forward_kwargs)
return output_tensor, partial(loss_function, args, batch, num_microbatches, step_global_batch_size)
```

对 policy actor 来说：

```text
output_tensor:
  logits [1, T_local, V_local]

loss_func:
  已经绑定了 args / batch / num_microbatches / step_global_batch_size 的 loss_function
  只差最后一个参数 logits
```

Megatron schedule 调用用户的 `forward_step_func` 后，会进入自己的 `forward_step_calc_loss(...)`：

```python
output_tensor, loss_func = forward_step_func(data_iterator, model)
output_tensor, num_tokens = forward_step_calc_loss(...)
```

真正算 loss 的位置在 `forward_step_calc_loss(...)` 里：

```python
if is_last_stage:
    outputs = loss_func(output_tensor)
```

所以这里等价于：

```python
logits = output_tensor
loss, num_tokens, loss_reduced = loss_function(
    args,
    batch,
    num_microbatches,
    step_global_batch_size,
    logits,
)
```

Slime 的 `loss_function` 再根据 `args.loss_type` 分发：

```python
match args.loss_type:
    case "policy_loss":
        func = policy_loss_function
    case "value_loss":
        func = value_loss_function
    case "sft_loss":
        func = sft_loss_function
```

Qwen3.5 这个脚本没有显式设置 `--loss-type`，默认是 `policy_loss`，所以实际进入 `policy_loss_function(...)`。它会用 logits 重新算当前模型 logprob：

```python
_, log_probs_and_entropy = get_log_probs_and_entropy(
    logits,
    unconcat_tokens=batch["unconcat_tokens"],
    total_lengths=batch["total_lengths"],
    response_lengths=batch["response_lengths"],
    ...
)
```

然后和 batch 里的：

```text
advantages
old log_probs / rollout_log_probs
ref_log_probs
loss_masks
```

一起算 PPO / GRPO policy loss。

最后 Megatron schedule 会把 `forward_step_calc_loss(...)` 返回的 `output_tensor` 当成 loss tensor 继续走 backward：

```text
forward_step(...)
  -> output_tensor = loss
  -> backward_step(..., output_tensor, ...)
```

所以完整链路是：

```text
train_one_step
  -> forward_backward_func
    -> Megatron schedule forward_step
      -> Slime forward_step
        -> get_batch
        -> model(**forward_kwargs)
        -> return logits, loss_func
      -> forward_step_calc_loss
        -> loss_func(logits)
        -> Slime loss_function
        -> policy_loss_function / sft_loss_function / value_loss_function
      -> backward_step(loss)
  -> optimizer.step()
```

`losses_reduced` 不是 backward 用的 loss 本体，而是 `forward_step_calc_loss(...)` 收集的 logging metrics。训练 step 结束后，Slime 在 pipeline last stage 上对它做 reduce，作为日志返回。

## 14. 看代码时的顺序

建议按这个顺序读：

```text
1. slime/ray/rollout.py
   看 train_data / rollout_data 里有哪些字段。

2. slime/backends/megatron_utils/actor.py
   看 train_actor 怎么安排 ref logprob、actor logprob、advantage、train。

3. slime/backends/megatron_utils/data.py
   看 get_data_iterator / get_batch，重点是 pack、loss mask、PackedSeqParams。CP 细节看 13 文档。

4. slime/backends/megatron_utils/model.py
   看 train_one_step.forward_step 怎么把 batch 传给 model。

5. Megatron-LM/megatron/core/models/gpt/gpt_model.py
   看 GPTModel.forward 的 preprocess / decoder / postprocess。

6. Megatron-LM/megatron/core/transformer/transformer_block.py
   看 layer 循环。

7. Megatron-LM/megatron/core/transformer/transformer_layer.py
   看 self_attention 和 mlp 两个核心调用。

8. slime_plugins/models/qwen3_5.py
   看 Qwen3.5 linear attention forward。

9. Megatron-LM/megatron/core/transformer/moe/moe_layer.py
    看 MoE route / dispatch / expert / combine。

10. Megatron-LM/megatron/core/transformer/moe/router.py 和 moe_utils.py
    看 TopKRouter 注册 replay、forward pre-hook 触发以及 compute_topk wrapper 的调用位置。

11. slime/utils/routing_replay.py
    看 R3 怎么 record / pop_forward / pop_backward。

12. slime/backends/megatron_utils/loss.py
    看 logits 如何对齐 response tokens 并算 PPO/SFT/value loss。

13. slime/utils/ppo_utils.py
    看当前 `_VocabParallelLogProbEntropy` 为什么保存 full-vocab softmax，以及 backward 如何原地构造 grad logits。

14. Megatron-LM/megatron/core/transformer/linear_cross_entropy.py
    看 `LinearCrossEntropyModule` 如何在普通 ColumnParallelLinear 和 fused linear CE 之间切换。

15. Megatron-LM/megatron/core/fusions/fused_linear_cross_entropy.py
    看 fused linear CE 的 autograd 入口、保存张量以及 reduction="none" 接口。

16. Megatron-LM/megatron/core/fusions/linear_cross_entropy/blackwell/
    看 forward 按 vocab split 在线归约，以及 backward 重算 partial dlogits 的具体实现。
```

