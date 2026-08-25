# Full Attention 的 All-Gather CP：简化版和 TE 源码对照

这篇只讨论普通 dense/full attention，不讨论 DSA、SparseMLA、linear attention。

先给结论：

```text
1. FlashAttention varlen 不需要额外传 per-token start/end。
2. 直接调用 flash_attn_varlen 时，需要传和当前 q/k/v 坐标系一致的 cu_seqlens_q/k。
3. 如果 CP 后 q 是 local chunk，k/v 是 all-gather 后再 slice 出来的片段，
   那 cu_seqlens_q/k 通常要从全局 cu_seqlens 重新计算成 local-relative。
4. TE 的 cp_comm_type="all_gather" 是另一套实现。
   本地 TE 版本里，这条路径明确不支持 qkv_format="thd"。
   也就是说，它不是直接等价于 ring-flash-attn 这个 varlen 简化版。
```

## 1. 简化版：ring-flash-attn 的 varlen all-gather

参考文件：

```text
https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/llama3_flash_attn_varlen.py
```

这份代码可以理解成：

```text
每个 CP rank 持有一段 contiguous local Q/K/V。
forward 前 all-gather K/V。
然后用 local Q 和一段合适的 K/V 调 flash_attn_varlen。
```

### 1.1 输入坐标

假设全局 packed token 流是：

```text
tokens_global:
  [0, 1, 2, ..., T_global - 1]

cu_seqlens:
  Tensor[B + 1]
```

`cu_seqlens` 是全局样本边界。例如两个样本长度分别是 5 和 7：

```text
sample0: global positions [0, 5)
sample1: global positions [5, 12)

cu_seqlens = [0, 5, 12]
```

如果 `cp_size = 2`，contiguous 切分：

```text
rank0 local Q global positions [0, 6)
rank1 local Q global positions [6, 12)
```

注意 rank0 的 Q 跨了两个样本：

```text
rank0:
  sample0 positions [0, 5)
  sample1 positions [5, 6)
```

rank1 的 Q 是 sample1 的后半段：

```text
rank1:
  sample1 positions [6, 12)
```

### 1.2 为什么要重新算 cu_seqlens_q/k

FlashAttention varlen API 的约定是：

```python
flash_attn_varlen_func(
    q,                # [total_q, nheads, headdim]
    k,                # [total_k, nheads_k, headdim]
    v,                # [total_k, nheads_k, headdim]
    cu_seqlens_q,     # 当前 q 张量里的样本边界
    cu_seqlens_k,     # 当前 k/v 张量里的样本边界
    max_seqlen_q,
    max_seqlen_k,
    causal=True,
)
```

这里的 `cu_seqlens_q/k` 不是“永远全局坐标”，而是要和当前传进去的 `q/k/v` 张量对齐。

所以当本 rank 的 `q` 已经变成 local chunk，`k/v` 又是 all-gather 后被 slice 过的一段时，不能无脑把全局 `[0, 5, 12]` 原样传进去。

简化版代码里的 `llama3_flash_attn_prepare_cu_seqlens(...)` 就是在做这件事：

```text
输入:
  全局 cu_seqlens
  当前 rank 的 local query 范围
  causal/window 信息

输出:
  cu_seqlens_q       # local Q 坐标系里的边界
  cu_seqlens_k       # sliced K 坐标系里的边界
  max_seqlen_q
  max_seqlen_k
  local_k_slice      # 从 all-gather 后 K/V 里取哪段
```

这不是 DSA 那种 per-token `start/end`。它还是 FlashAttention varlen 的常规边界格式。

### 1.3 例子：两个样本 `[5, 7]`，cp=2

全局：

```text
cu_seqlens_global = [0, 5, 12]
T_global = 12
cp_size = 2
chunk_len = 6
```

rank0：

```text
local Q global range:
  [0, 6)

local Q 里面的样本切分:
  sample0: [0, 5)  长度 5
  sample1: [5, 6)  长度 1

因此传给 flash_attn_varlen 的 cu_seqlens_q:
  [0, 5, 6]
```

因为 causal attention 下，rank0 的 query 最远只需要看到 global position 6 之前的 K：

```text
K global range:
  [0, 6)

K 里面的样本切分:
  sample0: [0, 5)  长度 5
  sample1: [5, 6)  长度 1

cu_seqlens_k:
  [0, 5, 6]

local_k_slice:
  slice(0, 6)
```

rank1：

```text
local Q global range:
  [6, 12)

local Q 只包含 sample1 的后半段:
  sample1 suffix: [6, 12)  长度 6

cu_seqlens_q:
  [0, 6]
```

rank1 的 Q 是 sample1 的后半段，但为了算 causal attention，它需要 sample1 从头到当前 token 的 K，所以 K 要包含 global `[5, 12)`：

```text
K global range:
  [5, 12)

K 只包含 sample1:
  长度 7

cu_seqlens_k:
  [0, 7]

local_k_slice:
  slice(5, 12)
```

这里有一个关键点：rank1 的 `seqlen_q = 6`，`seqlen_k = 7`。`causal=True` 时 FlashAttention 使用 bottom-right causal 对齐。也就是这 6 个 query 对应 sample1 的 positions `[6, 12)`，它们相对 K `[5, 12)` 是后 6 行，而不是从 sample1 位置 0 重新开始。

所以它不需要你传：

```text
query0 start/end
query1 start/end
...
```

只要 `cu_seqlens_q=[0,6]`、`cu_seqlens_k=[0,7]`、`causal=True` 是对齐的，kernel 内部就能根据 varlen 边界和 causal 规则算正确。

这个例子最核心的点是：

```text
varlen all-gather CP 下，每个 rank 传给 FlashAttention 的 K/V 长度不一定等于 Q 长度。
```

在上面的例子里：

```text
全局原始样本边界:
  cu_seqlens_global = [0, 5, 12]

rank0:
  Q global range = [0, 6)
  K global range = [0, 6)
  seqlen_q = 6
  seqlen_k = 6
  cu_seqlens_q_local = [0, 5, 6]
  cu_seqlens_k_local = [0, 5, 6]

rank1:
  Q global range = [6, 12)
  K global range = [5, 12)
  seqlen_q = 6
  seqlen_k = 7
  cu_seqlens_q_local = [0, 6]
  cu_seqlens_k_local = [0, 7]
```

rank1 的 Q 是 sample1 的后半段，但它需要看到 sample1 从开头开始的 K，所以 K/V 比 Q 多了 sample1 的第一个 token。FlashAttention varlen 本来就支持 `seqlen_q != seqlen_k`；只要 `cu_seqlens_q/k` 和实际传入的 `q/k/v` 对齐，并且 causal mask 用 bottom-right 对齐，这个结果就是对的。

对比关系是：


| rank  | 全局原始 `cu_seqlens` | local Q 对应的 global range | local K 对应的 global range | 传给 FA 的 `cu_seqlens_q` | 传给 FA 的 `cu_seqlens_k` |
| ----- | ----------------- | ------------------------ | ------------------------ | ---------------------- | ---------------------- |
| rank0 | `[0, 5, 12]`      | `[0, 6)`                 | `[0, 6)`                 | `[0, 5, 6]`            | `[0, 5, 6]`            |
| rank1 | `[0, 5, 12]`      | `[6, 12)`                | `[5, 12)`                | `[0, 6]`               | `[0, 7]`               |


这里的 `cu_seqlens_q_local/cu_seqlens_k_local` 都是相对于传入 FlashAttention 的局部 `q/k/v` 张量重新计数，不再是全局 token position。

### 1.4 简化版 forward 逻辑

可以抽象成：

```python
cu_seqlens_q, cu_seqlens_k, max_q, max_k, local_k_slice = prepare(global_cu_seqlens)

k_all = all_gather(k_local)
v_all = all_gather(v_local)

k_part = k_all[local_k_slice]
v_part = v_all[local_k_slice]

out_local = flash_attn_varlen_func(
    q_local,
    k_part,
    v_part,
    cu_seqlens_q,
    cu_seqlens_k,
    max_q,
    max_k,
    causal=True,
)
```

forward 输出是：

```text
out_local:
  只对应本 rank 的 local Q
```

所以 forward 后不需要对 output 做 CP reduce。每个 rank 算的是不同 query 行的结果。

backward 时，K/V 是 all-gather 过的，所以每个 rank 都可能对别的 rank 的 K/V 产生梯度贡献。因此 backward 需要把全局 K/V 梯度再 reduce-scatter 回各自 rank。

## 2. FlashAttention varlen API 本身如何使用 cu_seqlens

本地源码：

```text
/mnt/shared-storage-user/huanghaian/miniconda3/envs/slime_pt211/lib/python3.12/site-packages/flash_attn/flash_attn_interface.py
```

`flash_attn_varlen_func` 的参数说明写得很直接：

```text
q:
  [total_q, nheads, headdim]

k/v:
  [total_k, nheads_k, headdim]

cu_seqlens_q:
  [batch_size + 1]，用于索引 q 里的每个样本

cu_seqlens_k:
  [batch_size + 1]，用于索引 k/v 里的每个样本
```

对应源码位置：

```text
flash_attn_interface.py:1370
flash_attn_interface.py:1411
flash_attn_interface.py:1414
flash_attn_interface.py:1416
```

在 AMD Triton 参考实现里，可以直接看到 kernel 从 `cu_seqlens_q/k` 读每个 batch item 的边界：

```python
cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start

cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
```

对应源码：

```text
flash_attn_triton_amd/fwd_prefill.py:266
```

这说明普通 varlen attention 的样本边界就是 `cu_seqlens`。没有额外的 per-token `start/end` 参数。

## 3. Megatron 到 TE 的调用链

在 slime 里，模型 forward 最终会进 Megatron attention：

```text
Megatron SelfAttention
  -> self.core_attention(...)
  -> TEDotProductAttention.forward(...)
  -> transformer_engine.pytorch.DotProductAttention.forward(...)
```

Megatron wrapper 位置：

```text
Megatron-LM/megatron/core/extensions/transformer_engine.py
```

构造 `TEDotProductAttention` 时，如果 `context_parallel_size > 1`，Megatron 会把 CP group 和 `cp_comm_type` 传给 TE：

```python
extra_kwargs["cp_group"] = pg_collection.cp
extra_kwargs["cp_global_ranks"] = torch.distributed.get_process_group_ranks(pg_collection.cp)
extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream
extra_kwargs["cp_comm_type"] = cp_comm_type
```

对应源码：

```text
Megatron-LM/megatron/core/extensions/transformer_engine.py:944
Megatron-LM/megatron/core/extensions/transformer_engine.py:950
Megatron-LM/megatron/core/extensions/transformer_engine.py:968
```

forward 时，如果有 `PackedSeqParams`，Megatron wrapper 会把里面的字段转成 keyword args 传给 TE：

```python
packed_seq_kwargs = {
    key: getattr(packed_seq_params, key)
    for key in self.kept_packed_seq_params
}

core_attn_out = super().forward(
    query,
    key,
    value,
    attention_mask,
    **packed_seq_kwargs,
)
```

对应源码：

```text
Megatron-LM/megatron/core/extensions/transformer_engine.py:1079
Megatron-LM/megatron/core/extensions/transformer_engine.py:1111
```

## 4. TE 的 cp_comm_type="all_gather" 具体逻辑

TE CP 源码位置：

```text
/mnt/shared-storage-user/huanghaian/miniconda3/envs/slime_pt211/lib/python3.12/site-packages/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py
```

总入口是：

```python
attn_forward_func_with_cp(...)
```

它根据 `cp_comm_type` 分派：

```python
if cp_comm_type in ["p2p", "a2a+p2p"]:
    out = AttnFuncWithCPAndKVP2P.apply(*args)
elif cp_comm_type == "all_gather":
    out = AttnFuncWithCPAndKVAllGather.apply(*args)
elif cp_comm_type == "a2a":
    out = AttnFuncWithCPAndQKVOA2A.apply(*args)
```

对应源码：

```text
context_parallel.py:4630
context_parallel.py:4819
```

### 4.1 重要限制：本地 TE all_gather 不支持 thd

`AttnFuncWithCPAndKVAllGather.forward` 一开始就有：

```python
assert qkv_format != "thd", f"No support for cp_comm_type='all_gather' and {qkv_format=}."
assert "padding" not in attn_mask_type
```

对应源码：

```text
context_parallel.py:3064
context_parallel.py:3065
```

这意味着：

```text
本地 TE 版本的 cp_comm_type="all_gather"
  支持普通 bshd/sbhd fixed-shape attention；
  不支持 packed varlen thd；
  不支持 padding/padding_causal mask。
```

所以它和前面的 `llama3_flash_attn_varlen.py` 简化版不是完全同一条路径。

简化版关注：

```text
contiguous CP + varlen thd + flash_attn_varlen + local-relative cu_seqlens
```

TE all_gather 关注：

```text
DualChunkSwap CP layout + fixed-shape bshd/sbhd + KV all-gather
```

这里要注意：不支持 `thd` 不是因为 FlashAttention 本身不能算 varlen。FlashAttention 有 varlen 接口。真正缺的是 TE 这个通用 CP wrapper 在 `cp_comm_type="all_gather"` 下，没有实现 packed THD 的 all-gather 后元数据重建、chunk 切片、mask/window、backward 梯度归并等完整语义。

如果只做 GLM5.2/Slime 这种更窄的专用实现，难度并不大。GLM5.2 的做法是：

```text
local Q
global gathered KV / index_k
per-token starts/ends 表达每个 Q 能看的 K 范围
lighting_indexer + SparseMLA 自己消费这套 global index 空间
```

但 TE 的 `cp_comm_type="all_gather"` 是 dense attention 的通用生产路径，它还要同时覆盖：

```text
bshd/sbhd/thd
causal/non-causal/sliding window
FlashAttention/FusedAttention/FP8
dropout rng 和 softmax_lse 保存
DualChunkSwap CP layout
backward 里 dKV reduce-scatter 回原 CP rank
```

所以这里更准确的判断是：

```text
专门支持一个 Slime/GLM5.2 风格的 varlen all-gather CP，不是理论难题；
把 TE 通用 cp_comm_type="all_gather" 扩展到 thd varlen，是工程量和覆盖矩阵问题。
```

这也解释了 slime 为什么仍然可以把 packed 样本传进去训练：

```text
slime packed 样本:
  qkv_format = "thd"
  packed_seq_params = PackedSeqParams(cu_seqlens_q, cu_seqlens_kv, ...)

普通 dense attention + CP:
  默认 cp_comm_type = "p2p"
  不是 cp_comm_type = "all_gather"
```

也就是说，slime 的 packed 样本不是走本节这个 TE `AttnFuncWithCPAndKVAllGather`。它走的是 TE CP 的 p2p/ring 路径：

```text
attn_forward_func_with_cp(...)
  └── cp_comm_type in ["p2p", "a2a+p2p"]
      └── AttnFuncWithCPAndKVP2P.apply(...)
```

对应 TE 源码分派：

```text
context_parallel.py:4805
context_parallel.py:4818
```

Slime 的 `get_batch` 会固定构造：

```python
PackedSeqParams(
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_kv=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_kv=max_seqlen,
    qkv_format="thd",
)
```

对应源码：

```text
slime/backends/megatron_utils/data.py:107
```

Megatron wrapper 再把这些字段透传给 TE：

```text
Megatron-LM/megatron/core/extensions/transformer_engine.py:1079
Megatron-LM/megatron/core/extensions/transformer_engine.py:1111
```

TE backend 收到 `thd` 后，如果 `cu_seqlens_q_padded/cu_seqlens_kv_padded` 没有显式传，会先补成普通 `cu_seqlens_q/cu_seqlens_kv`：

```python
if (q_format == "thd" or "padding" in attn_mask_type) and cu_seqlens_q_padded is None:
    cu_seqlens_q_padded = cu_seqlens_q
if (kv_format == "thd" or "padding" in attn_mask_type) and cu_seqlens_kv_padded is None:
    cu_seqlens_kv_padded = cu_seqlens_kv
```

对应源码：

```text
transformer_engine/.../dot_product_attention/backends.py:2076
```

所以不要把下面两个开关混起来：

```text
Megatron/TE cp_comm_type="all_gather":
  TE dense attention 的一种 CP 通信实现。
  本地版本不支持 thd packed。

slime --allgather-cp:
  Slime 自己的数据布局开关。
  当前源码只允许 DSA attention 模型在 CP>1 时使用。
  这条路径不是 TE dense attention 的 cp_comm_type="all_gather"。
```

### 4.1.1 GLM5.2 为什么开了 `--allgather-cp`，但不是 TE all_gather

GLM5.2 脚本里确实开启了：

```text
--allgather-cp
```

对应源码：

```text
scripts/models/glm5.2-744B-A40B.sh:61
```

但这个参数首先进入的是 Slime 自己的数据准备逻辑：

```python
if allgather_cp:
    tokens = torch.cat(tokens, dim=0)
    cu_seqlens = torch.tensor(cu_seqlens_list, ...)
    tokens = tokens.chunk(cp_size, dim=0)[cp_rank]
```

对应源码：

```text
slime/backends/megatron_utils/data.py:69
```

这一步的完整含义是：

```text
1. 先把 micro batch 里的所有样本按原始顺序拼成一个 global packed token stream。
2. cu_seqlens 记录这个 global stream 里每个样本的边界。
3. 如果需要 pad，也是在 global stream 尾部补 pad。
4. 最后再把 global stream 按 CP rank 做 contiguous chunk。
```

也就是说，`--allgather-cp` 在 Slime 这里不是 Megatron/TE 的 attention kernel 选项，而是一个数据布局约定：

```text
global packed stream:
  [sample0][sample1][sample2]...[pad]

rank0 local tokens:
  global stream 的第一段 contiguous chunk

rank1 local tokens:
  global stream 的第二段 contiguous chunk

...
```

这样做的核心原因是：后面 GLM5.2 自定义 DSA attention 会把每个 CP rank 的 K/V、index_k 再 all-gather 回 global 顺序。如果数据侧本来就是 contiguous chunk，那么 gather 后只需要按 rank 顺序拼起来，就能恢复原来的 global packed stream。

它不会自动设置：

```text
Megatron/TE cp_comm_type = "all_gather"
```

GLM5.2 也没有走普通 TE dense attention。GLM5.2 的 `--spec` 是：

```text
--spec "slime_plugins.models.glm5.glm5" "get_glm5_spec"
```

在 `get_glm5_spec` 里，Slime 从 Megatron 默认 block spec 起步，但把每一层的 `self_attention` 替换成了：

```python
ModuleSpec(module=DSAMLASelfAttention, ...)
```

对应源码：

```text
slime_plugins/models/glm5/glm5.py:754
slime_plugins/models/glm5/glm5.py:775
```

所以 GLM5.2 attention 的真实路径是：

```text
TransformerLayer
  └── self_attention = slime_plugins.models.glm5.glm5.DSAMLASelfAttention
      └── 自定义 DSA / SparseMLA forward
```

不是：

```text
TransformerLayer
  └── Megatron SelfAttention
      └── TEDotProductAttention
          └── TE cp_comm_type="all_gather"
```

下面看真正的实现。

#### 4.1.1.1 数据侧：global packed 后再 contiguous 切分

在 `slime/backends/megatron_utils/data.py` 中，`allgather_cp=True` 时先算 global `cu_seqlens`：

```python
cu_seqlens_list: list[int] = [0]
for t in tokens:
    cu_seqlens_list.append(cu_seqlens_list[-1] + t.size(0))
```

然后拼成一个全局 packed token 流：

```python
tokens = torch.cat(tokens, dim=0)
```

如果需要 pad，pad 也加到 global stream 的最后：

```python
tokens = F.pad(tokens, (0, pad), value=pad_token_id)
cu_seqlens_list.append(cu_seqlens_list[-1] + pad)
```

最后才切给不同 CP rank：

```python
cu_seqlens = torch.tensor(cu_seqlens_list, dtype=torch.int, device=torch.cuda.current_device())
tokens = tokens.chunk(cp_size, dim=0)[cp_rank]
```

注意这里的 `cu_seqlens` 不是本 rank 的局部边界，而是 global packed stream 的样本边界。随后构造 `PackedSeqParams` 时，Q 和 KV 都用这一份 global 边界：

```python
packed_seq_params = PackedSeqParams(
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_kv=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_kv=max_seqlen,
    qkv_format="thd",
)
```

这和非 `allgather_cp` 路径不同。普通 CP 路径会对每个样本调用 `slice_with_cp`，它使用前后双 chunk 的 zigzag 切法；而 GLM5.2 DSA 路径需要的是简单 contiguous 切法，方便后续 all-gather 还原 global token 顺序。

#### 4.1.1.2 Attention 侧：local Q，global K/V

GLM5.2 里真正的 CP all-gather 发生在自定义 attention 内部：

```python
k_pos_emb = gather_from_sequence_parallel_region(
    k_pos_emb,
    group=parallel_state.get_context_parallel_group(),
)
kv_compressed = gather_from_sequence_parallel_region(
    kv_compressed,
    group=parallel_state.get_context_parallel_group(),
)
index_k = gather_from_sequence_parallel_region(
    index_k,
    group=parallel_state.get_context_parallel_group(),
)
```

对应源码：

```text
slime_plugins/models/glm5/glm5.py:572
slime_plugins/models/glm5/glm5.py:573
slime_plugins/models/glm5/glm5.py:632
```

更具体地说，`DSAMLASelfAttention.forward` 先调用：

```python
q, kv, wv, index_query, index_key, head_weights = self.get_absorb_query_key_value_tensors(
    hidden_states,
    key_value_states,
    position_ids,
    packed_seq_params,
    inference_context=inference_context,
)
```

这里输入的 `hidden_states` 是本 CP rank 的 local token：

```text
hidden_states:
  [T_local, 1, hidden_size]
```

`get_absorb_query_key_value_tensors` 里面会构造两类东西：

```text
普通 attention 计算用:
  q
  kv
  wv

DSA 选 topk key 用:
  index_query
  index_key
  head_weights
```

其中 Q 侧保持 local：

```text
q:
  [T_local, num_heads_per_tp, q_dim]

index_query:
  [T_local, index_heads_per_tp, index_dim]
```

K/V 侧会跨 CP group all-gather：

```python
k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb, group=parallel_state.get_context_parallel_group())
kv_compressed = gather_from_sequence_parallel_region(
    kv_compressed, group=parallel_state.get_context_parallel_group()
)
index_k = gather_from_sequence_parallel_region(index_k, group=parallel_state.get_context_parallel_group())
```

所以 gather 后：

```text
kv:
  [T_global_padded, 1, kv_dim]

index_key:
  [T_global_padded, 1, index_dim]
```

这就是 all-gather 后的第一步结论：

```text
每个 CP rank 只负责自己 local Q 对应 token 的输出，
但每个 CP rank 都拿到了完整 global K/V 和完整 global index_key。
```

#### 4.1.1.3 RoPE：local Q 和 gathered K 的位置不能错

GLM5.2 这段实现里还有一个容易忽略的点：RoPE 需要按 global packed sequence 的位置来加。

先区分两个名字：

```text
q_pos_emb:
  从 q 里 split 出来的那部分需要加 RoPE 的 Q feature。
  它不是 position embedding 表。
  shape 是 [T_local, heads, rope_dim]。

rotary_pos_emb:
  RoPE 的 position lookup table。
  packed THD 下，它的长度是 max_seqlen, pack 中最长的样本的序列长度，而不是 T_local 或 T_global。
  shape 约为 [max_seqlen, 1, 1, rope_dim]。
```

例如：

```text
sample0 length = 5
sample1 length = 7
global packed length = 12
cp_size = 2

rotary_seq_len = max_seqlen = 7
rotary_pos_emb shape = [7, 1, 1, rope_dim]
```

它表达的是每个样本内部 position 0..6 的 RoPE 表。至于 global packed token 6 应该用 sample 内 position 1，还是 position 6，不是 `rotary_pos_emb` 自己知道的，而是由 `cu_seqlens` 告诉 fused THD RoPE kernel。

源码里定义了 `fuse_rope(q, cu_seqlens, gathered=False)`：

```python
if gathered:
    return fused_apply_rotary_pos_emb_thd(t, cu_seqlens, rotary_pos_emb.squeeze(0))
else:
    seq_len = q.shape[0]
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()
    t = t.repeat(cp_size, 1, 1)
    out = fused_apply_rotary_pos_emb_thd(t, cu_seqlens, rotary_pos_emb.squeeze(0))
    return out[cp_rank * seq_len : (cp_rank + 1) * seq_len]
```

它对 Q 和 K 的处理不同：

```python
q_pos_emb = fuse_rope(q_pos_emb, cu_seqlens_q, gathered=False)
k_pos_emb = fuse_rope(k_pos_emb, cu_seqlens_kv, gathered=True)
```

含义是：

```text
K 已经 all-gather 成 global token stream:
  直接用 global cu_seqlens 加 RoPE。

Q 还是 local token:
  fused THD RoPE kernel 只有 global cu_seqlens，没有单独传本 rank 的 global offset；
  所以先临时构造一个长度等于 T_global 的输入；
  用 global cu_seqlens 让 kernel 按 packed 样本边界计算正确的样本内 position；
  最后再切回本 rank 对应的 global slice。
```

这个实现依赖前面的 contiguous CP split。因为 rank 的 local token 正好对应 global stream 中连续的一段：

```text
rank r 的 local Q global range:
  [r * T_local, (r + 1) * T_local)
```

所以 `out[cp_rank * seq_len : (cp_rank + 1) * seq_len]` 能拿回这个 rank 的正确 RoPE 结果。

这里 `repeat(cp_size)` 不是数学上必须的，它只是一个很省事的实现技巧。真正必须满足的是：

```text
1. 传给 fused_apply_rotary_pos_emb_thd 的输入长度要能和 global cu_seqlens 对齐。
2. 最后切出来的本 rank global slice 上，必须是本 rank 的 local Q。
```

因为 RoPE 是逐 token 的旋转，不会让 token 之间互相影响，所以其他 slice 上放什么值并不影响本 rank 最后取出的结果。理论上也可以这样实现：

```text
full_q = zeros([T_global, heads, rope_dim])
full_q[cp_rank * T_local : (cp_rank + 1) * T_local] = local_q
out = fused_apply_rotary_pos_emb_thd(full_q, global_cu_seqlens, rotary_pos_emb)
local_out = out[cp_rank * T_local : (cp_rank + 1) * T_local]
```

这和源码里的 `repeat + slice` 在本 rank 取出的那段上等价：

```text
rank0:
  repeat 后 [local_q, local_q]
  取 [0, T_local) -> local_q 的正确 RoPE 结果

rank1:
  repeat 后 [local_q, local_q]
  取 [T_local, 2 * T_local) -> local_q 的正确 RoPE 结果
```

但不能简单理解成“全 0 pad 一下也一定可以”。如果构造的是：

```text
[local_q, zeros, zeros, ...]
```

那么 rank0 取第一段可能没问题，rank1 取第二段就会取到 zeros。全 0 方案只有在把本 rank 的 local Q 放到它对应的 global slice 上时才等价。

所以这段代码的核心不是 `repeat` 本身，而是：

```text
用长度对齐 global cu_seqlens 的临时 tensor 调 fused THD RoPE，
让 kernel 算出正确的样本内 position，
并保证最后切回来的 slice 正好是本 rank 的 local Q。
```

#### 4.1.1.4 all-gather 后：用 start/end 限制每个 Q 能看哪些 K

拿到 global `index_key` 以后，不能让每个 Q 直接对所有 K 做 topk。因为 packed stream 里有多个样本，而且 causal attention 只能看同一个样本内当前位置之前的 token。

这里源码调用：

```python
starts, ends = generate_varlen_mask_params(packed_seq_params.cu_seqlens_q)
```

`generate_varlen_mask_params` 的逻辑是：

```python
seq_len = cu_seqlens[-1].item()
q_indices = torch.arange(0, seq_len, device=cu_seqlens.device)
seq_indices = torch.searchsorted(cu_seqlens, q_indices, right=True) - 1
starts = cu_seqlens[seq_indices]
ends = q_indices + 1
```

这里的 `cu_seqlens` 是 global packed stream 的样本边界，所以：

```text
cu_seqlens[-1]:
  global packed stream 的总 token 数，包含可能的 global pad。

q_indices:
  global packed stream 里每个 token 的全局下标。

seq_indices:
  每个 global token 属于第几个 packed sample。
```

`torch.searchsorted(cu_seqlens, q_indices, right=True) - 1` 的作用就是把 global token 下标映射回样本编号。这里必须用 `right=True`，因为样本边界上的 token 应该归到后一个样本：

```text
cu_seqlens = [0, 5, 12]

global token 4:
  sample0 的最后一个 token

global token 5:
  sample1 的第一个 token
```

也就是说，它先基于 global `cu_seqlens` 为 global stream 里的每个 token 算：

```text
starts[i] = token i 所属样本的起点坐标
ends[i]   = token i 在 causal attention 下能看到的终点坐标，也就是 i + 1
```

这正好表达 full causal attention 的可见范围：

```text
token i 可以看:
  K[starts[i] : ends[i])
```

然后再把 `starts` 和 `ends` 按 CP rank 切回 local：

```python
starts = scatter_to_sequence_parallel_region(starts, group=parallel_state.get_context_parallel_group())
ends = scatter_to_sequence_parallel_region(ends, group=parallel_state.get_context_parallel_group())
```

所以进入 indexer 的是：

```text
index_query:
  local Q 对应的 index query
  [T_local, index_heads_per_tp, index_dim]

index_key:
  gathered global index key
  [T_global_padded, index_dim]

starts / ends:
  local Q 对应的 global K 可见范围
  [T_local]
```

#### 4.1.1.5 topk_indices：在 global K/V 下标空间里选

随后 `fused_select_topk` 按 block 调用 `lighting_indexer`：

```python
indexer_topk_scores_block, topk_indices_block = lighting_indexer(
    index_q_block,
    index_k,
    w_block,
    starts_block.to(torch.int32),
    ends_block.to(torch.int32),
    self.index_topk,
    topk_indices=None,
)
```

这一步的关键是：`index_k` 是 global gathered 的，`starts_block` / `ends_block` 也是 global 下标。因此 `topk_indices_block` 返回的是 global K/V token 下标。

`lighting_indexer` 可以理解成 DSA 里的 varlen causal top-k indexer，但它不是完整的 attention output kernel。它不直接计算：

```text
softmax(QK^T)V
```

而是先计算：

```text
index_q 和 index_k 的相关性 logits
```

再根据 `starts_block` / `ends_block` 做 varlen causal mask，最后选 topk K/V 下标。

底层源码里，`indexer_fwd_interface` 会先构造：

```text
logits shape = [T_local_block, T_global_kv]
```

然后 `clean_logits_kernel` 按每个 Q token 的可见范围清理 logits：

```python
if idx < cu_k_s or idx >= cu_k_e:
    Logits[bx, idx] = -inf
```

这里：

```text
cu_k_s = starts_block[bx]
cu_k_e = ends_block[bx]
```

所以对第 `bx` 个 local Q 来说：

```text
只保留 K[starts_block[bx] : ends_block[bx])
其他 global K 位置全部置为 -inf
```

Python 层再做：

```python
index_score, topk_indices = torch.topk(logits, topk, dim=-1)
```

因此它和 varlen flash attention 的相似点是：

```text
都用边界信息避免 packed 样本之间互相看见；
都能表达 causal attention 的前缀可见范围。
```

但区别是：

```text
varlen flash attention:
  直接算 attention output。

lighting_indexer:
  只负责在合法 K 范围里选 topk_indices。
  真正的 attention output 后面由 SparseMLA.apply(q, kv, topk_indices, scale) 计算。
```

形状可以理解成：

```text
topk_indices:
  [T_local, 1, index_topk]
```

它表示：

```text
对本 rank 的每个 local Q token，
在它允许看的 global K 范围里，
选出最相关的 index_topk 个 K/V token。
```

如果开启了 index sharing，有些层不会重新算 topk，而是从 `packed_seq_params` 上的 holder 复用前面某个计算层保存的 `topk_indices`。但复用的仍然是这套 global K/V 下标空间。

#### 4.1.1.5.1 indexer 能不能通过主 loss 端到端训练

从当前 Slime 这份 GLM5.2 代码看，indexer 模块不能通过主 attention loss 端到端更新。

原因是 `lighting_indexer` 返回两个东西：

```python
return index_score, topk_indices
```

其中：

```text
index_score:
  topk 位置对应的连续 score，理论上可导。

topk_indices:
  topk 选出来的 K/V 下标，是 int index，不可导。
```

`IndexerFunction` 确实实现了 backward：

```python
def backward(ctx, grad_scores, grad_indices):
    index_q, index_k, weights, cu_seqlen_ks, cu_seqlen_ke, topk_indices = ctx.saved_tensors
    grad_q, grad_w, grad_k = indexer_bwd_interface(
        index_q,
        weights,
        index_k,
        topk_indices,
        grad_scores,
    )
    return grad_q, grad_k, grad_w, None, None, None, None, None, None, None
```

但这个 backward 接的是 `grad_scores`，也就是 `index_score` 的梯度。它不是对 `topk_indices` 求梯度。

GLM5.2 当前 forward 里实际把 `index_score` 丢掉了：

```python
_, topk_indices = fused_select_topk(index_query, index_key, head_weights, starts, ends)
```

后面只把 `topk_indices` 传给 `SparseMLA`：

```python
core_attn_out, _ = SparseMLA.apply(q, kv, topk_indices, self.softmax_scale)
```

而 `SparseMLA.backward` 明确不给 indices 返回梯度：

```python
return tl_dq, tl_dkv, None, None
```

所以主 loss 的梯度路径是：

```text
loss
  -> SparseMLA output
  -> q / kv
  -> MLA 主干参数
```

不是：

```text
loss
  -> topk_indices
  -> lighting_indexer
  -> indexer 参数
```

另外，indexer 输入前还做了 detach：

```python
q_compressed = q_compressed.detach()
hidden_states = hidden_states.detach()
rotary_pos_emb = rotary_pos_emb.detach()
```

这表示即使有某个额外的 indexer loss 使用 `index_score`，它也只会更新 indexer 自己的模块，例如：

```text
wq_b
wk
k_norm
weights_proj
```

不会把 indexer loss 反传回主干 hidden states。

因此这套设计更像下面两种训练方式之一。

第一种是两阶段训练：

```text
阶段 1：训练 indexer
  用 index_score 相关的辅助 loss / 蒸馏 loss / 监督 topk loss
  更新 wq_b / wk / k_norm / weights_proj
  其他主干通过 detach 被隔离

阶段 2：训练主模型
  固定或基本固定 indexer
  用 indexer 产出的 topk_indices 做 SparseMLA
  主 loss 更新 q / kv / wv / output projection / FFN / MoE 等主干参数
  但主 loss 不通过 topk_indices 端到端更新 indexer
```

第二种是联合训练但 loss 分路：

```text
主 LM/RL loss:
  更新主干参数

indexer auxiliary loss:
  通过 index_score 更新 indexer 参数
```

当前 Slime 这条 forward 没有看到 auxiliary loss，因为它直接写成：

```python
_, topk_indices = fused_select_topk(...)
```

所以基于这份代码能确定的是：

```text
IndexerFunction.backward 支持对 index_score 的可导训练；
topk_indices 本身不可导；
当前主训练 loss 只使用 topk_indices；
因此当前主 loss 不会端到端更新 indexer 模块。
```

#### 4.1.1.6 SparseMLA：local Q + global KV + local topk -> local output

最后真正算 attention output 的是：

```python
core_attn_out, _ = SparseMLA.apply(q, kv, topk_indices, self.softmax_scale)
core_attn_out = torch.einsum("thm,hdm->thd", core_attn_out, wv)
core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
output, bias = self.linear_proj(core_attn_out)
return output, bias
```

`SparseMLA.apply` 的输入约定是：

```text
q:
  [T_local, heads, dim]

kv:
  [T_global_padded, kv_group, dim]

topk_indices:
  [T_local, kv_group, index_topk]

output:
  [T_local, heads, value_dim]
```

所以 all-gather 之后不是再做一次普通 dense attention，而是：

```text
1. 每个 rank 保留 local Q。
2. 每个 rank 拿到 global KV。
3. 每个 rank 用 local Q 在 global index_key 里选 topk K/V。
4. SparseMLA 只为 local Q 计算输出。
5. 输出仍然是 [T_local, 1, hidden_size]。
```

这也解释了为什么 forward 结束时不需要把 attention output 再 CP all-reduce 或 all-gather：

```text
每个 rank 本来就只负责 global stream 中自己那段 token 的 hidden states。
attention 的输出也是这段 token 的输出。
后续层继续在这个 local token shard 上计算。
```

#### 4.1.1.7 一个具体例子

假设一个 micro batch 有两个样本：

```text
sample0 length = 5
sample1 length = 7
global packed stream length = 12
cp_size = 2
```

Slime 数据侧生成的 global `cu_seqlens` 是：

```text
global cu_seqlens = [0, 5, 12]
```

因为 `cp_size=2`，每个 rank 拿 6 个 token：

```text
rank0 local Q global range = [0, 6)
rank1 local Q global range = [6, 12)
```

注意 rank0 的 local Q 横跨了两个样本：

```text
rank0:
  token 0..4  属于 sample0
  token 5     属于 sample1 的第一个 token

rank1:
  token 6..11 属于 sample1 的后 6 个 token
```

进入 GLM5.2 DSA attention 后，每个 rank 会 all-gather K/V 和 index_key：

```text
rank0 gathered K/V global range = [0, 12)
rank1 gathered K/V global range = [0, 12)
```

然后基于 global `cu_seqlens = [0, 5, 12]` 生成全局未切分的 starts/ends：

```text
global positions:
  0  1  2  3  4 | 5  6  7  8  9 10 11

global starts:
  0  0  0  0  0 | 5  5  5  5  5  5  5

global ends:
  1  2  3  4  5 | 6  7  8  9 10 11 12
```

再 scatter 给各个 CP rank：

```text
rank0 local Q global range = [0, 6)
rank0 local starts = [0, 0, 0, 0, 0, 5]
rank0 local ends   = [1, 2, 3, 4, 5, 6]

rank1 local Q global range = [6, 12)
rank1 local starts = [5, 5, 5, 5, 5, 5]
rank1 local ends   = [7, 8, 9, 10, 11, 12]
```

所以每个 rank 上 indexer 实际看到的是：

```text
rank0:
  local Q = global [0, 6)
  global K/index_key = [0, 12)
  但每个 Q 只能在自己的 [start, end) 里选 topk

rank1:
  local Q = global [6, 12)
  global K/index_key = [0, 12)
  但每个 Q 只能在自己的 [start, end) 里选 topk
```

这就回答了前面那个关键问题：

```text
rank1 的 Q 是 sample1 的后半段。
为了 causal attention，它确实需要 sample1 从头到当前 token 的 K。
所以 rank1 的 K/V 可见范围比本地 Q range 更长。

例如 rank1 第一个 token 是 global position 6：
  它属于 sample1。
  它可以看 K[5:7)，也就是 sample1 的第 0、1 个 token。

rank1 最后一个 token是 global position 11：
  它可以看 K[5:12)，也就是完整 sample1。
```

而 rank0 对 sample1 的第一个 token，也只看：

```text
global position 5:
  starts = 5
  ends = 6
  可见 K[5:6)
```

因此，这条路径里 `seqlen_q` 和 `seqlen_k` 可以不一样：

```text
rank0:
  Q global range = [0, 6)
  K gathered range = [0, 12)
  但每个 Q 的实际 K 窗口由 starts/ends 决定。

rank1:
  Q global range = [6, 12)
  K gathered range = [0, 12)
  但每个 Q 的实际 K 窗口由 starts/ends 决定。
```

如果只看 rank1 的 sample1 后半段，可以把它等价理解成：

```text
rank1 local Q 覆盖 sample1 的后 6 个 token。
rank1 对这些 Q 需要的 K 覆盖 sample1 的前缀。
所以 rank1 对 sample1 的 K 可见范围最大是 global [5, 12)，长度 7。
```

但源码实现没有显式构造一个 rank1 专属的 `cu_seqlens_k = [0, 7]`。它采用的是：

```text
global K/V + local starts/ends
```

也就是用全局下标空间表达每个 local Q 的 causal 可见范围。

所以 GLM5.2 的 “allgather CP” 是：

```text
Slime 数据侧:
  contiguous CP split

GLM5.2 自定义 DSA attention:
  gather KV / index_k across CP group，恢复 global K/V 下标空间
  local Q 只保留本 rank 的 token
  用 global cu_seqlens 生成 starts/ends，再 scatter 回 local
  lighting_indexer 在 global K/V 下标空间里选 topk_indices
  SparseMLA 用 local Q + gathered KV + local topk_indices 计算 local output
```

它不是 TE dense attention 的：

```text
cp_comm_type = "all_gather"
```

所以看到这两个名字时要这样翻译：

```text
--allgather-cp:
  Slime/GLM DSA 路径的 all-gather CP 数据和自定义 attention 约定。

--cp-comm-type all_gather:
  Megatron/TE 普通 dense attention 的 CP 通信策略。
```

### 4.2 TE all_gather 的 shape 流程

以最简单的 full causal attention 为例：

```text
global sequence length S = 12
cp_size = 2
每个 CP rank 持有 S / cp_size = 6 个 token
```

TE 的注释说它用 DualChunkSwap 做负载均衡。对于 `cp_size=2`，每个 rank 实际持有两个 chunk：

```text
chunk_size = S / (2 * cp_size) = 12 / 4 = 3

全局 chunk:
  chunk0: positions [0, 3)
  chunk1: positions [3, 6)
  chunk2: positions [6, 9)
  chunk3: positions [9, 12)
```

TE 的 local chunk 分配是：

```text
rank0:
  chunk0 + chunk3
  positions [0,1,2, 9,10,11]

rank1:
  chunk1 + chunk2
  positions [3,4,5, 6,7,8]
```

这和“rank0 拿 [0,6)，rank1 拿 [6,12)”的 contiguous 简化版不一样。

TE 源码里：

```python
max_seqlen_q = max_seqlen_q // (2 * cp_size)
max_seqlen_kv = max_seqlen_kv // (2 * cp_size)
```

这里把全局 sequence length 变成 per-chunk length。例子里就是：

```text
12 // 4 = 3
```

对应源码：

```text
context_parallel.py:3116
```

然后 Q 被拆成两个本地 chunk：

```python
q = q.view(..., 2, q.shape[seq_dim_qkv] // 2, ...)
```

对应源码：

```text
context_parallel.py:3175
```

K/V 做 all-gather：

```python
k_ag, _ = gather_along_first_dim(k, cp_group)
v_ag, _ = gather_along_first_dim(v, cp_group)
```

对应源码：

```text
context_parallel.py:3182
```

all-gather 后再拆成 `2 * cp_size` 个 chunk，并按 attention 前需要的顺序重排：

```python
k_ag = k_ag.view(2 * cp_size, k.shape[0] // 2, *k.shape[1:])
chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering_before_attn(cp_size, k.device)
k_ag = torch.index_select(k_ag, dim=0, index=chunk_ids_for_kv_ag)
k_ag = k_ag.view(-1, *k.shape[1:])
```

对应源码：

```text
context_parallel.py:3185
context_parallel.py:3189
context_parallel.py:3193
```

### 4.3 每个 Q chunk 选择多少 K/V

TE all_gather 不是每次都把完整 K/V 都塞进 FlashAttention。它先算当前 Q chunk 需要的 KV range：

```python
kv_seq_range_per_step[i], window_size_per_step[i] =
    get_kv_seq_info_after_all_gather(...)

seq_start_idx, seq_end_idx = kv_seq_range_per_step[i]
k_part = k_ag[seq_start_idx:seq_end_idx]
v_part = v_ag[seq_start_idx:seq_end_idx]
```

对应源码：

```text
context_parallel.py:3226
context_parallel.py:3236
context_parallel.py:3246
```

所以这里要把“all-gather”分成两层理解：

```text
通信层面:
  k_ag / v_ag 是 all-gather 后的完整 K/V buffer。
  每个 CP rank 都拿到了重排后的全局 K/V。

kernel 调用层面:
  每个 Q chunk 只从 k_ag / v_ag 里 slice 自己需要的 [seq_start_idx, seq_end_idx)。
  传给 FlashAttention 的是 k_part / v_part，不一定是完整 K/V。
```

也就是说，TE all_gather 的语义不是“每个 Q chunk 都用全量 KV 算一遍”。更准确地说是：

```text
先通过通信准备好全局 K/V；
再按当前 local Q chunk 的 causal/window 依赖范围切出实际需要的 K/V；
最后只把这段 K/V 传给 attention kernel。
```

`get_kv_seq_info_after_all_gather(...)` 的核心逻辑是：

```python
local_chunk_end_idx = (local_chunk_id + 1) * max_seqlen_kv
full_seq_end_idx = max_seqlen_kv * cp_size * 2

if causal:
    window_size = (-1, 0)

seq_start_idx = 0
seq_end_idx = local_chunk_end_idx
```

对应源码：

```text
context_parallel.py:2991
context_parallel.py:2995
context_parallel.py:2998
context_parallel.py:3008
```

用前面的 `S=12, cp=2, chunk_size=3` 举例：

```text
rank0 local_seq_chunk_ids = [0, 3]

rank0 第一个 Q chunk = global [0,3)
  local_chunk_id = 0
  seq_end_idx = (0 + 1) * 3 = 3
  K range = [0,3)

rank0 第二个 Q chunk = global [9,12)
  local_chunk_id = 3
  seq_end_idx = (3 + 1) * 3 = 12
  K range = [0,12)
```

rank1：

```text
rank1 local_seq_chunk_ids = [1, 2]

rank1 第一个 Q chunk = global [3,6)
  K range = [0,6)

rank1 第二个 Q chunk = global [6,9)
  K range = [0,9)
```

这就是 full causal attention 的正确依赖范围。

### 4.4 FlashAttention 调用

TE 如果不走 fused attention，会调用 FlashAttention backend：

```python
fa_outputs = flash_attn_fwd(
    q_part,
    k_part,
    v_part,
    *fa_forward_args_thd,
    causal=causal,
    **fa_forward_kwargs,
)
```

对应源码：

```text
context_parallel.py:3319
```

对于 `qkv_format="bshd"`，这里不是 varlen thd，所以 `get_fa_args(...)` 不会传 `cu_seqlens_q/k`，而是传 fixed-shape 的 `max_seqlen_q/max_seqlen_kv`。

这就是 TE all_gather 和简化版 varlen 最大的区别：

```text
简化版:
  flash_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, ...)

TE all_gather fixed-shape:
  flash_attn(q, k, v, max_seqlen_q, max_seqlen_k, ...)
```

### 4.5 forward 输出和 backward 的 K/V 梯度

每个 step 的输出写回 `out_f16`：

```python
out_f16[:, i - 1].copy_(out_per_step[i - 1])
out_f16 = out_f16.view(orig_o_shape)
```

对应源码：

```text
context_parallel.py:3345
context_parallel.py:3369
```

forward 输出仍然只对应本 rank 的 local Q。

backward 里，K/V 梯度先累加到 all-gather 后的全局 K/V 坐标：

```python
dk[seq_start_idx:seq_end_idx].add_(dk_per_step[i - 1])
dv[seq_start_idx:seq_end_idx].add_(dv_per_step[i - 1])
```

然后 reduce-scatter 回本 rank：

```python
dk, _ = reduce_scatter_along_first_dim(dk, ctx.cp_group)
dv, _ = reduce_scatter_along_first_dim(dv, ctx.cp_group)
```

对应源码：

```text
context_parallel.py:3814
context_parallel.py:3831
```

这和简化版的原则一致：

```text
forward:
  local Q 得到 local output，不需要 reduce。

backward:
  gathered K/V 被多个 rank 使用，K/V 梯度需要 reduce-scatter。
```

## 5. 把两套逻辑放在一起看


| 项目                | ring-flash-attn 简化版                 | TE cp_comm_type="all_gather"                                 |
| ----------------- | ----------------------------------- | ------------------------------------------------------------ |
| 目标                | 直接演示 varlen all-gather CP           | TE/Megatron 生产实现之一                                           |
| QKV layout        | varlen/thd                          | 本地版本不支持 thd，主要是 bshd/sbhd                                    |
| CP token layout   | contiguous chunk                    | DualChunkSwap，两段 chunk/ rank                                 |
| 样本边界              | 重新算 local-relative `cu_seqlens_q/k` | fixed-shape 时不走 varlen cu_seqlens；fused path 另传内部 cu_seqlens |
| 是否需要 start/end    | 不需要                                 | 不需要                                                          |
| 是否 all-gather K/V | 是                                   | 是                                                            |
| forward 输出        | local Q 的 output                    | local Q 的 output                                             |
| backward K/V 梯度   | reduce-scatter                      | reduce-scatter                                               |


最终可以记成一句话：

```text
普通 full attention 的 all-gather CP，本质是 local Q + 可见范围内的 gathered K/V。
边界要么用 cu_seqlens 表达，要么用 fixed-shape chunk/window 表达。
它不需要 DSA 那种 per-token start/end。
```

更细一点：

```text
直接调用 flash_attn_varlen:
  你自己负责把全局 cu_seqlens 转成当前 q/k/v 坐标系下的 cu_seqlens_q/k。

走 Megatron/TE:
  Megatron 把 cp_group/cp_comm_type/PackedSeqParams 传给 TE；
  TE 根据 cp_comm_type 选择 p2p/all_gather/a2a；
  本地 TE 的 all_gather 路径不支持 thd varlen。
```

## 6. Qwen3.5 的 CP 路径

Qwen3.5 和 GLM5.2 不一样。Qwen3.5 没有开启 Slime 的 `--allgather-cp`。例如当前 Qwen3.5 35B RL 脚本里是：

```text
--context-parallel-size 1
```

对应源码：

```text
scripts/run-qwen3.5-35B-rl-eagle-hha.sh:79
```

所以这个脚本实际运行时没有 CP。

但代码层面，Qwen3.5 是有 CP 处理的。要分两类 attention 层看。

### 6.1 full_attention 层

Qwen3.5 的 spec 先从 Megatron 默认 GPT decoder block 开始：

```python
transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)
```

然后只替换 HF config 里 `layer_types == "linear_attention"` 的层：

```python
if text_config.layer_types[layer_id + offset] == "linear_attention":
    layer_specs.submodules.self_attention = ModuleSpec(
        module=Attention,
        params={"args": args},
    )
```

对应源码：

```text
slime_plugins/models/qwen3_5.py:206
slime_plugins/models/qwen3_5.py:225
```

所以：

```text
Qwen3.5 full_attention 层:
  没被 slime_plugins.models.qwen3_5.Attention 替换。
  仍然走 Megatron 默认 SelfAttention。
```

如果 `context_parallel_size > 1`，这些 full attention 层走 Megatron/TE 的 CP。默认 `--cp-comm-type` 是 `p2p`，不是 `all_gather`：

```text
Megatron/TE dense attention:
  cp_comm_type = "p2p"   # 默认
  qkv_format = "thd"     # Slime packed sample
```

也就是说，Qwen3.5 的 full attention CP 不是 GLM5.2 的 `--allgather-cp`，也不是 TE `cp_comm_type="all_gather"`。

### 6.2 linear_attention 层

Qwen3.5 的 linear attention 层被替换成：

```text
slime_plugins.models.qwen3_5.Attention
```

这个类继承：

```text
slime_plugins.models.hf_attention.HuggingfaceAttention
```

`HuggingfaceAttention.forward` 对 CP 的处理是：

```text
1. 如果 sequence_parallel:
     先 gather TP/SP 切分。

2. 如果 context_parallel_size > 1:
     把各 CP rank 的 hidden_states all-gather 起来。

3. 按 Slime 的 zigzag CP layout 重组为完整序列顺序。

4. 调用 Qwen3.5 的 hf_forward，也就是 GatedDeltaNet。

5. 把完整 output 再按当前 CP rank 切回 local chunk。

6. 如果 sequence_parallel:
     scatter 回 TP/SP 区域。
```

对应源码：

```text
slime_plugins/models/hf_attention.py:107
slime_plugins/models/hf_attention.py:117
slime_plugins/models/hf_attention.py:129
slime_plugins/models/hf_attention.py:143
slime_plugins/models/hf_attention.py:147
slime_plugins/models/hf_attention.py:152
slime_plugins/models/hf_attention.py:164
```

更具体地，CP gather 后它用 `cu_seqlens // cp_size` 和 zigzag chunk 顺序恢复完整序列：

```python
local_cu_seqlens = cu_seqlens // cp_size
for i in range(len(cu_seqlens) - 1):
    seqlen = cu_seqlens[i + 1] - cu_seqlens[i]
    chunk_size = seqlen // 2 // cp_size
    whole_hidden_states_list.extend(
        [
            hidden_states_list[cp_rank][local_cu_seqlens[i] : local_cu_seqlens[i] + chunk_size]
            for cp_rank in range(cp_size)
        ]
        + [
            hidden_states_list[cp_rank][local_cu_seqlens[i] + chunk_size : local_cu_seqlens[i + 1]]
            for cp_rank in range(cp_size)
        ][::-1],
    )
hidden_states = torch.cat(whole_hidden_states_list, dim=0)
```

对应源码：

```text
slime_plugins/models/hf_attention.py:129
```

Qwen3.5 的 `hf_forward` 里，把完整序列和 packed 边界传给 GatedDeltaNet：

```python
hidden_states = self.linear_attn(
    hidden_states=hidden_states,
    cu_seqlens=packed_seq_params.cu_seqlens_q,
)
```

对应源码：

```text
slime_plugins/models/qwen3_5.py:186
```

GatedDeltaNet 里 `cu_seqlens` 会继续传给短卷积和 `chunk_gated_delta_rule`：

```python
mixed_qkv, _ = self.conv1d(x=mixed_qkv, cu_seqlens=cu_seqlens)

core_attn_out, _ = self.chunk_gated_delta_rule(
    query,
    key,
    value,
    ...,
    cu_seqlens=cu_seqlens,
)
```

对应源码：

```text
slime_plugins/models/qwen3_5.py:101
slime_plugins/models/qwen3_5.py:130
```

### 6.3 Qwen3.5 CP 小结

Qwen3.5 如果开 CP，可以这样理解：

```text
full_attention 层:
  Megatron/TE 默认 dense attention CP
  通常是 p2p/ring，不是 all_gather

linear_attention 层:
  Slime HuggingfaceAttention 自己做 CP gather
  gather full sequence
  跑 GatedDeltaNet
  再 slice 回当前 CP rank
```

它和 GLM5.2 的区别：

```text
GLM5.2:
  --allgather-cp
  contiguous CP split
  DSAMLASelfAttention 自己 gather KV/index_k
  SparseMLA / DSA

Qwen3.5:
  默认不启用 --allgather-cp
  默认 CP layout 是 Slime/Megatron 的 zigzag CP
  full_attention 走 Megatron/TE p2p CP
  linear_attention 走 HuggingfaceAttention 的 gather-full-sequence 再 slice-back
```

## 7. TE 其他 CP 通信方式：p2p / a2a / a2a+p2p

前面重点看了 `cp_comm_type="all_gather"`。TE 里还支持几种 CP 通信方式：

```python
if cp_comm_type in ["p2p", "a2a+p2p"]:
    out = AttnFuncWithCPAndKVP2P.apply(*args)
elif cp_comm_type == "all_gather":
    out = AttnFuncWithCPAndKVAllGather.apply(*args)
elif cp_comm_type == "a2a":
    out = AttnFuncWithCPAndQKVOA2A.apply(*args)
```

对应源码：

```text
transformer_engine/.../context_parallel.py:4805
transformer_engine/.../context_parallel.py:4819
transformer_engine/.../context_parallel.py:4833
```

可以先用一句话区分：

```text
p2p:
  Q 留在本 rank，KV 沿 CP ring 一步步传过来，边通信边算。

all_gather:
  KV 先 all-gather 成全局 buffer，再按 Q chunk 切需要的 KV 算。

a2a:
  用 all-to-all 把 sequence 维 gather，同时把 head 维切开。
  attention kernel 在“全序列 + 部分 heads”的布局上算。

a2a+p2p:
  层级 CP。
  低层 CP group 用 a2a 切 heads / gather sequence；
  高层 CP group 再用 p2p ring 传 KV。
```

### 7.1 p2p：KV ring 传递

P2P 路径入口是：

```text
AttnFuncWithCPAndKVP2P
```

源码注释写得很直接：

```text
Exchange KV between CP ranks with P2P in ring topology.
Split attention compute into multiple steps, and overlap current-step compute with next-step communication.
```

对应源码：

```text
transformer_engine/.../context_parallel.py:1407
```

P2P 的核心不是把 KV 一次性 gather 全，而是：

```text
每个 rank 固定持有自己的 local Q。
每个 rank 一开始也只有自己的 local K/V。
然后 K/V 沿 ring 方向一步步传。
每一步拿当前手里的某个 K/V shard 和 local Q 算一个 partial attention。
最后把多个 partial attention output 用 softmax_lse 做数值正确的合并。
```

源码里 `cp_size = 4` 的注释把 causal 情况分成三类 tile：

```text
          step
section | 0  1  2  3
--------------------
   G  0 | d, u, u, u
   P  1 | l, d, u, u
   U  2 | l, l, d, u
      3 | l, l, l, d
```

对应源码：

```text
transformer_engine/.../context_parallel.py:1832
```

这里：

```text
d = diagonal tile:
  Q 和 K/V 来自同一个 CP rank 的局部序列。
  causal mask 仍然需要保留。

l = lower-triangle tile:
  K/V 在 Q 的历史侧。
  这块通常是全可见，不需要 causal mask。

u = upper-triangle tile:
  K/V 在 Q 的未来侧。
  对 causal attention 来说，只能处理部分情况，TE 会切 half 并做特殊 correction。
```

P2P 每一步会先准备当前 step 的 Q/K/V 和 varlen metadata：

```python
prepare_outputs = cp_p2p_fwd_prepare_qkv(*prepare_inputs, section)
```

对应源码：

```text
transformer_engine/.../context_parallel.py:1850
transformer_engine/.../context_parallel.py:1879
transformer_engine/.../context_parallel.py:1908
```

然后调用 FlashAttention 或 FusedAttention：

```python
out_per_step[i], softmax_lse_per_step[i], rng_states[i] =
    cp_p2p_fwd_flash_attn(...)
```

对应源码：

```text
transformer_engine/.../context_parallel.py:1871
transformer_engine/.../context_parallel.py:1900
transformer_engine/.../context_parallel.py:1929
```

每一步算出来的不是最终 output，而是一个 partial output。因为 attention 的 softmax 分母跨多个 KV shard，不能把 partial output 直接相加。TE 后面会用 `softmax_lse` 做 correction：

```python
flash_attn_fwd_softmax_lse_correction(...)
flash_attn_fwd_out_correction(...)
tex.thd_out_correction(...)
```

对应源码：

```text
transformer_engine/.../context_parallel.py:1996
transformer_engine/.../context_parallel.py:2036
transformer_engine/.../context_parallel.py:2052
```

所以 P2P 的数学含义是：

```text
完整 attention = softmax(Q K_all^T) V_all

P2P 分步算:
  step0: Q 和 K/V shard0 得到 partial output + lse0
  step1: Q 和 K/V shard1 得到 partial output + lse1
  ...
  最后用 lse 做稳定合并，得到和全量 KV attention 等价的 local output。
```

#### 7.1.1 一个 p2p 例子

假设：

```text
global sequence length S = 16
cp_size = 4
每个 CP rank 持有 4 个 token
```

先用普通 contiguous chunk 理解通信，不考虑 DualChunkSwap：

```text
rank0 Q = global [0, 4)
rank1 Q = global [4, 8)
rank2 Q = global [8, 12)
rank3 Q = global [12, 16)
```

causal attention 下，rank2 的 Q 理论上需要：

```text
K/V global [0, 12)
```

P2P 不会让 rank2 一开始就拿到 `[0, 12)`。它会让 KV 沿 ring 流动。rank2 分多步看到：

```text
step0:
  rank2 本地 K/V = global [8, 12)
  diagonal tile，带 causal mask

step1:
  收到 rank1 的 K/V = global [4, 8)
  lower-triangle tile，全可见

step2:
  收到 rank0 的 K/V = global [0, 4)
  lower-triangle tile，全可见

step3:
  收到 rank3 的 K/V = global [12, 16)
  future side，对 rank2 causal 不可见或只参与特殊 half/correction 逻辑
```

真实 TE 实现还叠加了 zigzag / 双 half 负载均衡，所以源码里会出现：

```text
diagonal
lower-triangle
upper-triangle
```

以及 THD 下的半段读取：

```python
tex.thd_read_half_tensor(...)
```

对应源码：

```text
transformer_engine/.../context_parallel.py:819
```

但直观上可以记成：

```text
p2p = local Q 不动，KV 一块块转过来，逐块 FlashAttention，最后用 lse 合并。
```

#### 7.1.2 p2p 是否支持 packed THD

支持。

P2P 入口没有像 all_gather 那样 assert 掉 `qkv_format="thd"`。相反，它有大量 THD 分支：

```python
if qkv_format == "thd":
    cu_seqlens_q_padded = cu_seqlens_q_padded // cp_size
    cu_seqlens_kv_padded = cu_seqlens_kv_padded // cp_size
```

对应源码：

```text
transformer_engine/.../context_parallel.py:1497
```

P2P 的 per-step prepare 里也专门处理 THD：

```python
elif qkv_format == "thd":
    cu_seqlens_q_per_step = cu_seqlens_q // cp_size
    cu_seqlens_kv_per_step = cu_seqlens_kv // cp_size
```

以及 lower/upper tile 的 half：

```python
cu_seqlens_kv_per_step = cu_seqlens_kv // (cp_size * 2)
cu_seqlens_q_per_step = cu_seqlens_q // (cp_size * 2)
```

对应源码：

```text
transformer_engine/.../context_parallel.py:771
transformer_engine/.../context_parallel.py:800
transformer_engine/.../context_parallel.py:837
```

调用 FlashAttention 时，如果是 THD，会走 varlen 参数：

```python
fa_forward_args_thd = get_fa_args(
    True,
    use_flash_attn_3,
    qkv_format,
    cu_seqlens_q=cu_seqlens_q_,
    cu_seqlens_kv=cu_seqlens_kv_,
    max_seqlen_q=max_seqlen_q_,
    max_seqlen_kv=max_seqlen_kv_,
)
```

对应源码：

```text
transformer_engine/.../context_parallel.py:1070
```

同时总入口要求 THD CP 必须提供 padded cu_seqlens：

```python
assert qkv_format != "thd" or (
    cu_seqlens_q_padded is not None and cu_seqlens_kv_padded is not None
)
```

对应源码：

```text
transformer_engine/.../context_parallel.py:4759
```

这里要注意，TE 的 assert 只能保证字段存在，不能证明输入语义一定正确。TE backend 更前面还有一个兜底逻辑：如果没有显式传 `cu_seqlens_q_padded/cu_seqlens_kv_padded`，它可能会把它们默认补成普通 `cu_seqlens_q/cu_seqlens_kv`：

```python
if (q_format == "thd" or "padding" in attn_mask_type) and cu_seqlens_q_padded is None:
    cu_seqlens_q_padded = cu_seqlens_q
if (kv_format == "thd" or "padding" in attn_mask_type) and cu_seqlens_kv_padded is None:
    cu_seqlens_kv_padded = cu_seqlens_kv
```

对应源码：

```text
transformer_engine/.../dot_product_attention/backends.py:2076
```

所以 `packed THD + CP p2p` 的 padding 不是 TE attention kernel 临时帮你做的。正确分层应该是：

```text
输入/批处理层:
  把每个独立样本 pad 到能按 2 * cp_size 等分；
  生成真实样本边界 cu_seqlens；
  生成包含 padding 的 cu_seqlens_padded；
  按 zigzag CP layout 切出当前 rank 的 token。

TE attention 层:
  消费已经切好的 q/k/v；
  消费 cu_seqlens / cu_seqlens_padded；
  按 per-step tile 调 FlashAttention/FusedAttention。
```

Slime 普通 CP 路径里，样本级 padding 和 zigzag 切分发生在 `slice_with_cp`：

```python
chunk_size = (token_len + 2 * cp_size - 1) // (2 * cp_size)
pad = 2 * cp_size * chunk_size - token_len
tokens = pad_tokens(tokens, pad)

start_1 = chunk_size * cp_rank
start_2 = chunk_size * (2 * cp_size - cp_rank - 1)
return torch.cat([tokens[start_1:end_1], tokens[start_2:end_2]])
```

对应源码：

```text
slime/backends/megatron_utils/cp_utils.py:307
```

也就是说，packed 不等于完全没有 padding。packed 是多个样本压成一个 THD token stream；CP p2p 还要求每个样本在内存布局上能被 `2 * cp_size` 规整切分，不能整除就要在样本尾部 pad。

这些 padding token 一般不需要额外 attention mask，前提是 decoder causal attention 且 padding 只追加在每个样本尾部：

```text
真实 token:
  在 padding token 前面。
  causal attention 看不到未来 padding。

padding token:
  可能能看见前面的真实 token。
  但 padding token 的输出不参与 loss，loss_mask = 0。

下一个 sample:
  被 cu_seqlens 边界隔开，不会看见前一个 sample 的真实 token 或 padding token。
```

所以这里会多算 padding token 的 forward/backward，但不会污染有效 token 的 loss。真正危险的是把 padding 插到样本中间，或者把 `cu_seqlens/cu_seqlens_padded` 传错，让后续真实 token 可以 attend 到 padding token；这种语义错误 TE 的基础 assert 不一定能发现。

所以可以总结：

```text
p2p:
  支持 packed THD。
  需要 cu_seqlens_q/k 和 cu_seqlens_q_padded/k_padded。
  支持 causal/non-causal。
  不支持 sliding window attention。
```

其中 sliding window 的限制来自总入口：

```python
assert not sliding_window_attn or cp_comm_type in ["a2a", "all_gather"]
```

对应源码：

```text
transformer_engine/.../context_parallel.py:4766
```

### 7.2 a2a：sequence 和 head 互换

A2A 路径入口是：

```text
AttnFuncWithCPAndQKVOA2A
```

源码注释说：

```text
Like Ulysses, applying A2A to QKVO.
```

对应源码：

```text
transformer_engine/.../context_parallel.py:3877
```

A2A 的核心不是 KV ring，也不是 KV all-gather buffer，而是把并行维度换一下：

```text
attention 前:
  每个 rank 持有部分 sequence、完整 heads。

A2A 后:
  每个 rank 持有完整 sequence、部分 heads。

attention kernel:
  在完整 sequence 上算自己那部分 heads。

attention 后:
  再 A2A 一次，把输出从“完整 sequence、部分 heads”换回“部分 sequence、完整 heads”。
```

这里有一个容易误解的点：A2A 本身不像 P2P 那样强依赖 zigzag / DualChunkSwap 来做 causal attention 负载均衡。

原因是 A2A 在 attention 前会把每个 rank 变成：

```text
full sequence + partial heads
```

于是每个 rank 都算完整 sequence，只是 heads 不同。causal attention 的序列负载在各 rank 之间基本一致，不存在“某个 rank 只拿前段 token，另一个 rank 只拿后段 token”导致的明显不均衡。

如果只为 A2A 从零设计输入布局，理论上可以用更直接的 contiguous sequence shard：

```text
rank0: chunk0
rank1: chunk1
rank2: chunk2
rank3: chunk3
```

但 TE/Megatron 的 CP 数据侧使用统一的 CP 输入布局：

```text
每个样本切成 2 * cp_size 个 chunk；
rank r 拿 chunk r 和 chunk 2 * cp_size - r - 1。
```

这套布局对 P2P 很自然，也方便同一套 dataloader / packed metadata / CP batch slicing 在不同 `cp_comm_type` 之间复用。所以 A2A 沿用了这个输入布局，然后在 attention 前后做 reorder：

```text
attention 前:
  zigzag CP layout -> logical sequence order

attention 后:
  logical sequence order -> zigzag CP layout
```

也就是说：

```text
P2P:
  直接消费 zigzag CP layout。

A2A:
  为了兼容统一 CP 输入布局，先把 zigzag 顺序整理成 attention kernel 期望的正常序列顺序。
```

源码注释里直接写了 shape：

```text
# a2a: gather s and split h
# [b, s//cp, h, d] -> [b, s, h//cp, d]
# [s//cp, b, h, d] -> [s, b, h//cp, d]
# [t//cp, h, d] -> [t, h//cp, d]
```

对应源码：

```text
transformer_engine/.../context_parallel.py:4040
```

具体通信函数是：

```python
q, k, v = flash_attn_a2a_communicate(
    [q, k, v],
    chunk_ids_for_a2a,
    seq_dim_qkv,
    cp_size,
    cp_group,
    cp_stream,
    before_attn=True,
    qkv_format=qkv_format,
    cu_seqlens_q_padded=cu_seqlens_q_padded,
    cu_seqlens_kv_padded=cu_seqlens_kv_padded,
    a2a_input_names=["q", "k", "v"],
)
```

对应源码：

```text
transformer_engine/.../context_parallel.py:4044
```

#### 7.2.1 A2A 的 overlap 粒度

A2A 也用了异步通信和 `cp_stream`，但它的 overlap 粒度和 P2P 不一样。

A2A 通信函数输入通常是三个 tensor：

```python
flash_attn_a2a_communicate(
    [q, k, v],
    ...,
    a2a_input_names=["q", "k", "v"],
)
```

所以内部：

```text
len(a2a_inputs) = 3
```

源码里循环是：

```python
for i in range(len(a2a_inputs) + 2):
```

也就是：

```text
i = 0, 1, 2, 3, 4
```

它做的是 Q/K/V tensor 级别的小流水，不是 micro-batch 级别：

```text
i = 0:
  准备 q 的 all_to_all 输入布局。

i = 1:
  发起 q 的 all_to_all。
  准备 k 的 all_to_all 输入布局。

i = 2:
  发起 k 的 all_to_all。
  等 q 的 all_to_all 完成。
  在 cp_stream 上重排 q 的输出。
  准备 v 的 all_to_all 输入布局。

i = 3:
  发起 v 的 all_to_all。
  等 k 的 all_to_all 完成。
  在 cp_stream 上重排 k 的输出。

i = 4:
  等 v 的 all_to_all 完成。
  在 cp_stream 上重排 v 的输出。
```

对应源码是：

```python
if 0 < i < len(a2a_inputs) + 1:
    a2a_reqs[i - 1] = torch.distributed.all_to_all_single(
        a2a_outputs[i - 1],
        a2a_inputs[i - 1],
        group=cp_group,
        async_op=True,
    )

if i > 1:
    with torch.cuda.stream(cp_stream):
        a2a_reqs[i - 2].wait()
        x = a2a_outputs[i - 2]
        reorder...
```

对应源码：

```text
transformer_engine/.../context_parallel.py:473
transformer_engine/.../context_parallel.py:477
transformer_engine/.../context_parallel.py:480
transformer_engine/.../context_parallel.py:481
```

函数返回前会同步：

```python
torch.cuda.current_stream().wait_stream(cp_stream)
```

对应源码：

```text
transformer_engine/.../context_parallel.py:586
```

所以 A2A 的 overlap 是：

```text
后一个 tensor 的 all_to_all 通信
  overlap
前一个 tensor 通信完成后的 reorder / reshape
```

它不是：

```text
batch0 attention compute
  overlap
batch1 A2A communication
```

也不是 P2P 那种：

```text
当前 KV shard 的 attention compute
  overlap
下一 KV shard 的 P2P communication
```

原因是 A2A 的 forward 主流程是：

```text
pre-attn A2A 完成
  -> 得到 full sequence + partial heads 的 q/k/v
  -> 调一次 FlashAttention/FusedAttention
  -> post-attn A2A
```

也就是说，attention kernel 开始前，pre-attn A2A 已经完成。A2A 没有像 P2P 那样把一个 attention 拆成多个 KV step，所以也没有 step-level compute/comm overlap。

`flash_attn_a2a_communicate` 内部先把 head 维拆成 `cp_size` 份：

```python
x = x.view(..., cp_size, h // cp_size, ...)
x = x.movedim(head_dim, 0).contiguous()
torch.distributed.all_to_all_single(...)
```

对应源码：

```text
transformer_engine/.../context_parallel.py:513
transformer_engine/.../context_parallel.py:524
transformer_engine/.../context_parallel.py:477
```

attention 后再做反向 A2A：

```text
# a2a: split s and gather h
# [b, s, h//cp, d] -> [b*s//cp, h, d]
# [s, b, h//cp, d] -> [s//cp*b, h, d]
# [t, h//cp, d] -> [t//cp, h, d]
```

对应源码：

```text
transformer_engine/.../context_parallel.py:4184
```

#### 7.2.2 一个 a2a 例子

假设：

```text
global sequence length S = 16
num_heads = 8
cp_size = 4
每个 rank 初始持有 4 个 token、8 个 heads
```

attention 前：

```text
rank0: sequence [0, 4),   heads [0, 8)
rank1: sequence [4, 8),   heads [0, 8)
rank2: sequence [8, 12),  heads [0, 8)
rank3: sequence [12, 16), heads [0, 8)
```

A2A 会把 head 切成 4 份，同时把 sequence gather 起来：

```text
rank0 after A2A:
  sequence [0, 16), heads [0, 2)

rank1 after A2A:
  sequence [0, 16), heads [2, 4)

rank2 after A2A:
  sequence [0, 16), heads [4, 6)

rank3 after A2A:
  sequence [0, 16), heads [6, 8)
```

这时每个 rank 都可以直接调用普通 attention kernel：

```text
Q/K/V 都是完整 sequence，但只有部分 heads。
```

算完 output 后再反向 A2A：

```text
rank0 output:
  sequence [0, 4), heads [0, 8)

rank1 output:
  sequence [4, 8), heads [0, 8)

rank2 output:
  sequence [8, 12), heads [0, 8)

rank3 output:
  sequence [12, 16), heads [0, 8)
```

所以 A2A 的直觉是：

```text
p2p 用时间步换 KV shard；
a2a 用 head 维换 sequence 维。
```

#### 7.2.3 a2a 是否支持 packed THD

支持，但有条件。

A2A 入口没有禁止 THD，反而会在 `qkv_format == "thd"` 时选择 varlen FlashAttention：

```python
if qkv_format == "thd":
    flash_attn_fwd = _flash_attn_varlen_fwd
```

对应源码：

```text
transformer_engine/.../context_parallel.py:3969
```

A2A 通信函数也有 THD 专门逻辑：

```text
# [t, h, d] -> [t, cp, h//cp, d]
# [cp, t, h//cp, d] -> [cp*t, h//cp, d]
reorder_seq_chunks_after_a2a_before_attn_thd(...)
```

对应源码：

```text
transformer_engine/.../context_parallel.py:513
transformer_engine/.../context_parallel.py:501
transformer_engine/.../context_parallel.py:504
```

attention 后也有 THD 反向重排：

```python
reorder_seq_chunks_before_a2a_after_attn_thd(...)
```

对应源码：

```text
transformer_engine/.../context_parallel.py:550
```

限制主要有：

```text
1. THD CP 总入口要求 cu_seqlens_q_padded / cu_seqlens_kv_padded 不为空。

2. q/k/v 的 head 数必须能被 cp_size 整除。

3. seq_len 需要满足偶数要求。

4. bshd/sbhd 下不支持 padding mask；THD 通过 cu_seqlens 表达 packed 边界。

5. sliding window attention 支持 a2a，但需要 FusedAttention 或 FlashAttention >= 2.3。
```

对应源码：

```text
transformer_engine/.../context_parallel.py:4759
transformer_engine/.../context_parallel.py:3952
transformer_engine/.../context_parallel.py:3948
transformer_engine/.../context_parallel.py:3932
transformer_engine/.../context_parallel.py:3940
```

所以可以总结：

```text
a2a:
  支持 packed THD。
  需要 padded cu_seqlens。
  需要 num_heads % cp_size == 0。
  支持 sliding window attention。
  更像 Ulysses：全序列、分 heads。
```

### 7.3 a2a+p2p：层级 CP 混合方案

`a2a+p2p` 不是一个完全独立的新 attention kernel。入口处它仍然走：

```python
AttnFuncWithCPAndKVP2P.apply(...)
```

对应源码：

```text
transformer_engine/.../context_parallel.py:4805
```

区别在于 `cp_group` 不是单个 group，而是两个 group：

```python
cp_group = [a2a_cp_group, p2p_cp_group]
```

入口会检查：

```python
assert isinstance(cp_group, list) and len(cp_group) == 2
```

对应源码：

```text
transformer_engine/.../context_parallel.py:4726
```

进入 `AttnFuncWithCPAndKVP2P.forward` 后，会把 group 拆开：

```python
if isinstance(cp_group, list):
    cp_group_a2a = cp_group[0]
    cp_size_a2a = get_distributed_world_size(cp_group_a2a)
    rank_a2a = get_distributed_rank(cp_group_a2a)
    cp_group = cp_group[1]
```

对应源码：

```text
transformer_engine/.../context_parallel.py:1458
```

可以理解成：

```text
总 CP group 被拆成二维：
  一个维度做 a2a，把 heads 切开、sequence gather 到子组内；
  另一个维度做 p2p ring，在更高层 group 之间传 KV。
```

#### 7.3.1 一个 a2a+p2p 例子

假设：

```text
context_parallel_size = 8
a2a_cp_size = 2
p2p_cp_size = 4
num_heads = 16
global sequence length = 32
```

可以把 8 个 CP rank 看成一个 2 x 4 网格：

```text
           p2p dim
          0   1   2   3
a2a 0:   r0  r1  r2  r3
a2a 1:   r4  r5  r6  r7
```

在每个 a2a 子组里，先做类似 Ulysses 的 head/sequence 交换：

```text
a2a_size = 2
每个 rank 只保留 16 / 2 = 8 个 heads
但在这个 a2a 子组内获得更长的 sequence 视野
```

然后在 p2p 维度上，KV 继续沿 ring 流动：

```text
p2p_size = 4
每个 rank 分 4 个 step 接收其他 p2p rank 的 KV shard
边通信边算 partial attention
最后用 softmax_lse correction 合并
```

直觉上：

```text
a2a:
  降低每个 rank 要算的 head 数。

p2p:
  避免在整个 CP=8 上一次性 all-gather 全 KV。

a2a+p2p:
  用二维分解降低单一通信方式的压力。
```

#### 7.3.2 a2a+p2p 是否支持 packed THD

当前不支持。

入口处直接 assert：

```python
assert qkv_format != "thd"
```

对应源码：

```text
transformer_engine/.../context_parallel.py:4730
```

同时也要求：

```text
attn_bias_type == "no_bias"
```

对应源码：

```text
transformer_engine/.../context_parallel.py:4733
```

另外，如果其中一个子 group size 是 1，会自动退化：

```python
if get_distributed_world_size(cp_group[0]) == 1:
    cp_group = cp_group[1]
    cp_comm_type = "p2p"
elif get_distributed_world_size(cp_group[1]) == 1:
    cp_group = cp_group[0]
    cp_comm_type = "a2a"
```

对应源码：

```text
transformer_engine/.../context_parallel.py:4736
```

所以可以总结：

```text
a2a+p2p:
  当前不支持 packed THD。
  适合 fixed-shape dense attention 的层级 CP。
  一个维度做 a2a，一个维度做 p2p。
  当 a2a 或 p2p 子组大小为 1 时，会退化成单独的 p2p 或 a2a。
```

### 7.4 四种 CP 通信方式对比

| cp_comm_type | 核心思路 | packed THD 支持 | 主要限制 | 直觉 |
| --- | --- | --- | --- | --- |
| `p2p` | KV 沿 ring 逐步传递，local Q 分步计算 partial attention | 支持 | THD 需要 padded cu_seqlens；不支持 sliding window | 省 KV 峰值通信/显存，靠多 step + lse correction 合并 |
| `all_gather` | 先 gather 完整 KV，再给每个 Q chunk slice 需要的 KV | 本地 TE 不支持 | 不支持 THD，不支持 padding mask | 实现直观，但 KV buffer 更大 |
| `a2a` | all-to-all：gather sequence、split heads，attention 后反向 A2A | 支持 | heads 必须能被 CP size 整除；THD 需要 padded cu_seqlens | 用 head 维换 sequence 维，类似 Ulysses |
| `a2a+p2p` | 层级 CP：一个维度 A2A，另一个维度 P2P | 当前不支持 | 需要 `[a2a_cp_group, p2p_cp_group]`；不支持 THD | 二维 CP，降低单一通信方式压力 |

对 Slime 当前学习最有用的记法是：

```text
普通 dense attention + packed THD + CP:
  优先看 TE p2p。

需要 Ulysses 风格 head/sequence 交换:
  看 TE a2a。

GLM5.2 的 --allgather-cp:
  不是 TE all_gather，也不是 TE p2p/a2a。
  它是 Slime/GLM 自定义的 local Q + gathered KV/index_k + starts/ends + SparseMLA。
```
