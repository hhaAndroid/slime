# Slime 对 Megatron-LM 的 patch 审查

> 本文记录 Slime 为什么需要 patch Megatron、当前 patch 修改了哪些模块，以及后续升级新版 Megatron 时应如何拆分和迁移。
>
> 审查日期：2026-08-04。

## 1. 范围和结论

本文主要审查：

```text
docker/patch/latest/megatron.patch
```

它由当前 CUDA Dockerfile 应用到固定的 Megatron commit：

```text
MEGATRON_COMMIT=1dcf0dafa884ad52ffb243625717a3471643e087
```

应用入口：

```text
docker/Dockerfile
build_conda.sh
```

Docker/build 脚本使用 `git apply --3way` 修改 Megatron 源码。因此
`docker/patch/latest` 的 `latest` 表示“当前 Slime 默认镜像组合使用的 patch”，
不表示它可以无条件应用到任意最新 Megatron release。

当前 patch 的统计是：

```text
修改 21 个 Megatron 文件
增加 362 行
删除 75 行
```

最重要的结论：

```text
这不是一份只为 Routing Replay/R3 存在的 patch。

它混合了：
  Routing Replay
  训推共置显存释放
  checkpoint 宽松加载
  distributed optimizer 恢复
  SGLang 权重同步元数据
  INT4 QAT
  MTP RL 训练
  GLM/Qwen 模型结构
  多模态 packed sequence + CP
  MLA RoPE Triton kernel 修复
  Pipeline P2P 和 tokenizer 兼容
```

其中 R3 对 Megatron 源码的直接修改实际上只有两个文件、6 行新增代码。

本文不逐项审查以下变体：

```text
docker/patch/v*/megatron.patch
docker/amd_patch/*/megatron.patch
docker/npu_patch/megatron.patch
```

这些 patch 面向不同历史版本或硬件后端，不能和 CUDA `latest` patch 机械合并分析。

## 2. patch 是如何应用的

当前 Docker 构建流程先 clone 并 checkout 固定 Megatron commit：

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
git checkout ${MEGATRON_COMMIT}
pip install -e .
```

随后把 Slime patch 复制到 Megatron worktree 并应用：

```bash
git apply megatron.patch --3way
```

`build_conda.sh` 也执行同类流程，并在 patch 已应用或不适用时跳过。

因此后续升级 Megatron 时不能只修改 `MEGATRON_COMMIT`：

```text
升级 Megatron commit
  -> 重新检查 patch 是否已经 upstream
  -> 重新检查函数签名和调用顺序
  -> 删除已不需要的 hunk
  -> 迁移仍然属于 Slime 的集成逻辑
  -> 重新生成与新 commit 精确匹配的 patch
```

## 3. R3 实际只 patch 了两个位置

### 3.1 在每个 TopKRouter 上注册 Slime RoutingReplay

修改文件：

```text
megatron/core/transformer/moe/router.py
```

新增：

```python
from slime.utils.routing_replay import register_routing_replay
register_routing_replay(self)
```

作用：

```text
每个本地 MoE TopKRouter
  -> 创建 module.routing_replay
  -> 注册 forward pre-hook
  -> 追加到 RoutingReplay.all_routing_replays
```

### 3.2 包装 Megatron 的 compute_topk

修改文件：

```text
megatron/core/transformer/moe/moe_utils.py
```

新增：

```python
from slime.utils.routing_replay import get_routing_replay_compute_topk
compute_topk = get_routing_replay_compute_topk(compute_topk)
```

作用：

```text
Megatron 原始 compute_topk
  -> 外包一层 Slime wrapper
  -> 根据环境变量选择：
       fallthrough
       record
       replay_forward
       replay_backward
```

### 3.3 为什么说它重复了 Megatron 原生机制

当前 patch 的上游上下文已经包含：

```python
self.router_replay = None
if self.config.moe_enable_routing_replay:
    self.router_replay = RouterReplay()
```

也就是说，被 Slime patch 的这个 Megatron commit 本身已经具有原生
`RouterReplay`。最终同一个 `TopKRouter` 可以同时存在：

```text
self.router_replay
  Megatron 原生 RouterReplay

self.routing_replay
  Slime RoutingReplay
```

Slime 自己实现的状态是：

```text
每层：
  top_indices_list = [MB0, MB1, ...]
  forward_index
  backward_index

全局：
  all_routing_replays
  当前层 ROUTING_REPLAY 指针
  ROUTING_REPLAY_STAGE 环境变量
```

Megatron 原生对象使用：

```text
RouterReplayAction.RECORD
RouterReplayAction.REPLAY_FORWARD
RouterReplayAction.REPLAY_BACKWARD
target_topk_idx
recorded_topk_idx
replay_backward_list
```

从代码可以确定，两者接口和 microbatch 状态组织方式不同。Slime 选择自己的实现，
能够直接接入：

```text
SGLang rollout_routed_experts
  -> packed token / CP / SP 对齐
  -> PP/VPP layer 遍历
  -> 一次预填多个 microbatch
  -> actor logprob/training/recompute 阶段切换
```

至于最初为何没有直接扩展 Megatron 原生 API，patch 内没有明确设计说明。根据实现推断，
主要原因是 Slime 当时采用了“少改 Megatron、在 Router 构造点和 top-k 点插钩子”的
快速集成方式。

## 4. 安装期 Megatron patch 全量分组

### 4.1 Checkpoint 加载和兼容

涉及文件：

```text
megatron/core/dist_checkpointing/strategies/common.py
megatron/core/dist_checkpointing/strategies/torch.py
megatron/core/optimizer/distrib_optimizer.py
```

主要修改：

1. `torch.load(..., weights_only=False)`

   用于加载包含非 tensor Python 状态的 common checkpoint，兼容 PyTorch
   `weights_only` 默认行为变化。

2. 遇到 model 中存在、checkpoint 中不存在的 key 时跳过

   原实现抛 `KeyError`，patch 改为打印并继续，同时启用
   `allow_partial_load=True`。

3. shape mismatch 列表只处理 checkpoint metadata 中实际存在的 key

4. DistributedOptimizer 不再把 `step` 当成 gradient-buffer bucket state

   `step` 由 optimizer param groups 单独恢复，避免保存端和加载端 optimizer placement
   不一致导致 state skeleton 不匹配。

用途判断：

```text
支持模型结构或训练角色存在差异时的部分加载；
支持 MTP/actor/critic 等配置变化；
修复 distributed optimizer checkpoint 恢复兼容。
```

风险：

```text
宽松跳过 missing key 可能把真正的模型配置错误隐藏成“加载成功”。
迁移新版时应优先替换为显式 allowlist 或调用方配置，而不是全局跳过。
```

### 4.2 训推共置与 torch_memory_saver

涉及文件：

```text
megatron/core/distributed/distributed_data_parallel.py
megatron/core/distributed/param_and_grad_buffer.py
megatron/training/training.py
megatron/core/inference/contexts/dynamic_context.py
```

新增参数：

```text
disable_grad_buffers_cpu_backup
disable_param_buffers_cpu_backup
```

这些参数从 Slime args 传入 Megatron DDP，再传到 `_ParamAndGradBuffer`。创建
param/grad buffer 时，patch 可以进入：

```python
torch_memory_saver.region(
    tag=...,
    enable_cpu_backup=False,
)
```

作用：

```text
训推共置释放训练 GPU 显存时，
避免把可以重建或另有来源的大型 param/grad buffer 全部备份到 CPU，
降低 host memory 和 offload/resume 成本。
```

patch 还在 Megatron dynamic inference context 中关闭其自己的
`HAVE_TORCH_MEMORY_SAVER` 路径，避免 Megatron 和 Slime 同时管理 TMS hook。

限制：

```text
该实现显式禁止与 nccl_ub=True 一起使用。
```

这是明显属于 Slime 训推共置生命周期的集成，升级 Megatron 后也可能仍需保留，
但最好迁移到 Megatron 提供的 allocator/offload 扩展点，而不是长期维护源码 patch。

### 4.3 权重同步所需的 TP 元数据

涉及文件：

```text
megatron/core/extensions/transformer_engine.py
```

`TELinear` 初始化时给参数记录：

```python
setattr(param, "parallel_mode", parallel_mode)
```

Slime 权重同步代码读取它来区分：

```text
column parallel
row parallel
duplicated
```

这决定 Megatron 参数如何转换、切片或 gather 成 SGLang/Hugging Face 所需布局。

### 4.4 INT4 fake-QAT

涉及文件：

```text
megatron/core/extensions/transformer_engine.py
```

patch 增加 `_FakeInt4QuantizationSTE`，并在环境变量打开时包装
`TEGroupedLinear._get_weight_tensors()`：

```text
OPEN_TRAINING_INT4_FAKE_QAT_FLAG=1
OPEN_TRAINING_INT4_GROUP_SIZE=<group_size>
```

作用：

```text
expert weight
  -> group-wise fake INT4 quantize/dequantize
  -> forward 使用量化误差后的 weight
  -> backward 用 STE 直通梯度
```

这是 Slime 低精度 RL 训练功能，不属于普通 R3。

### 4.5 MLA YaRN RoPE Triton kernel 修复

涉及文件：

```text
megatron/core/fusions/fused_mla_yarn_rope_apply.py
```

主要修改：

```text
k_dim 使用 next_power_of_2 后的 Triton block 范围
修正跨 head block 的 offset
分别处理 K/V mask
支持 v_dim == 0
同步修改 forward/backward kernel
```

这是特定 MLA/DSA/模型组合的 kernel 正确性和 shape 兼容修复。升级 Megatron 或
Triton 后应先检查是否已经 upstream，不能直接把旧 kernel hunk 搬过去。

### 4.6 多模态 mRoPE + packed sequence + CP

涉及文件：

```text
megatron/core/models/common/embeddings/rotary_pos_embedding.py
megatron/core/models/gpt/gpt_model.py
```

patch 给 `MultimodalRotaryEmbedding.forward()` 增加 `packed_seq` 参数。

THD packed sequence 下跳过这里的整段 CP slicing，因为后续
`_apply_rotary_pos_emb_thd` 会按 sequence 处理。如果两处都 slice，会导致位置编码
和 packed token layout 错位。

### 4.7 MTP 在 RL 训练中的适配

涉及文件：

```text
megatron/core/models/gpt/gpt_model.py
megatron/core/transformer/multi_token_prediction.py
```

主要修改：

```text
GPTModel.forward 接受 mtp_kwargs
Slime 显式传 mtp_labels
labels 和 loss_mask 按 next-token 语义 roll
MTP output weight detach
MTP embedding/hidden-state graph detach 调整
position_ids=None 时不 roll
activation checkpoint 只接收 Tensor 参数，非 Tensor 参数由 closure 捕获
```

目的不是让 GPTModel 内部计算主 PPO loss，而是允许 MTP 辅助目标在 Slime
`labels=None` 的主模型调用方式下仍然工作。

### 4.8 GLM sandwich/post layernorm

涉及文件：

```text
megatron/core/models/gpt/gpt_layer_specs.py
megatron/core/transformer/transformer_config.py
megatron/core/transformer/transformer_layer.py
megatron/training/arguments.py
```

增加两个可选模块：

```text
post_self_attn_layernorm
post_mlp_layernorm
```

并把它们接入：

```text
TransformerConfig
  -> GPT layer spec
  -> TransformerLayerSubmodules
  -> attention output
  -> MLP output
```

用于 GLM 等需要 attention/MLP 输出后再做 layernorm 的结构。

### 4.9 Qwen gated attention 参数

涉及文件：

```text
megatron/training/arguments.py
```

新增：

```bash
--use-gated-attention
```

供 Qwen3-Next/Qwen3.5 model/bridge 配置使用。

### 4.10 Pipeline P2P 行为

涉及文件：

```text
megatron/core/pipeline_parallel/p2p_communication.py
```

patch 构造 `torch.distributed.P2POp` 时移除了显式 `group`：

```python
P2POp(isend/irecv, tensor, peer_rank)
```

patch 本身没有留下原因说明。它看起来是针对某个 PyTorch/Megatron 组合的
process-group/peer-rank 兼容处理，但迁移时必须通过 PP 测试重新确认，不能把该推断
当作稳定接口要求。

### 4.11 Tokenizer 和零散兼容

涉及文件：

```text
megatron/training/tokenizer/tokenizer.py
megatron/core/parallel_state.py
```

修改：

```text
AutoTokenizer.from_pretrained(..., trust_remote_code=True)
parallel_state.py 补 import torch.distributed as dist
```

强制 `trust_remote_code=True` 能加载自定义 tokenizer，但扩大了远程代码执行范围，
更合适的做法是由用户配置显式控制。

## 5. 21 个文件的快速索引

| Megatron 文件 | 功能分类 |
|---|---|
| `core/dist_checkpointing/strategies/common.py` | checkpoint/PyTorch 兼容 |
| `core/dist_checkpointing/strategies/torch.py` | partial checkpoint load |
| `core/distributed/distributed_data_parallel.py` | TMS buffer 参数传递 |
| `core/distributed/param_and_grad_buffer.py` | TMS param/grad buffer allocation |
| `core/extensions/transformer_engine.py` | TP 元数据、INT4 fake-QAT |
| `core/fusions/fused_mla_yarn_rope_apply.py` | MLA RoPE kernel |
| `core/inference/contexts/dynamic_context.py` | 禁用 Megatron 自己的 TMS hook |
| `core/models/common/embeddings/rotary_pos_embedding.py` | mRoPE packed CP |
| `core/models/gpt/gpt_layer_specs.py` | post layernorm spec |
| `core/models/gpt/gpt_model.py` | mRoPE、MTP RL |
| `core/optimizer/distrib_optimizer.py` | optimizer checkpoint state |
| `core/parallel_state.py` | 缺失 import |
| `core/pipeline_parallel/p2p_communication.py` | P2P group 行为 |
| `core/transformer/moe/moe_utils.py` | Slime replay top-k wrapper |
| `core/transformer/moe/router.py` | Slime replay 注册 |
| `core/transformer/multi_token_prediction.py` | MTP RL/checkpoint |
| `core/transformer/transformer_config.py` | post layernorm config |
| `core/transformer/transformer_layer.py` | post layernorm 执行 |
| `training/arguments.py` | post norm/gated attention args |
| `training/tokenizer/tokenizer.py` | remote tokenizer code |
| `training/training.py` | TMS DDP 参数传递 |

## 6. 不在 megatron.patch 中的运行时 monkey patch

Slime 还会在 Python 运行时替换部分 Megatron 或相关组件行为。这些修改不会出现在
`git apply --numstat docker/patch/latest/megatron.patch` 中。

### 6.1 TP gradient 分块 coalesce/all-reduce

文件：

```text
slime/backends/megatron_utils/megatron_patch/
  megatron_chunked_grad_coalesce_patch.py
```

它替换 Megatron：

```text
_allreduce_non_tensor_model_parallel_grads
_allreduce_layernorm_grads
```

原实现可能一次 flatten 全部 TP-side gradients，需要一个很大的连续显存块。
Slime 改为按默认 1 GiB 分块：

```text
grad list
  -> split chunks
  -> 每块 flatten
  -> all_reduce
  -> unflatten/copy back
```

这在数学上仍是逐元素 SUM/AVG，主要用于降低 allocator fragmentation 下的 OOM。

### 6.2 ShardedTensor metadata 校验旁路

文件：

```text
slime/backends/megatron_utils/checkpoint.py
```

它替换部分 PyTorch ShardedTensor 初始化/校验逻辑，跳过大模型大量 shard metadata
的非重叠校验，以降低 checkpoint 加载时间。

这主要 patch PyTorch 行为，但由 Slime Megatron backend 导入并生效。

### 6.3 StatelessAdam

文件：

```text
slime/backends/megatron_utils/model.py
```

仅在 `--use-stateless-adam` 时临时把 Megatron optimizer 模块中的 `Adam/CPUAdam`
替换为 Slime `StatelessAdam`，构造完成后恢复原类。

### 6.4 Megatron Bridge/Qwen3-VL 兼容

文件：

```text
slime/backends/megatron_utils/__init__.py
slime/utils/megatron_bridge_utils.py
```

包括：

```text
Qwen3-VL rotary embedding forward 接受但忽略 packed_seq_params
临时给 model config 补 share_embeddings_and_output_weights
修正 HF rope config 字段兼容
```

### 6.5 ROCm async checkpoint writer

`slime/backends/megatron_utils/model.py` 还会在 ROCm 条件下替换异步 filesystem
writer，用于 HIP 兼容。这不属于 CUDA `latest/megatron.patch` 的主体。

## 7. 升级最新 Megatron 时如何处理

不要把当前 1007 行 patch 整体搬到新版。建议分三类处理。

### 7.1 优先迁移并删除源码 patch

候选：

```text
Routing Replay 两个 hook
MLA kernel 修复
MTP 修复
mRoPE packed CP
模型结构支持
checkpoint bugfix
```

这些功能在新版 Megatron 中可能已经部分或全部 upstream。应以目标 release 源码和
测试为准，已有原生实现的直接删除旧 hunk。

R3 的目标结构应是：

```text
Megatron 原生 RouterReplay：
  每层 replay 对象
  top-k record/replay
  forward/backward recompute action

Slime：
  rollout_routed_experts 接收
  packed token/CP/SP 对齐
  PP/VPP/microbatch 数据编排
  ref/teacher/actor/train 阶段控制
```

这样可以删除对 `router.py` 和 `moe_utils.py` 的 Slime import。

### 7.2 可能仍属于 Slime 的集成

```text
torch_memory_saver param/grad buffer 管理
SGLang 权重同步所需 parallel_mode
INT4 RL QAT
TP gradient 分块同步
rollout routing 数据变换和训练阶段编排
```

这些需求由 Slime 的训推共置或 SGLang 集成产生。即使新版 Megatron 没有等价功能，
也应优先寻找公开 hook/config/API，再决定是否维护小而独立的 patch。

### 7.3 高风险兼容覆盖

```text
全局跳过 checkpoint missing keys
强制 trust_remote_code=True
移除 Pipeline P2P group
关闭 Megatron dynamic-context TMS
```

这类改动会改变默认安全性或分布式语义。迁移时必须重新证明必要性，不能因为旧 patch
中存在就默认保留。

## 8. 推荐的后续阅读顺序

如果后续逐项清理 patch，建议按下面顺序：

1. Routing Replay 两处 hunk

   改动最小，且新版 Megatron 已有原生能力，最适合先做无源码 patch 迁移。

2. TMS/offload 四个文件

   直接决定训推共置时的 GPU/CPU 内存行为，风险高但边界相对集中。

3. MTP 与 packed CP

   需要结合 Qwen3.5/MTP/CP 训练测试验证数值和 shape。

4. Checkpoint 与 distributed optimizer

   需要覆盖保存、恢复、partial load、角色切换和 optimizer state。

5. GLM/Qwen/MLA/INT4 等模型特定 patch

   按实际要训练的模型和硬件选择，不要默认全部迁移。

6. P2P、tokenizer 和运行时 monkey patch

   单独做兼容性和安全性审计。

## 9. 复现本次审查的命令

查看 patch 修改文件和行数：

```bash
git apply --numstat docker/patch/latest/megatron.patch
```

查看所有 diff 文件和 hunk：

```bash
rg -n '^diff --git|^@@' docker/patch/latest/megatron.patch
```

查找直接从 Megatron import Slime 的位置：

```bash
rg -n 'from slime' docker/patch/latest/megatron.patch
```

查找运行时 patch：

```bash
rg -n 'patched|monkey|setattr|=' \
  slime/backends/megatron_utils/megatron_patch \
  slime/backends/megatron_utils
```

查看 patch 的历史来源：

```bash
git log --oneline -- docker/patch/latest/megatron.patch
git blame docker/patch/latest/megatron.patch
```

## 10. 外部参考

- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Megatron Core releases](https://github.com/NVIDIA/Megatron-LM/releases)
- [Megatron Core Router Replay 设计文档](https://docs.nvidia.com/megatron-core/developer-guide/0.17.1/api-guide/router_replay.html)

## 11. 当前阶段的最终判断

```text
Slime patch Megatron 的根本原因不是单一的“Megatron 不支持 RL”。

它是为了把一个通用训练内核接入 Slime 特有的：
  训推共置生命周期
  SGLang 权重和 rollout 数据
  多种新模型结构
  低精度训练
  大模型 checkpoint/offload 限制

当前 patch 因历史积累已经把多种独立需求放进同一个文件。

后续升级新版 Megatron 的正确方向是：
  逐项确认 upstream
  删除重复功能
  把 Slime 特有逻辑留在 Slime
  只为没有公开扩展点的少数位置保留最小 patch
```
