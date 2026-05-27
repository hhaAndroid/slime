# Slime MTP（Multi-Token Prediction）实现梳理

> 对应启动脚本：`scripts/run-mimo-7B-rl-eagle-hha.sh`，模型：MiMo-7B-RL

---

## 1. 什么是 MTP

MTP（Multi-Token Prediction）是在基础 transformer 之后附加的特殊层，训练模型"预测未来多个 token"的能力。在 MiMo-7B-RL 中，模型自带 1 层 MTP head（DeepSeek 系的 MTP 结构）。

**关键参数：**

```bash
--mtp-num-layers 1             # MTP 层数
--enable-mtp-training          # 开启 MTP 参数梯度更新
--mtp-loss-scaling-factor 0.2  # MTP loss 权重
```

**参数定义位置：** `slime/utils/arguments.py` ~1322 行

```python
def add_mtp_training_arguments(parser):
    reset_arg(parser, "--mtp-num-layers", type=int, default=None)
    reset_arg(parser, "--mtp-loss-scaling-factor", type=float, default=0.2)
    parser.add_argument("--enable-mtp-training", action="store_true", default=False)

# 校验
if args.enable_mtp_training:
    assert args.mtp_num_layers, "必须同时设置 --mtp-num-layers"
```

---

## 2. 模型构建：注入 MTP block spec

**文件：** `slime/backends/megatron_utils/model_provider.py` ~208 行

```python
if args.mtp_num_layers:
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
    mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, ...)
    kwargs["mtp_block_spec"] = mtp_block_spec  # 注入 GPTModel
```

Megatron core 会在主 transformer 后面拼接 MTP 层。

---

## 3. sglang Rollout：MTP head 作为 EAGLE draft 模型

**核心思路：** sglang 把模型自带的 MTP head 当作 EAGLE speculative decoding 的 draft 模型，在 rollout 时加速生成。

**配置（启动脚本）：**

```bash
--sglang-speculative-algorithm EAGLE
--sglang-speculative-num-steps 3
--sglang-speculative-eagle-topk 1
--sglang-speculative-num-draft-tokens 4
```

**关键代码：** `slime/backends/sglang_utils/sglang_engine.py`

- 传入 `enable_draft_weights_cpu_backup: True`，让 MTP 权重在 CPU 上有备份，sglang 可独立管理 draft 权重
- rollout 结束后收集 speculative 指标：
  - `spec_accept_token_num`：被接受的 draft token 数
  - `spec_draft_token_num`：总共 draft 的 token 数
  - `spec_accept_rate`：接受率

---

## 4. batch["tokens"] 是什么

**来源：** `slime/rollout/sglang_rollout.py`

```python
sample.tokens = prompt_ids                           # 初始化为 prompt token ids
sample.tokens = sample.tokens + new_response_tokens  # rollout 后拼上 response
```

**结论：**
- `batch["tokens"]` = **prompt token ids + response token ids** 拼接的完整序列
- **没有 -100**，全是真实 token id

---

## 5. Megatron 训练：前向传播与 loss 计算

### 5.1 前向传播入口

**文件：** `slime/backends/megatron_utils/model.py` ~396 行

```python
forward_kwargs = {
    "input_ids": batch["tokens"],       # 完整 token 序列
    "labels": None,                     # 主模型不在此算 loss
    "loss_mask": batch["full_loss_masks"],
    ...
}

if args.enable_mtp_training:
    forward_kwargs["mtp_kwargs"] = {"mtp_labels": batch["tokens"]}  # 和 input_ids 一样
```

> 注意：`labels=None` 说明主模型的 policy loss 不走 Megatron 内部的 cross-entropy，而是由 slime 在外部单独计算 GRPO loss。

### 5.2 loss_mask 的构造

**文件：** `slime/backends/megatron_utils/data.py` ~141 行

```python
loss_mask = F.pad(loss_mask, (prompt_length - 1, 1), value=0)
# response 位置 = 1，prompt 位置 = 0
batch["full_loss_masks"] = loss_masks
```

### 5.3 MTP loss 计算（Megatron core）

**文件：** `Megatron-LM/megatron/core/models/gpt/gpt_model.py` ~570 行

```python
mtp_labels = mtp_kwargs['mtp_labels'].clone()

# 第 1 次 shift：位置 t 对应 t+1 的 token（主模型 LM 对齐）
mtp_labels, _ = roll_tensor(mtp_labels, shifts=-1, ...)
loss_mask, _  = roll_tensor(loss_mask,  shifts=-1, ...)

for mtp_layer_number in range(mtp_num_layers):  # 通常 1 层
    # 第 2 次 shift：位置 t 对应 t+2 的 token（MTP 目标）
    mtp_labels, _ = roll_tensor(mtp_labels, shifts=-1, ...)
    loss_mask, _  = roll_tensor(loss_mask,  shifts=-1, ...)

    mtp_loss = compute_output_layer_and_language_model_loss(
        hidden_states_list[mtp_layer_number + 1],
        labels=mtp_labels,
    )
    mtp_loss = loss_mask * mtp_loss   # element-wise mask，清零 prompt 区域
```

**shift 的直觉：**

```
原始:       [p1, p2, ..., pN,  r1,  r2,  r3, ..., rM]
mask:       [ 0,  0, ...,  0,   1,   1,   1, ...,  1]

shift ×1:   [p2, ..., pN,  r1,  r2, ..., rM,  ?]
mask:       [ 0, ...,  0,   1,   1, ...,  1,  0]

shift ×2:   [..., pN,  r1,  r2, ..., rM,  ?,  ?]
mask:       [...,  0,   1,   1, ...,  1,  0,  0]
```

经过两次 shift 后，`loss_mask * mtp_loss` 保证只在 response 区域计算 MTP cross-entropy loss。

### 5.4 MTP loss 汇总

**文件：** `slime/backends/megatron_utils/model.py` ~609 行

```python
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

tracker = MTPLossLoggingHelper.tracker
torch.distributed.all_reduce(tracker["values"], ...)
mtp_losses = (tracker["values"] * mtp_loss_scale).item()
MTPLossLoggingHelper.clean_loss_in_tracker()
```

**最终 loss：**

```
total_loss = GRPO_policy_loss + mtp_loss_scaling_factor × mtp_loss
           = GRPO_policy_loss + 0.2 × mtp_loss
```

---

## 6. MTP 学习目标总结

| 组件 | 位置 t 的预测目标 | Loss 类型 |
|------|-----------------|-----------|
| 主模型 | `t+1` token | GRPO loss（RL reward）|
| MTP head | `t+2` token | Teacher-forcing cross-entropy |

- **MTP head 学的是：** 给定当前位置的 hidden state，预测 2 步之后的真实 token
- **只在 response 区域计算 loss**，prompt 区域通过 loss_mask 清零
- **不使用 -100**，用 loss_mask（0/1 float tensor）来屏蔽 prompt 区域

---

## 7. 整体数据流

```
[Rollout 阶段]
  sglang 加载完整模型（含 MTP head）
  MTP head 作为 EAGLE draft → 投机解码加速
  生成轨迹：tokens = [prompt_ids + response_ids]
  收集 spec_accept_rate 等指标

        ↓

[Training 阶段]
  data.py：构造 full_loss_masks（response=1, prompt=0）
  model.py：
    forward_kwargs["input_ids"]  = tokens          # 完整序列
    forward_kwargs["loss_mask"]  = full_loss_masks  # response mask
    forward_kwargs["mtp_kwargs"] = {"mtp_labels": tokens}

  gpt_model.py（Megatron core）：
    主模型：不算内部 loss（labels=None），hidden states 返回给 slime
    MTP 分支：tokens shift ×2 → 算 t+2 预测的 CE loss → mask → 存入 tracker

  model.py：
    GRPO loss（主模型）+ 0.2 × MTP loss（tracker 读取）= total loss
    反向传播，同时更新主模型参数和 MTP head 参数

        ↓

  参数更新后同步回 sglang，下一轮 rollout
```

---

## 8. 关键文件索引

| 文件 | 内容 |
|------|------|
| `slime/utils/arguments.py` ~1322 | MTP 相关参数定义与校验 |
| `slime/backends/megatron_utils/model_provider.py` ~208 | 注入 MTP block spec 到 GPTModel |
| `slime/backends/sglang_utils/sglang_engine.py` | sglang EAGLE speculative decoding 配置 |
| `slime/rollout/sglang_rollout.py` | tokens = prompt_ids + response_ids 的构造 |
| `slime/backends/megatron_utils/data.py` ~131 | full_loss_masks 构造（response=1, prompt=0） |
| `slime/backends/megatron_utils/model.py` ~396 | forward_kwargs 组装，mtp_kwargs 注入 |
| `slime/backends/megatron_utils/model.py` ~609 | MTP loss 从 tracker 读取并汇总 |
| `Megatron-LM/megatron/core/models/gpt/gpt_model.py` ~570 | MTP loss 实际计算（shift + mask + CE） |
