# Megatron 训练侧学习记录：以 Qwen3.5 MoE + R3 为例

本文单独记录 slime 中 Megatron 训练侧的主路径。视角是：

- 先从 slime 的 orchestration / rollout / train actor 说起。
- 再深入 Megatron 内部的模型构造、并行组、MoE router、token dispatch、forward/backward。
- 示例采用 `scripts/run-qwen3.5-35B-rl-eagle-hha.sh`，也就是 Qwen3.5-35B-A3B MoE，并开启 `--use-rollout-routing-replay`。

关键入口：

- 启动脚本：`scripts/run-qwen3.5-35B-rl-eagle-hha.sh`
- 模型参数：`scripts/models/qwen3.5-35B-A3B.sh`
- 总训练循环：`train.py`
- Ray 资源和 actor group：`slime/ray/placement_group.py`、`slime/ray/actor_group.py`
- rollout 数据打包：`slime/ray/rollout.py`
- SGLang 请求：`slime/rollout/sglang_rollout.py`、`slime/rollout/sglang_streaming_rollout.py`
- Megatron actor：`slime/backends/megatron_utils/actor.py`
- Megatron 初始化：`slime/backends/megatron_utils/initialize.py`
- Megatron 模型 / optimizer / train step：`slime/backends/megatron_utils/model.py`
- Megatron batch packing：`slime/backends/megatron_utils/data.py`
- CP / SP 对齐工具：`slime/backends/megatron_utils/cp_utils.py`
- routing replay 队列：`slime/utils/routing_replay.py`
- Qwen3.5 spec：`slime_plugins/models/qwen3_5.py`
- Megatron patch：`docker/patch/latest/megatron.patch`

## 1. 例子配置先建立坐标

`scripts/run-qwen3.5-35B-rl-eagle-hha.sh` 最终通过 Ray job 运行：

```bash
python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --colocate \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${GRPO_ARGS[@]} \
  ${PERF_ARGS[@]} \
  ${SGLANG_ARGS[@]} \
  ${MISC_ARGS[@]}
```

这个例子训练侧是 GRPO：

- `--advantage-estimator grpo`
- `--use-kl-loss`
- `--kl-loss-coef 0.00`
- `--entropy-coef 0.00`
- `--eps-clip 0.2`
- `--eps-clip-high 0.28`
- `--use-rollout-routing-replay`

模型侧来自 `scripts/models/qwen3.5-35B-A3B.sh`：

- `--spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"`
- 40 层。
- hidden size 2048。
- attention heads 16，query groups 2，kv channels 256。
- vocab size 248320。
- MoE 全层开启：`MOE_LAYER_FREQ` 是长度 40 的 `[1, 1, ..., 1]`。
- 256 个 routed experts。
- 每 token top-8：`--moe-router-topk 8`。
- shared expert intermediate size 512。
- router score function 是 softmax。
- Megatron MoE dispatcher 是 `alltoall`。
- router 计算用 fp32：`--moe-router-dtype fp32`。

并行参数：

- dense / attention 侧：`TP=2`，`PP=1`，`CP=1`，开启 `--sequence-parallel`。
- expert 侧：`EP=8`，`ETP=1`。
- 8 个 Megatron rank 中，非 expert 层的 dense data parallel size 是 `8 / (TP * PP * CP) = 4`。
- expert 层使用单独的 expert rank generator，`ETP * EP * PP = 8`，所以 expert data parallel size 是 1。
- 直观理解：attention / dense 参数按 TP=2、DP=4 训练；MoE expert 参数按 EP=8 分布，每张卡持有 256 / 8 = 32 个 local experts。

这个“dense DP 和 expert DP 不同”的点容易误判。`initialize.py` 调用 Megatron `mpu.initialize_model_parallel(...)` 时会同时传入 `tensor_model_parallel_size` 和 `expert_model_parallel_size`，Megatron 内部为普通 transformer 层和 expert 层建两套 rank generator。

## 2. 全局流程

从一次完整训练 run 看，主链路是：

1. shell 脚本启动 Ray head，然后提交 `train.py`。
2. `train.py` 解析所有 slime + Megatron + SGLang 参数。
3. `create_placement_groups(args)` 创建 Ray placement group。
4. `create_rollout_manager(args, pgs["rollout"])` 创建 rollout manager，里面管理 SGLang engines。
5. `create_training_models(args, pgs, rollout_manager)` 创建 Megatron actor group。
6. 每个 Megatron Ray actor 初始化一个 rank：分布式组、模型、optimizer、scheduler、checkpoint、weight updater。
7. 初始 actor 权重同步到 rollout engine。
8. 每个 `rollout_id`：
   - rollout manager 生成样本。
   - rollout 样本转换成训练数据，并按 Megatron DP / micro-batch schedule 切分。
   - actor group 调用每个 rank 的 `MegatronTrainRayActor.train(...)`。
   - Megatron 执行 old logprob / ref logprob / advantage / actor train。
   - actor 备份新权重。
   - 周期性 save。
   - actor 权重再次同步到 SGLang rollout engine。
   - 周期性 eval。

对于这个 GRPO 例子，没有 critic，所以 `create_training_models` 只建 actor，不建 critic。`with_ref=True` 由 `use_kl_loss` 触发，所以 actor rank 内部除了当前 actor 权重，还会加载 reference 权重。

## 3. Ray 层如何把训练和 rollout 放到同一批 GPU

`slime/ray/placement_group.py` 中，`--actor-num-nodes 1 --actor-num-gpus-per-node 8 --colocate` 会让 placement group 申请 8 个 GPU bundle。

`RayTrainGroup._allocate_gpus_for_actor(...)` 会创建 8 个 Ray actor，每个 actor 是一个 Megatron rank。这里传给 Ray 的 `num_gpus_per_actor` 是 0.4，但 actor 被绑到 placement group bundle 上，本质上还是让每个 rank 固定落到一个逻辑 GPU bundle。

关键环境变量：

- `NOSET_VISIBLE_DEVICES` 相关变量：避免 Ray 改写可见 GPU 后影响 Megatron / SGLang 的设备关系。
- `NCCL_CUMEM_ENABLE` 等 NCCL / CUDA 环境。
- 如果 `args.use_routing_replay and role == "actor"`，给 actor 设置 `ENABLE_ROUTING_REPLAY=1`。

注意：注释明确说 critic 不能做 routing replay。R3 是 actor policy 路径的东西，critic 没有 rollout router 对齐语义。

## 4. Megatron actor 初始化

`MegatronTrainRayActor.init(...)` 是每个 Megatron rank 的初始化入口。

核心顺序：

1. monkey patch torch distributed，注册可 reload 的 process group。
2. 调用 `slime/backends/megatron_utils/initialize.py:init(args)`。
3. 读取 HF config 和 tokenizer。
4. 调用 `initialize_model_and_optimizer(args, role)`。
5. 记录训练并行信息：
   - `dp_size`
   - `cp_size`
   - `vpp_size`
   - `microbatch_group_size_per_vp_stage`
6. actor 角色创建 `TensorBackuper`，先备份当前 actor。
7. 如果需要 ref，调用 `load_other_checkpoint("ref", args.ref_load)`。
8. 初始化 weight updater，用于 actor -> rollout engine 权重同步。

`initialize.py:init(args)` 做的是 Megatron 标准初始化：

- `set_args(args)` 把 argparse namespace 放进 Megatron global vars。
- `mpu.initialize_model_parallel(...)` 建 TP / PP / CP / EP / ETP / DP 组。
- 检查 numpy 版本。
- 设置随机种子。
- `_build_tokenizer(args)`。
- 初始化 Megatron microbatch calculator。
- 可选 TP communication overlap。
- 可选 custom init hook。

`is_megatron_main_rank()` 的定义是：

- DP rank 为 0。
- TP rank 为 0。
- PP last stage。

所以日志、tracking、train metrics 通常只在这个 rank 记录。

## 5. 模型如何从 slime spec 进入 Megatron GPTModel

`slime/backends/megatron_utils/model.py:setup_model_and_optimizer(...)` 调用：

```python
model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)
```

`get_model_provider_func(...)` 位于 `model_provider.py`，默认路径会：

1. 用 Megatron `core_transformer_config_from_args(args)` 从命令行构造 `TransformerConfig`。
2. 如果 `args.spec` 不为空，用 Megatron `import_module(args.spec)` 导入模型 spec。
3. 对 Qwen3.5，本例导入 `slime_plugins.models.qwen3_5:get_qwen3_5_spec`。
4. 构造 `GPTModel(...)`，传入：
   - transformer layer spec。
   - padded vocab size。
   - max sequence length。
   - rope 参数。
   - embedding/output weight 是否 untie。
   - `parallel_output=True`。

Qwen3.5 spec 的作用：

- 先基于 Megatron `get_gpt_decoder_block_spec(config, use_transformer_engine=True, ...)` 拿到标准 GPT decoder block spec。
- 如果 `args.num_experts` 为空，会把 `config.moe_layer_freq` 设成全 dense；本例有 `num_experts=256`，所以保留 MoE。
- 读取 HF config 中的 `layer_types`。如果没有，按 `full_attention_interval` 推导 full attention / linear attention。
- 对 `linear_attention` 层，把 Megatron spec 中的 self attention 替换成 slime 自定义 `Attention`。
- 这个自定义 `Attention` 内部包了 Qwen3.5 的 `Qwen3_5GatedDeltaNet`，支持 GDN 后端。

所以本例的模型不是纯 Megatron 原生 GPT block，而是：

- GPTModel / transformer block / MoE MLP 使用 Megatron。
- Qwen3.5 特有的部分注意力层由 slime plugin 替换。
- checkpoint 转换和权重命名再由 slime 的 HF/Megatron bridge 或 raw 映射处理。

## 6. rollout routing replay 的数据链路

`--use-rollout-routing-replay` 在参数校验阶段会自动打开：

```python
if args.use_rollout_routing_replay:
    args.use_routing_replay = True
```

也就是说 R3 包含训练侧 routing replay 能力。

### 6.1 SGLang 请求让 rollout 返回 routed experts

普通 rollout 路径 `slime/rollout/sglang_rollout.py` 中：

```python
payload = {
    "sampling_params": sampling_params,
    "return_logprob": True,
}
if args.use_rollout_routing_replay:
    payload["return_routed_experts"] = True
```

streaming rollout 路径也一样。SGLang engine 启动参数里也会设置：

```python
kwargs["enable_return_routed_experts"] = True
```

因此每次 SGLang generate 除了 token logprob，还会把 MoE router 的 top-k expert id 编码到 `meta_info["routed_experts"]`。

### 6.2 Sample 解码 routed experts

`slime/utils/types.py:Sample.append_response(...)` 会解码 `routed_experts`：

- 从 meta_info 里读取 int32 数组。
- 期望元素数是：

```text
expected_rows * args.num_layers * args.moe_router_topk
```

- reshape 成：

```text
[num_generated_logit_positions, num_layers, topk]
```

对本例就是：

```text
[num_generated_logit_positions, 40, 8]
```

`expected_rows = len(sample.tokens) - 1 - routed_experts_start_len`。这里的 `len(tokens)-1` 对应 next-token prediction 的 logit position：输入 token 序列长度是 T，产生可监督 next-token logits 的位置是 T-1 个。

如果是 partial / streaming append，`routed_experts_start_len` 会让新片段拼到已有 `sample.rollout_routed_experts` 后面。

### 6.3 rollout manager 转成 train_data

`RolloutManager._convert_samples_to_train_data(...)` 会把 sample 字段打包成训练字段：

- `tokens`
- `response_lengths`
- `rewards`
- `loss_masks`
- `rollout_ids`
- `rollout_mask_sums`
- 可选 `rollout_log_probs`
- 可选 top-p replay 字段
- 可选 `rollout_routed_experts`

只要第一个 sample 有 `rollout_routed_experts`，就会写入：

```python
train_data["rollout_routed_experts"] = [
    sample.rollout_routed_experts for sample in samples
]
```

随后 `_split_train_data_by_dp(...)` 根据当前 Megatron dense DP size 和 micro-batch schedule，把字段分到每个 DP rank 的 `rollout_data`。`rollout_routed_experts` 会跟 `tokens`、`loss_masks` 一起按 partition 分发。

`process_rollout_data(...)` 在训练 rank 上按 DP rank 取出对应 Box，并恢复本 rank 的 `total_lengths`。

## 7. R3 在 Megatron actor 内如何填队列

actor 训练入口是 `MegatronTrainRayActor.train(...)`：

1. `_get_rollout_data(...)` 从 Ray object store 取本 DP rank 的数据。
2. 把 `tokens`、`loss_masks`、logprob 等移动到 GPU。
3. `train_actor(...)` 创建 `DataIterator`。
4. 如果 `args.use_rollout_routing_replay`，先调用 `fill_routing_replay(...)`。

`fill_routing_replay(...)` 的核心是按后续训练完全相同的 micro-batch 顺序，把 rollout expert id 预先写入每个 MoE router 的 replay queue。

步骤：

1. reset 所有 data iterator。
2. 循环 `sum(num_microbatches)` 次，每次取一个 micro-batch：

```python
batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens"])
```

3. 调用 `prepare_routed_experts_for_routing_replay(...)`，把 rollout expert id 对齐到 Megatron packed token layout。
4. 遍历本 rank 持有的 model chunks / VP stage / local transformer layers。
5. 根据 `config.moe_layer_freq` 跳过 dense 层。
6. 对每个 MoE layer，取：

```python
layer_routed_experts = rollout_routed_experts[:, layer_id]
```

7. 写入当前 router 对应的 `RoutingReplay`：

```python
RoutingReplay.all_routing_replays[routing_replay_offset].record(layer_routed_experts)
```

8. 删除 `rollout_data["rollout_routed_experts"]`，避免后续通用 data path 误用。
9. reset data iterator，让 old logprob / train 能从同一 micro-batch schedule 开始。

这里有几个关键对齐点。

第一，rollout routed experts 的原始 shape 是 `[T-1, num_layers, topk]`，而 Megatron model forward 处理的是 input token stream `[T]`。`prepare_routed_experts_for_routing_replay` 会对每条序列多 pad 1 行 expert id，让它能和 Megatron 的 token stream 长度一致。最后一行不参与真实 loss，expert id 可以是占位值。

第二，CP 对齐和 `get_batch(...)` 一致：

- `allgather_cp=True` 时，先全局 concat，再按 CP rank chunk。
- 否则每条序列先按 `slice_with_cp(...)` 切，再 concat。

本例 `CP=1`，所以这部分实际不切。

第三，SP 对齐很重要。本例 `--sequence-parallel` 且 `TP=2`，所以 routed experts 在最后会按 TP rank 切：

```python
seqlen = routed_experts.size(0)
start = seqlen // tp_size * tp_rank
end = seqlen // tp_size * (tp_rank + 1)
routed_experts = routed_experts[start:end]
```

Megatron sequence parallel 会把 sequence dimension 分给 TP ranks。R3 队列必须与每个 TP rank 实际看到的 local token slice 一致，否则 replay 的 top_indices shape 会和 router scores shape 对不上。

## 8. RoutingReplay 的运行阶段

`slime/utils/routing_replay.py` 定义了一个很轻的队列对象：

- `top_indices_list`: 每个 micro-batch 记录一次 top-k expert id。
- `forward_index`: replay forward 消费位置。
- `backward_index`: replay backward 消费位置。
- `record(...)`: 把 expert id 复制到 CPU pinned memory。
- `pop_forward(...)`: 取一个 batch 到当前 CUDA device。
- `pop_backward(...)`: 取一个 batch 到当前 CUDA device。
- `clear_forward(...)`: 只重置 forward index。
- `clear(...)`: 清空整个队列。

状态由环境变量控制：

- `fallthrough`: 不 replay，调用 Megatron 原 top-k。
- `record`: 调用 Megatron 原 top-k，同时记录 top_indices。
- `replay_forward`: 从队列取 top_indices，按这些 expert id gather scores。
- `replay_backward`: backward activation recompute 时从队列取 top_indices。

`use-routing-replay` 和 `use-rollout-routing-replay` 的区别：

- `--use-routing-replay`: 训练内部先 record 当前 actor forward 的 top-k，再在 backward recompute 阶段 replay，保证同一次训练 step 的 forward/backward 路由一致。
- `--use-rollout-routing-replay`: rollout 阶段 SGLang 已经选过 expert，训练阶段直接 replay rollout expert id，让 actor old logprob / actor train 尽量使用和 rollout 一致的 MoE 路由。

R3 下，reference model forward 会被设成 `fallthrough`。代码注释也说明：actor path replay rollout routing，ref logprob 则用正常 routing。因此一开始 actor/ref KL 不一定严格为 0。

## 9. actor train 中 R3 的时序

`train_actor(...)` 中和 R3 相关的顺序可以简化成：

```text
fill_routing_replay()

if need ref_log_probs:
    ROUTING_REPLAY_STAGE = fallthrough
    switch ref
    compute ref logprob

switch actor / old_actor
if need old log_probs:
    ROUTING_REPLAY_STAGE = replay_forward
    compute actor old logprob
    RoutingReplay.clear_all_forward()

compute advantages

ROUTING_REPLAY_STAGE = replay_backward
train_one_step()

RoutingReplay.clear_all()
backup new actor weights
```

为什么 old logprob 后要 `clear_all_forward()`：

- fill 阶段已经为每个 router 存了一整轮 micro-batch 队列。
- old logprob forward 会消耗 `forward_index`。
- 训练 forward 还要再 replay 同一批 rollout expert id。
- 所以只重置 forward index，不清空数据。

`train_one_step(...)` 里还有一个细节：actor train 外层把 stage 设成 `replay_backward`，但 `model.py` 的 `forward_step(...)` 在真正调用 `model(**forward_kwargs)` 前会临时改成 `replay_forward`，forward 结束后恢复旧 stage。

这使得：

- 正常 forward 使用 `pop_forward()`。
- activation checkpoint / recompute 触发的 backward-time forward 使用恢复后的 `replay_backward`，从 `pop_backward()` 取同一批 top-k。

这是 R3 能同时覆盖 forward 和 recompute backward 的关键。

## 10. Megatron 内部：MoE router 被 slime patch 到哪里

当前仓库通过 `docker/patch/latest/megatron.patch` 修改 Megatron：

1. 在 `megatron/core/transformer/moe/moe_utils.py` 的 `topk_routing_with_score_function(...)` 中，把局部 `compute_topk` 包成：

```python
from slime.utils.routing_replay import get_routing_replay_compute_topk
compute_topk = get_routing_replay_compute_topk(compute_topk)
```

2. 在 `megatron/core/transformer/moe/router.py` 的 `TopKRouter.__init__` 中注册：

```python
from slime.utils.routing_replay import register_routing_replay
register_routing_replay(self)
```

本地 `../Megatron-LM` 中也已经能看到这两个 patch。

`register_routing_replay(self)` 做两件事：

- 给每个 `TopKRouter` 创建一个独立 `RoutingReplay()`。
- 注册 forward pre-hook，在 router forward 前把全局 `ROUTING_REPLAY` 指到当前 router 的队列。

所以虽然 `get_routing_replay_compute_topk(...)` 用的是模块级全局变量 `ROUTING_REPLAY`，但每个 router forward 前都会把它切到“当前层当前 router”的队列。

`RoutingReplay.all_routing_replays` 的顺序来自 router 构造顺序。`fill_routing_replay(...)` 遍历 model chunk / local layer 的顺序必须和 router 构造顺序一致。这个顺序假设成立后，offset 就能把 rollout 的第 N 个 MoE layer expert id 写到第 N 个 router queue。

## 11. Megatron 内部：TopKRouter 做了什么

Megatron `TopKRouter.forward(input)` 的主路径：

1. `_maintain_float32_expert_bias()`。
2. `apply_input_jitter(...)`。
3. `gating(input)`，用 router weight 做线性投影，得到 logits。
4. 可选 force load balancing。
5. `routing(logits)`。

`routing(logits)` 中：

1. reshape logits 到 `[num_tokens, num_moe_experts]`。
2. apply z-loss。
3. 调 `topk_routing_with_score_function(...)`。
4. 可选 token dropping。
5. 可选 aux loss / seq aux loss / global aux loss。
6. 可选 expert bias update。
7. 返回：
   - `probs`: `[num_tokens, num_experts]` dense routing probability。
   - `routing_map`: `[num_tokens, num_experts]` bool mask。

对本例：

- `score_function="softmax"`。
- `topk=8`。
- 没有设置 pre-softmax 时，正常逻辑是先在 logits 上 top-k，再对 top-k logits 做 softmax。
- R3 replay 时，top-k indices 不来自当前 Megatron logits 的 `torch.topk`，而来自 rollout 队列；Megatron 仍然会用当前 logits gather 这些 expert 的 score，再计算 top-k 内概率。

也就是说 R3 固定的是“选哪些 expert”，不是把 rollout 的 router probability 也原样复制过来。概率仍由训练时 actor 当前参数计算。

## 12. Megatron 内部：MoELayer 的 route / dispatch / expert / combine

`../Megatron-LM/megatron/core/transformer/moe/moe_layer.py` 中，`MoELayer.forward(...)` 的结构非常清楚：

```text
shared_experts_compute
route
preprocess
dispatch
routed_experts_compute
combine
```

展开看：

1. `route(hidden_states)` 调 `self.router(hidden_states)`，拿到 `probs` 和 `routing_map`。
2. `preprocess(hidden_states, probs, routing_map)` 调 token dispatcher 的 `dispatch_preprocess(...)`。
3. `dispatch(hidden_states, probs)` 执行实际跨 EP rank 的 token dispatch。
4. `routed_experts_compute(...)`：
   - dispatcher postprocess 得到按 expert 排好的 tokens。
   - local experts 对本 rank 持有的 tokens 做 MLP。
   - dispatcher combine preprocess。
5. `combine(...)`：
   - token combine 把 expert output 发回原 token 位置。
   - 加 shared expert output。

本例 `--moe-token-dispatcher-type alltoall`，所以 dispatcher 是 `MoEAlltoAllTokenDispatcher`。如果启用 Megatron DeepEP，一般会用 `--moe-token-dispatcher-type flex` 和相关 deepep 参数，路径会变成 `MoEFlexTokenDispatcher`。

## 13. packed batch 如何进入 Megatron forward

训练和 forward-only 都通过 `get_batch(...)` 生成 Megatron batch。

输入 rollout_data 是 list-of-samples：

- `tokens`: 每条样本完整 token。
- `loss_masks`: response token mask。
- `total_lengths`
- `response_lengths`
- 可选 logprob / top-p / routed experts 等。

`get_batch(...)` 会：

1. `data_iterator.get_next(keys)` 取当前 micro-batch 的样本列表。
2. 保留原始 token list 到 `unconcat_tokens`。
3. 处理 CP：
   - `allgather_cp=False` 时，每条序列先 `slice_with_cp(...)`。
   - `allgather_cp=True` 时，先 concat 再 chunk。
4. concat 多条序列。
5. pad 到 `tp_size * data_pad_size_multiplier` 的倍数，减少碎片并满足并行要求。
6. 构造 `PackedSeqParams(cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, qkv_format="thd")`。
7. 把 tokens 变成 `[1, T_padded]`。
8. 把 response-only loss mask pad 到完整 token stream，并和 tokens shape 对齐。

最终 `model.py` 的 forward step 调用：

```python
output_tensor = model(
    input_ids=batch["tokens"],
    position_ids=None,
    attention_mask=None,
    labels=None,
    packed_seq_params=batch["packed_seq_params"],
    loss_mask=batch["full_loss_masks"],
)
```

Qwen3.5 的 GDN / attention 自定义层会使用 `packed_seq_params.cu_seqlens_q` 来支持 varlen。

## 14. Megatron train_one_step 的执行

`slime/backends/megatron_utils/model.py:train(...)` 会按 rollout 内的 step 调 `train_one_step(...)`。

`train_one_step(...)` 的核心：

1. 清空 grad buffer 和 optimizer grad。
2. 定义 Megatron pipeline forward step。
3. `get_forward_backward_func()` 拿到 Megatron 的 pipeline schedule。
4. 调：

```python
forward_backward_func(
    forward_step_func=...,
    data_iterator=data_iterator,
    model=model,
    num_microbatches=num_microbatches,
    seq_length=args.seq_length,
    micro_batch_size=args.micro_batch_size,
    decoder_seq_length=args.decoder_seq_length,
    forward_only=False,
)
```

5. Megatron pipeline engine 负责 forward/backward、PP 通信、micro-batch 调度。
6. optimizer `prepare_grads()` / grad norm / step。
7. scheduler 按 `step_global_batch_size` 推进。
8. PP last stage、TP rank 0、DP rank 0 记录 train metrics。

slime 的 loss closure 来自 `loss_function(args, batch, num_microbatches, step_global_batch_size)`。这部分已经在 `08_slime_ppo_grpo_algorithm_analysis.md` 里详细记录；这里重点是 Megatron 只需要一个 standard forward step + loss closure，算法细节被 slime 封装在 loss callback 里。

## 15. 为什么 Qwen3.5 MoE 特别需要关注 R3

MoE RL 训练里，rollout engine 和 train engine 可能出现几类不一致：

- SGLang 和 Megatron 使用的 kernel / dtype / dispatch 实现不同。
- rollout 可能是 FP8 或特殊 serving kernel，训练是 BF16 / Megatron kernel。
- router logits 很接近时，top-k expert 的边界对数值误差敏感。
- 每个 token 进入不同 expert 后，后续 hidden state 和 logprob 都会变化。

如果 GRPO/PPO 的 old logprob 是训练侧重算出来的，而训练侧路由和 rollout 侧路由不同，那么 policy ratio 的 old policy 口径会偏离真正采样策略。

R3 的目标是让训练 actor path 重放 rollout 阶段的 expert 选择：

```text
rollout SGLang 选 expert
-> slime 记录 expert id
-> Megatron actor old logprob replay expert id
-> Megatron actor train forward replay expert id
-> Megatron backward recompute replay expert id
```

它不解决所有 train / infer mismatch，但能把 MoE expert 选择这个最离散、最容易放大的 mismatch 固定住。

## 16. 权重同步在这个流程中的位置

actor 初始化后会立刻：

```python
actor_model.update_weights()
```

每个 rollout 训练后也会：

```python
actor_model.update_weights()
```

在 colocate 场景，本例一般走 `UpdateWeightFromTensor`：

- actor rank 持有 Megatron 参数。
- `weights_backuper.backup("actor")` 保存最新 actor 权重视图。
- weight updater 把 actor 权重同步到同机 SGLang engine。

MoE 模型还有 expert 参数 routing 相关的优化路径，入口在：

- `slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py`
- `slime/backends/megatron_utils/update_weight/expert_routing.py`

这部分属于“训练后把 Megatron 权重发回 rollout engine”，不是 R3 的 expert routing replay。名字都包含 routing，但语义不同：

- R3 routing replay：训练 forward 使用哪个 expert。
- weight update expert routing：同步 expert 权重时，如何把 Megatron expert 参数发到对应 SGLang engine / rank。

## 17. 排查和阅读建议

如果要 debug R3，优先检查：

1. SGLang 是否真的返回 routed experts：
   - 请求 payload 是否有 `return_routed_experts=True`。
   - engine kwargs 是否有 `enable_return_routed_experts=True`。
2. `Sample.append_response(...)` 是否 reshape 成 `[rows, num_layers, topk]`。
3. `RolloutManager._convert_samples_to_train_data(...)` 是否写入 `rollout_routed_experts`。
4. `_split_train_data_by_dp(...)` 是否把该字段分给每个 DP rank。
5. `fill_routing_replay(...)` 是否在 actor train 前执行。
6. `prepare_routed_experts_for_routing_replay(...)` 的 shape 是否和 SP / CP 后的 router scores 一致。
7. `RoutingReplay.all_routing_replays` 数量是否等于本 rank local MoE routers 数量。
8. `ROUTING_REPLAY_STAGE` 在 ref / old logprob / actor train / backward recompute 的切换是否符合预期。

常见 shape 预期：

- rollout sample routed experts：`[T - 1, num_layers, topk]`。
- Qwen3.5 本例：`[T - 1, 40, 8]`。
- padding 后进入 Megatron replay：`[T_packed_local, num_layers, topk]`。
- 每层写入 router queue：`[T_packed_local, topk]`。
- router replay 时，Megatron scores shape：`[T_packed_local, num_experts]`。
- replay top_indices shape：`[T_packed_local, topk]`。

如果 assert 报 `top_indices shape ... does not match scores shape ...`，优先看 CP / SP / padding / micro-batch 顺序是否和 `get_batch(...)` 一致。

## 18. 一句话总结

slime 把 Megatron 当作“分布式训练执行引擎”：Ray 负责 rank 生命周期和 GPU 放置，rollout manager 负责生成并打包样本，Megatron actor 负责 model/optimizer/forward_backward/loss，weight updater 负责把新 actor 发回 SGLang。R3 则是在这条链路上额外把 SGLang rollout 的 MoE top-k expert id 带回训练侧，并通过 Megatron router patch 固定 actor path 的 expert 选择，从而降低 MoE RL 中 rollout 和训练路由不一致带来的 logprob / gradient mismatch。
