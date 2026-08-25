# Qwen3.5 MoE 模型在 slime + Megatron 中如何构建

本文参考 `/mnt/shared-storage-user/huanghaian/code/slime_package/Megatron-LM/hha_code/megatron_model_build.md` 的梳理方式，只补 slime 接入 Megatron 的部分。核心是分清两个层次：

```text
1. 调用链:
   slime / Megatron 的工厂函数谁调用谁。

2. 模型树:
   最终 torch.nn.Module 里谁包含谁。
```

slime 的 `model_provider.py`、Qwen3.5 的 `get_qwen3_5_spec()` 都属于调用链里的“模型选择 / spec 修改”层，不是最终模型树里的模块。最终模型树仍然是 Megatron-Core 的：

```text
GPTModel
├── LanguageModelEmbedding       # pre_process=True 的 PP/VPP stage
├── TransformerBlock
│   └── TransformerLayer x local_layers
│       ├── input_layernorm
│       ├── self_attention       # Qwen3.5 部分层会被 slime spec 替换
│       ├── self_attn_bda
│       ├── pre_mlp_layernorm
│       ├── mlp                  # Qwen3.5 MoE 层这里是 MoELayer
│       └── mlp_bda
└── output_layer                 # post_process=True 的 PP/VPP stage
```

## 1. 本例的模型参数

入口脚本是 `scripts/models/qwen3.5-35B-A3B.sh`，核心参数：

```bash
--spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

--num-layers 40
--hidden-size 2048
--ffn-hidden-size 512
--num-attention-heads 16
--num-query-groups 2
--kv-channels 256
--vocab-size 248320

--position-embedding-type rope
--rotary-percent 0.25
--rotary-base 10000000

--num-experts 256
--moe-router-topk 8
--moe-layer-freq "$MOE_LAYER_FREQ"
--moe-ffn-hidden-size 512
--moe-shared-expert-intermediate-size 512
--moe-token-dispatcher-type alltoall
--moe-router-score-function softmax
--moe-router-dtype fp32
--moe-grouped-gemm
```

`MOE_LAYER_FREQ` 在脚本里被构造成长度 40 的 list。`FIRST_K_DENSE_REPLACE=0`，所以 40 层都是 MoE 层：

```text
[1, 1, 1, ..., 1]
```

这里有两类信息：

- 结构参数：`num_layers`、`hidden_size`、`num_experts`、`moe_router_topk` 等，进入 Megatron `TransformerConfig`。
- spec 参数：`--spec slime_plugins.models.qwen3_5 get_qwen3_5_spec`，告诉 slime/Megatron 每层内部用什么 module spec。

## 2. slime 下的构建调用链

和 Megatron 原生 pretrain 脚本相比，slime 不走 `pretrain_gpt.py -> gpt_builder`，而是从 Ray actor 里初始化训练模型：

```text
train.py
└── create_training_models(...)
    └── RayTrainGroup.create(...)
        └── MegatronTrainRayActor.init(...)
            └── initialize_model_and_optimizer(args, role)
                └── setup_model_and_optimizer(args, role) # slime 中自己写了一遍
                    └── get_model(
                        model_provider_func = get_model_provider_func(args, role),
                        model_type = ModelType.encoder_or_decoder
                    )
                        ├── Megatron get_model 判断 pre_process / post_process / vp_stage
                        └── 调 slime 返回的 model_provider(...)
                            ├── core_transformer_config_from_args(args)
                            ├── import / 调用 args.spec
                            ├── 得到 transformer_layer_spec
                            └── GPTModel(
                                config,
                                transformer_layer_spec,
                                vocab_size,
                                max_sequence_length,
                                pre_process,
                                post_process,
                                ...
                            )
```

关键点：Megatron 训练框架仍然调用自己的 `get_model(...)`。slime 只是把 `model_provider_func` 换成了自己的 provider。

对应源码：

- `slime/backends/megatron_utils/actor.py:init(...)`
- `slime/backends/megatron_utils/model.py:initialize_model_and_optimizer(...)`
- `slime/backends/megatron_utils/model.py:setup_model_and_optimizer(...)`
- `slime/backends/megatron_utils/model_provider.py:get_model_provider_func(...)`

`setup_model_and_optimizer(...)` 里真正触发模型构建的是：

```python
model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)
```

后面才是 optimizer 和 scheduler：

```text
model build
-> get_megatron_optimizer(...)
-> OptimizerParamScheduler(...)
-> load_checkpoint(...)
```

所以本文只看模型 build，optimizer / checkpoint 属于后续阶段。

### 2.1 Megatron `get_model()` 到底做什么

你的理解是对的：`get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)` 里，`get_model()` 是 Megatron 通用训练代码，slime 没有改它。它的核心输入就是一个 `model_provider_func`。

可以把职责分成两边：

```text
model_provider_func:
    负责真正构造“裸模型”。
    在 Qwen3.5 这里，slime 的 provider 会创建 TransformerConfig，
    调 get_qwen3_5_spec 得到 transformer_layer_spec，
    然后 new GPTModel(...)。

Megatron get_model:
    负责围绕裸模型做训练框架需要的前后处理。
    它不关心 Qwen3.5 哪些层是 linear attention，
    也不关心 MoE 的 layer spec 怎么生成。
```

`get_model()` 源码在 `/mnt/shared-storage-user/huanghaian/code/slime_package/Megatron-LM/megatron/training/training.py`。主流程可以简化成：

```text
get_model(model_provider_func, model_type, wrap_with_ddp=True)
├── args = get_args()
├── args.model_type = model_type
│
├── build_model()
│   ├── 如果 PP>1 且开启 VPP:
│   │   └── for vp_stage in virtual_pipeline_model_parallel_size:
│   │       ├── pre_process = is_pipeline_first_stage(..., vp_stage=i)
│   │       ├── post_process = is_pipeline_last_stage(..., vp_stage=i)
│   │       ├── model_provider_func(pre_process, post_process, vp_stage=i)
│   │       └── append 到 model list
│   │
│   └── 否则:
│       ├── pre_process = is_pipeline_first_stage()
│       ├── post_process = is_pipeline_last_stage()
│       └── model_provider_func(pre_process, post_process)
│
├── 确保返回值是 list
├── 给每个 parameter 补 tensor-parallel 默认属性
├── 打印本 TP/PP rank 上参数量
├── model.cuda(current_device)
├── 如果 fp16/bf16，包 Float16Module
├── 如果 wrap_with_ddp:
│   ├── 选择 DDP / Megatron FSDP / Torch FSDP2
│   ├── 构造 DistributedDataParallelConfig
│   └── 把每个 model chunk 包成 DP wrapper
└── 如果 data_parallel_random_init，广播 DP 参数
```

这里最关键的是 `build_model()` 这一层。Megatron 根据当前 rank 的 PP/VPP 位置算出 `pre_process`、`post_process`、`vp_stage`，然后把这些值传给 provider。provider 才真正决定构造什么模型。

所以对 Qwen3.5 MoE 来说：

```text
get_model:
    只知道“我要在当前 rank/chunk 上建一个 encoder_or_decoder 模型”。

slime provider:
    知道“这个模型要用 Qwen3.5 的 spec”。

get_qwen3_5_spec:
    知道“哪些 local layer 的 self_attention 要替换成 Qwen3.5 Attention”。

GPTModel / TransformerBlock:
    根据 provider 传入的 transformer_layer_spec 实例化最终模型树。
```

`get_model()` 的“前后处理”不只是表面包装，它和分布式训练强相关：

- `pre_process / post_process` 决定当前 PP/VPP chunk 是否有 embedding / output layer。
- VPP 时它会返回多个 model chunks，即 `[GPTModel(vp_stage=0), GPTModel(vp_stage=1), ...]`。
- FP16/BF16 时它会包 `Float16Module`。
- DDP/FSDP wrapper 是 optimizer、gradient reduce、overlap param gather 等后续训练机制的基础。
- 所有参数补 tensor-parallel attributes 后，Megatron optimizer 才能正确识别哪些参数已经被 TP 切分。

因此它不是模型结构选择层，而是 Megatron 训练框架的模型装配层。模型结构的选择权在 `model_provider_func`，slime 只是利用这个 Megatron 预留入口，把 Qwen3.5 的 provider/spec 接进去。

### 2.2 `wrap_model_provider_with_freeze` 是什么

slime 暴露给 Megatron `get_model()` 的 provider 不是直接 `_get_model_provider_func(args, role)`，而是：

```python
def get_model_provider_func(args, role="actor"):
    return wrap_model_provider_with_freeze(_get_model_provider_func(args, role), args)
```

也就是在“真正建模型的 provider”外面再包一层 freeze wrapper。

调用关系变成：

```text
Megatron get_model
└── wrapped_provider(pre_process, post_process, vp_stage)
    ├── original_provider(...)
    │   └── GPTModel(...) / custom provider / bridge provider
    ├── freeze_model_params(model, args)
    └── return model
```

这个 wrapper 不改变模型结构，也不参与 Qwen3.5 spec 选择。它只在模型已经构建出来后，根据参数名规则设置 `requires_grad`。

支持两类规则：

```text
--only-train-params-name-list:
    先把所有参数 requires_grad=False，
    再把匹配 regex 的参数打开训练。

--freeze-params-name-list:
    只把匹配 regex 的参数 requires_grad=False，
    其它参数保持可训练。
```

参数校验会禁止这两个选项同时使用，避免“只训练某些参数”和“冻结某些参数”的语义冲突。

把它放进构建链路里，边界是：

```text
_get_model_provider_func:
    决定如何建模型。
    对 Qwen3.5，就是 core_transformer_config_from_args
    -> get_qwen3_5_spec
    -> GPTModel。

wrap_model_provider_with_freeze:
    决定建完后哪些参数参与训练。
    只改 requires_grad。

Megatron get_model:
    决定 PP/VPP 调 provider、搬设备、包 Float16Module/DDP/FSDP。
```

所以如果调试“Qwen3.5 模型结构为什么这样”，主要看 `_get_model_provider_func` 和 `get_qwen3_5_spec`；如果调试“为什么某些参数没有梯度 / 不训练”，再看 `wrap_model_provider_with_freeze` 和 `freeze_model_params`。

### 2.3 `_get_model_provider_func` 是什么

`_get_model_provider_func(args, role)` 是 slime 训练侧最重要的模型 provider 选择函数。名字里有 `get`，但它不是立即构建模型，而是返回一个 provider 函数。后面 Megatron `get_model()` 会按当前 PP/VPP rank 调这个 provider。

它的返回值大致有三种可能：

```text
1. custom_model_provider_path 分支:
   返回用户自定义 provider。

2. megatron_to_hf_mode == "bridge" 分支:
   返回 Megatron Bridge 的 provider.provide。

3. 默认 raw 分支:
   返回 slime 自己定义的 model_provider 闭包。
   Qwen3.5 这个例子走这一支。
```

#### 2.3.1 custom provider 分支

如果设置了：

```text
--custom-model-provider-path
```

slime 会加载这个函数：

```python
custom_model_provider = load_function(args.custom_model_provider_path)
```

然后返回一个 wrapper。Megatron `get_model()` 调 wrapper 时，wrapper 再调用户自定义 provider：

```text
wrapped_model_provider(pre_process, post_process, vp_stage)
└── custom_model_provider(pre_process, post_process, vp_stage?)
```

这里会检查自定义函数签名里有没有 `vp_stage`：

- 有 `vp_stage`：传入 `vp_stage`。
- 没有 `vp_stage`：只传 `pre_process`、`post_process`。

如果当前 role 是 critic，且 `post_process=True`，slime 会把模型最后的 output layer 替换成 value head：

```python
model.output_layer = LinearForLastLayer(
    input_size=model.config.hidden_size,
    output_size=1,
    config=model.config,
)
```

所以 custom provider 分支的语义是：用户完全接管模型构建，slime 只补 critic head 这类 RL 角色差异。

#### 2.3.2 Megatron Bridge 分支

如果：

```text
args.megatron_to_hf_mode == "bridge"
```

slime 会走 Megatron Bridge：

```python
bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
bridge = patch_auto_bridge_hf_config(bridge)
provider = bridge.to_megatron_provider(load_weights=False)
```

注意这里 `load_weights=False`，也就是说 Bridge provider 只负责建 Megatron 模型结构，不在 provider 阶段加载权重。真实权重仍然是后面 `load_checkpoint(...)` 或 bridge checkpoint load 路径处理。

slime 会手动把训练并行参数写到 bridge provider 上：

```text
tensor_model_parallel_size
pipeline_model_parallel_size
expert_model_parallel_size
expert_tensor_parallel_size
sequence_parallel
context_parallel_size
variable_seq_lengths
moe_token_dispatcher_type
num_layers_in_first_pipeline_stage
num_layers_in_last_pipeline_stage
```

然后：

```python
provider.finalize()
return provider.provide
```

critic role 仍然会包一层 `_critic_provide`，在 `post_process=True` 时替换 output layer 为 `LinearForLastLayer`。

Bridge 分支的语义是：模型结构主要由 HF config + Megatron Bridge 推导，slime 负责把训练并行设置补进去。

#### 2.3.3 默认 raw 分支

如果没有 custom provider，也不是 bridge mode，就进入默认分支。Qwen3.5 当前脚本没有设置 `--custom-model-provider-path`，也没有设置 `--megatron-to-hf-mode bridge`，所以默认 `raw`，走这一支。

默认分支返回一个闭包：

```python
def model_provider(pre_process=True, post_process=True, vp_stage=None):
    ...
    return GPTModel(...)
```

这才是 Qwen3.5 例子真正构建模型的 provider。主流程是：

```text
model_provider(pre_process, post_process, vp_stage)
├── use_te = args.transformer_impl == "transformer_engine"
├── config = core_transformer_config_from_args(args)
│
├── 选择 transformer_layer_spec
│   ├── 如果 args.spec:
│   │   ├── import_module(args.spec)
│   │   ├── 如果导入对象 callable:
│   │   │   └── result = spec_fn(args, config, vp_stage)
│   │   ├── 如果 result 是完整 provider:
│   │   │   └── 直接委托 result(pre_process, post_process, vp_stage)
│   │   └── 否则 transformer_layer_spec = result
│   │
│   └── 如果没有 args.spec:
│       ├── args.num_experts 非空:
│       │   └── get_gpt_decoder_block_spec(config, use_transformer_engine=use_te, ...)
│       └── 否则:
│           └── get_gpt_layer_with_transformer_engine_spec(...)
│               或 get_gpt_layer_local_spec(...)
│
├── 如果 fp8_param_gather，进入 fp8_model_init context
├── 组装 GPTModel kwargs
├── GPTModel(**kwargs)
├── 如果 role == critic 且 post_process=True，替换 output_layer 为 value head
└── return model
```

对 Qwen3.5，这里的关键路径是：

```text
args.spec =
    ("slime_plugins.models.qwen3_5", "get_qwen3_5_spec")

import_module(args.spec)
    -> get_qwen3_5_spec

get_qwen3_5_spec(args, config, vp_stage)
    -> transformer_layer_spec

GPTModel(transformer_layer_spec=transformer_layer_spec, ...)
```

`GPTModel` 的主要 kwargs 来自 slime provider：

```text
config
transformer_layer_spec
vocab_size=args.padded_vocab_size
max_sequence_length=args.max_position_embeddings
pre_process
post_process
fp16_lm_cross_entropy
parallel_output=True
share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights
position_embedding_type=args.position_embedding_type
rotary_percent=args.rotary_percent
rotary_base=args.rotary_base
rope_scaling=args.use_rope_scaling
vp_stage
mtp_block_spec  # 如果 args.mtp_num_layers
```

所以 `_get_model_provider_func` 的职责可以概括为：

```text
根据 args 选择“谁来提供 Megatron 模型”：

custom provider:
    用户完全自定义。

bridge provider:
    Megatron Bridge 从 HF config 生成 Megatron provider。

default raw provider:
    slime 手动用 Megatron GPTModel + layer spec 构造模型。
```

Qwen3.5 MoE 当前例子属于 default raw provider，且通过 `args.spec` 把 `get_qwen3_5_spec` 接入 layer spec 选择阶段。

#### 2.3.4 repo 里三种 provider 路径的例子

这三条分支在仓库里都有例子，后续可以按下面路径继续看。

custom provider 分支：

```text
scripts/models/gemma4-12B.sh
scripts/models/gemma4-26B-A4B.sh
scripts/models/gemma4-31B.sh
```

这些脚本使用：

```bash
--custom-model-provider-path "slime_plugins.models.gemma4_provider.model_provider"
```

也就是让 Gemma4 插件完全提供 model provider。适合看 `custom_model_provider_path` 分支怎么绕过默认 `GPTModel + args.spec` 逻辑。

Megatron Bridge 分支：

```text
scripts/run-gpt-oss-20B.sh
examples/geo3k_vlm/run_geo3k_vlm.sh
examples/geo3k_vlm/run_geo3k_qwen35.sh
examples/geo3k_vlm/run_geo3k_vlm_sft.sh
examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn.py
examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn_ppo_npu.py
examples/geo3k_vlm_multi_turn/run_geo3k_vlm_multi_turn_grpo_npu.py
tests/test_qwen2.5_0.5B_short.py
tests/test_qwen2.5_0.5B_debug_rollout_then_train.py
tests/test_qwen2.5_0.5B_sglang_config.py
```

这些脚本或测试使用：

```bash
--megatron-to-hf-mode bridge
```

适合看 `AutoBridge.from_hf_pretrained(...) -> bridge.to_megatron_provider(load_weights=False) -> provider.provide` 这条路。

默认 raw provider + `--spec` 分支：

```text
scripts/models/qwen3.5-35B-A3B.sh
scripts/run-qwen3.5-35B-rl-eagle-hha.sh
scripts/run-qwen3.5-35B-A3B-sft.sh
scripts/run-qwen3.5-27B.sh
```

这些脚本使用 Qwen3.5 spec：

```bash
--spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
```

并且没有设置 `--megatron-to-hf-mode bridge` 时，`--megatron-to-hf-mode` 默认是 `raw`。这就是本文 Qwen3.5 MoE 例子的主路径。

## 3. 参数先进入 Megatron TransformerConfig

slime 的 provider 里先做：

```python
config = core_transformer_config_from_args(args)
```

这一步和 Megatron 原生 GPT builder 是同一个心智模型：CLI 上的大量参数会被整理进 `TransformerConfig`。

几个本例重要映射：

```text
args.num_layers                         -> config.num_layers
args.hidden_size                        -> config.hidden_size
args.num_attention_heads                -> config.num_attention_heads
args.group_query_attention              -> config.group_query_attention
args.num_query_groups                   -> config.num_query_groups
args.kv_channels                        -> config.kv_channels
args.normalization / norm_epsilon        -> config normalization / epsilon
args.position_embedding_type             -> config.position_embedding_type
args.rotary_base                         -> config.rotary_base

args.num_experts                         -> config.num_moe_experts
args.moe_layer_freq                      -> config.moe_layer_freq
args.moe_router_topk                     -> config.moe_router_topk
args.moe_ffn_hidden_size                 -> config.moe_ffn_hidden_size
args.moe_shared_expert_intermediate_size -> config.moe_shared_expert_intermediate_size
args.moe_token_dispatcher_type           -> config.moe_token_dispatcher_type
```

slime 还会在 `slime/backends/megatron_utils/arguments.py` 做 HF config 校验。比如：

- HF `hidden_size` 必须等于 Megatron `hidden_size`。
- HF `num_hidden_layers` 必须等于 Megatron `num_layers`。
- HF `moe_intermediate_size` 必须等于 Megatron `moe_ffn_hidden_size`。
- HF `shared_expert_intermediate_size` 必须等于 Megatron `moe_shared_expert_intermediate_size`。
- HF `rope_theta` 必须等于 Megatron `rotary_base`。
- HF `tie_word_embeddings` 和 Megatron `untie_embeddings_and_output_weights` 要反向匹配。

这解释了为什么 Megatron backend 下脚本要手写模型参数，但 slime 仍然能防一部分错配：Megatron 需要显式 args 来建模型，slime 用 HF config 做一致性校验。

## 4. `--spec` 在 slime 里具体怎么生效

`--spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"` 被 Megatron argparse 解析成一个二元路径。slime provider 中：

```python
if args.spec is not None:
    transformer_layer_spec = import_module(args.spec)
    if callable(transformer_layer_spec):
        result = transformer_layer_spec(args, config, vp_stage)
        ...
        transformer_layer_spec = result
```

`import_module(args.spec)` 来自 Megatron `spec_utils.py`，本质相当于：

```python
from slime_plugins.models.qwen3_5 import get_qwen3_5_spec
```

然后 slime 调：

```python
get_qwen3_5_spec(args, config, vp_stage)
```

这个函数返回的不是 `GPTModel`，而是 `transformer_layer_spec`。后面 slime 再把它交给 Megatron：

```python
model = GPTModel(
    config=config,
    transformer_layer_spec=transformer_layer_spec,
    vocab_size=args.padded_vocab_size,
    max_sequence_length=args.max_position_embeddings,
    pre_process=pre_process,
    post_process=post_process,
    ...
)
```

所以 `--spec` 的职责是：

```text
决定 TransformerBlock 里面每个 TransformerLayer 的子模块 spec。
```

不是：

```text
替代 Megatron GPTModel。
```

slime 也支持另一种更重的接口：如果 `args.spec` 返回的是一个带 `pre_process` 参数的 callable，slime 会把它当完整 model provider 调用。这是给特殊多模态模型等场景用的。Qwen3.5 这里不是这条路，Qwen3.5 返回的是 block/layer spec。

## 5. Qwen3.5 的 `get_qwen3_5_spec` 做了什么

源码在 `slime_plugins/models/qwen3_5.py`。

主逻辑：

```python
def get_qwen3_5_spec(args, config, vp_stage):
    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers

    kwargs = {"use_transformer_engine": True}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage

    transformer_layer_spec = get_gpt_decoder_block_spec(config, **kwargs)

    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage)

    hf_config = _load_hf_config(args.hf_checkpoint)
    text_config = _get_text_config(hf_config)

    if not hasattr(text_config, "layer_types"):
        interval = getattr(text_config, "full_attention_interval", 4)
        n = text_config.num_hidden_layers
        text_config.layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(n)
        ]

    for layer_id in range(num_layers_to_build):
        if text_config.layer_types[layer_id + offset] == "linear_attention":
            layer_specs = copy.deepcopy(transformer_layer_spec.layer_specs[layer_id])
            layer_specs.submodules.self_attention = ModuleSpec(
                module=Attention,
                params={"args": args},
            )
            transformer_layer_spec.layer_specs[layer_id] = layer_specs

    return transformer_layer_spec
```

可以拆成三步。

第一步：先让 Megatron 构造标准 GPT decoder block spec。

```text
get_gpt_decoder_block_spec(config, use_transformer_engine=True)
```

由于本例有 `num_experts=256` 且 `moe_layer_freq` 全是 1，Megatron 在生成 layer_specs 时会把每层的 `mlp` 生成成 MoE spec，也就是最终会 build `MoELayer`。

第二步：按 PP/VPP 切出本 rank 的 local layers。

`get_gpt_decoder_block_spec(...)` 返回的是 `TransformerBlockSubmodules(layer_specs=local_layer_specs, layer_norm=...)`，里面的 `layer_specs` 已经只包含当前 PP/VPP rank 需要构建的层。

本例脚本 `PP=1` 且未开 VPP，所以 local layers 就是 40 层。若 `PP>1` 或 `VPP>1`，这里每个 rank/chunk 只会看到自己那段 layer specs。

第三步：对 linear attention 层替换 self_attention。

Qwen3.5 的 HF config 里会提供或推导 `layer_types`：

```text
full_attention
linear_attention
```

对 `linear_attention` 层，slime 深拷贝当前 layer spec，然后只替换：

```python
layer_specs.submodules.self_attention = ModuleSpec(
    module=Attention,
    params={"args": args},
)
```

这意味着该层的其它部分不变：

- layernorm 仍来自 Megatron spec。
- MLP/MoE 仍来自 Megatron spec。
- residual / bda 等 TransformerLayer 框架仍来自 Megatron spec。
- 只把 self_attention 子模块替换成 Qwen3.5 插件实现。

## 6. Qwen3.5 最终模型树长什么样

在本例配置下，单个 PP/VPP chunk 的模型树大致是：

```text
GPTModel
├── embedding
│   └── LanguageModelEmbedding
│       └── VocabParallelEmbedding
│
├── rotary_pos_emb
│   └── RotaryEmbedding(rotary_base=10000000, rotary_percent=0.25)
│
├── decoder
│   └── TransformerBlock
│       ├── TransformerLayer 1
│       │   ├── self_attention
│       │   │   ├── full attention 层: Megatron/TE SelfAttention
│       │   │   └── linear attention 层: slime_plugins.models.qwen3_5.Attention
│       │   └── mlp
│       │       └── MoELayer
│       │           ├── TopKRouter(num_experts=256, topk=8)
│       │           ├── token_dispatcher=MoEAlltoAllTokenDispatcher
│       │           ├── experts=GroupedMLP / TE expert implementation
│       │           └── shared_experts=SharedExpertMLP
│       ├── ...
│       └── TransformerLayer 40
│
└── output_layer
    └── ColumnParallelLinear(hidden_size -> vocab_size)
```

注意这个树不是所有 rank 都完整持有：

- TP 会切 attention / output / embedding 等 tensor parallel 参数。
- EP 会切 MoE experts。本例 `EP=8`，每个 EP rank 持有 32 个 local experts。
- PP/VPP 会切层。本例 `PP=1`，所以每个 rank 的 model chunk 都有 40 层；如果 PP>1，则每个 PP rank 只持有一部分层。
- `pre_process=False` 的 PP stage 不建 embedding。
- `post_process=False` 的 PP stage 不建 output layer。

## 7. full attention 层和 linear attention 层的区别

`get_qwen3_5_spec` 不会把所有 attention 都替换掉。它按 HF `layer_types` 判断：

```text
full_attention:
    保留 Megatron 原本 get_gpt_decoder_block_spec 生成的 self_attention。

linear_attention:
    替换成 slime_plugins.models.qwen3_5.Attention。
```

被替换的 `Attention` 继承自 `HuggingfaceAttention`，内部关键是：

```text
Attention
└── Qwen3_5GatedDeltaNet
    ├── ShortConvolution
    ├── in_proj_qkv
    ├── in_proj_z
    ├── in_proj_b
    ├── in_proj_a
    ├── FusedRMSNormGated
    └── out_proj
```

forward 时：

```text
hidden_states
-> input_layernorm
-> Qwen3_5GatedDeltaNet(..., cu_seqlens=packed_seq_params.cu_seqlens_q)
-> output
```

所以 Qwen3.5 插件的主要价值是：让 Megatron 的 TransformerLayer 能在部分层使用 Qwen3.5 的 GatedDeltaNet / linear attention 实现，同时继续复用 Megatron 的外层 block、MoE、并行和训练调度。

## 8. MoE 部分是谁构建的

Qwen3.5 的 `get_qwen3_5_spec` 没有手写 `MoELayer`。MoE 层是 Megatron 的 `get_gpt_decoder_block_spec(...)` 根据 `TransformerConfig` 自动生成的。

逻辑来自 Megatron：

```text
get_gpt_decoder_block_spec(...)
└── get_gpt_decoder_layer_specs(...)
    ├── 根据 config.moe_layer_freq 决定每层是 dense MLP 还是 MoE
    ├── 根据 config.num_moe_experts 判断 MoE 是否真的启用
    └── 对 MoE 层生成 mlp=ModuleSpec(module=MoELayer, ...)
```

本例：

```text
num_moe_experts = 256
moe_layer_freq = [1] * 40
```

所以 40 层的 `mlp` 都会是 MoE。

最终每个 `MoELayer` 初始化时会：

```text
1. 创建 TopKRouter
2. 根据 moe_token_dispatcher_type 创建 token dispatcher
   - 本例 alltoall -> MoEAlltoAllTokenDispatcher
3. 根据 EP rank 计算 local experts
   - num_local_experts = num_moe_experts / ep_size = 256 / 8 = 32
4. build local experts
5. 如果有 shared expert，build shared_experts
```

也就是说：

```text
Qwen3.5 spec:
    负责替换 linear attention 子模块。

Megatron decoder block spec:
    负责根据 MoE 参数构建 MoELayer / router / experts。
```

这两个改动作用在同一个 `TransformerLayer` spec 上，但负责不同子树。

## 9. slime 为什么不直接用 HF config 自动建模型

Megatron 训练模型通常不是直接 `AutoModel.from_pretrained(...)`。原因是：

- Megatron 需要在构建时就知道 TP / PP / CP / EP / ETP。
- 参数 shape 和并行切分强相关，例如 vocab parallel、column/row parallel、expert parallel。
- MoE dispatcher、grouped GEMM、sequence parallel、recompute 等都是 Megatron build-time 决策。
- checkpoint 加载也依赖 Megatron 构建出的参数名和 sharding 结构。

所以 slime 的 Megatron backend 采用：

```text
HF checkpoint/config:
    提供 tokenizer、校验、特殊层信息、权重来源。

Megatron args:
    显式声明训练模型结构和并行策略。

--spec:
    补 Megatron 标准 GPT spec 覆盖不了的模型结构差异。
```

Qwen3.5 正是这种模式：大部分骨架由 Megatron 标准 GPT/MoE spec 构建，Qwen3.5 特有的 linear attention / GDN 由 slime plugin 插进去。

## 10. 和 Megatron 原生文档的对应关系

你在 Megatron-LM 文档里写的：

```text
pretrain_gpt.py
└── pretrain(...)
    └── setup_model_and_optimizer(...)
        └── get_model(model_provider_func, ...)
            └── model_provider(...)
                └── gpt_builder(...)
                    ├── core_transformer_config_from_args(args)
                    ├── 选择 transformer_layer_spec
                    └── GPTModel(...)
```

在 slime 中对应为：

```text
MegatronTrainRayActor.init(...)
└── initialize_model_and_optimizer(...)
    └── setup_model_and_optimizer(...)
        └── get_model(get_model_provider_func(args, role), ...)
            └── slime.backends.megatron_utils.model_provider.model_provider(...)
                ├── core_transformer_config_from_args(args)
                ├── import_module(args.spec)
                ├── get_qwen3_5_spec(args, config, vp_stage)
                └── GPTModel(...)
```

也就是：

```text
Megatron 原生:
    gpt_builder 负责选 spec。

slime:
    slime model_provider 负责选 spec。

共同点:
    最后都把 transformer_layer_spec 交给 Megatron GPTModel。
```

## 11. 一句话总结

Qwen3.5 MoE 在 slime 里不是由 slime 从零实现一个完整模型，而是由 slime 的 model provider 调用 Megatron `GPTModel` 构建主模型树。`--spec get_qwen3_5_spec` 是一个“修改 Megatron layer spec 的钩子”：它先让 Megatron 按 Qwen3.5 的 MoE 参数生成 40 层 decoder block spec，再把 HF config 标记为 `linear_attention` 的层替换成 Qwen3.5 插件 attention。最终 Megatron 继续负责 embedding、TransformerBlock、MoELayer、router、expert dispatch、output layer、并行 wrapper 和训练执行。
