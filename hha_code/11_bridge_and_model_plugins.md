# Slime 里的 models / mbridge / megatron_bridge 怎么区分

这篇只解决一个容易绕晕的问题：

```text
slime_plugins/models
slime_plugins/mbridge
slime_plugins/megatron_bridge
mbridge
megatron.bridge
tools/convert_hf_to_torch_dist.py
```

它们不是同一层东西。尤其是 `mbridge` 和 `Megatron-Bridge` 有历史关系，但在当前代码里是两个 Python import namespace。

## 1. 先记住一句话

```text
slime_plugins/models
  负责“Megatron 模型长什么样、forward 怎么跑”。

slime_plugins/mbridge
  负责“旧 mbridge 体系下，HF 权重名/张量怎么映射到 Megatron 参数”。

slime_plugins/megatron_bridge
  负责“新 Megatron-Bridge / megatron.bridge 体系下的自定义 bridge 注册”。
```

所以模型结构和权重转换是两件事：

```text
模型结构:
  model_provider_func -> GPTModel / custom model -> layer spec -> forward

权重映射:
  HF 参数名 / 张量布局 <-> Megatron 参数名 / 张量布局
```

## 2. mbridge 和 Megatron-Bridge 的关系

历史上：

```text
mbridge  ->  Megatron-Bridge
旧前身       新官方延续方向
```

但在 Python 里它们是两个包：

```python
import mbridge
import megatron.bridge
```

当前环境里也能同时看到两个包：

```text
mbridge
  package name: mbridge
  home-page: https://github.com/ISEEKYAN/mbridge

megatron-bridge
  package name: megatron-bridge
  import namespace: megatron.bridge
  home-page: https://github.com/NVIDIA-NeMo/Megatron-Bridge
```

因此不要按名字猜。要看源码里 import 的到底是谁：

```python
from mbridge import AutoBridge
```

这是旧 `mbridge`。

```python
from megatron.bridge import AutoBridge
```

这是新 `Megatron-Bridge`。

## 3. 三个 slime 插件目录分别干什么

### 3.1 `slime_plugins/models`

这是模型结构层。它会被 `--spec` 接进 Megatron 的 model provider。

典型入口：

```bash
--spec "slime_plugins.models.glm5.glm5" "get_glm5_spec"
--spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
```

调用链在 `slime/backends/megatron_utils/model_provider.py`：

```python
config = core_transformer_config_from_args(args)

if args.spec is not None:
    transformer_layer_spec = import_module(args.spec)
    if callable(transformer_layer_spec):
        result = transformer_layer_spec(args, config, vp_stage)
        transformer_layer_spec = result

model = GPTModel(
    config=config,
    transformer_layer_spec=transformer_layer_spec,
    ...
)
```

也就是说，`slime_plugins/models` 里的函数不是“权重转换器”，而是“告诉 Megatron 这一层应该用什么模块”的结构插件。

比如：

```text
Qwen3.5:
  slime_plugins/models/qwen3_5.py
  -> get_qwen3_5_spec
  -> 把 HF config 标记为 linear_attention 的层替换成 Qwen3.5 Attention / GatedDeltaNet

GLM5.2:
  slime_plugins/models/glm5/glm5.py
  -> get_glm5_spec
  -> 把每层 self_attention 替换成 DSAMLASelfAttention
  -> 实现 DSA indexer / SparseMLA / cross-layer top-k sharing
```

### 3.2 `slime_plugins/mbridge`

这是旧 `mbridge` 的插件目录。

入口文件：

```python
import slime_plugins.mbridge
from mbridge import AutoBridge
```

`slime_plugins/mbridge/*.py` 里用的是：

```python
from mbridge.core import register_model
```

它的职责是注册旧 mbridge 的模型权重映射。比如：

```python
@register_model(["qwen3_5", "qwen3_5_moe"])
class Qwen3_5Bridge(Qwen2MoEBridge):
    ...
```

或者 GLM5.2：

```python
@register_model(["deepseek_v32", "glm_moe_dsa"])
class DeepseekV32Bridge(DeepseekV3Bridge):
    ...
```

这里的重点是“权重映射”，不是 forward。

### 3.3 `slime_plugins/megatron_bridge`

这是新 `Megatron-Bridge` 的插件目录。

入口代码常见于 bridge mode：

```python
from megatron.bridge import AutoBridge
import slime_plugins.megatron_bridge
```

这个目录里目前主要是：

```text
slime_plugins/megatron_bridge/glm4v_moe.py
```

它用的是新 Megatron-Bridge 的注册方式：

```python
@MegatronModelBridge.register_bridge(...)
class Glm4vMoeBridge(MegatronModelBridge):
    ...
```

所以它是新 `megatron.bridge` 体系的扩展点，不是旧 `mbridge`。

## 4. HF -> torch_dist 转换到底会不会构建模型

会。

`tools/convert_hf_to_torch_dist.py` 不是直接把 HF 文件改名。它先构建 Megatron 模型，再把 HF 权重 load 进去，最后用 Megatron checkpoint 保存。

核心流程：

```python
model = get_model(get_model_provider_func(args), ModelType.encoder_or_decoder, wrap_with_ddp=False)

bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
bridge.load_weights(model, hf_model_path, memory_efficient=True)

save_checkpoint(1, model, None, None, 0)
```

注意这个脚本顶部是：

```python
import slime_plugins.mbridge
from mbridge import AutoBridge
```

所以 **HF -> torch_dist 加载/映射 HF 权重时，用的是旧 `mbridge`**。

但是它构建 Megatron 模型时仍然走：

```python
get_model_provider_func(args)
```

如果 `MODEL_ARGS` 里有 `--spec`，就会导入 `slime_plugins/models`。

因此 HF -> torch_dist 的真实流程是：

```text
1. 根据 MODEL_ARGS 构建 Megatron 模型结构
   get_model_provider_func
   -> args.spec
   -> slime_plugins/models/...
   -> GPTModel(...)

2. 用旧 mbridge 加载 HF 权重
   import slime_plugins.mbridge
   from mbridge import AutoBridge
   bridge.load_weights(model, hf_checkpoint)

3. 保存 Megatron torch_dist checkpoint
   save_checkpoint(...)
```

## 5. GLM5.2 的实际路径

GLM5.2 转换文档里是：

```bash
source scripts/models/glm5.2-744B-A40B.sh

PYTHONPATH=/root/Megatron-LM/ torchrun \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint $BASE_DIR/GLM-5.2/ \
   --save $BASE_DIR/GLM-5.2_torch_dist/
```

`scripts/models/glm5.2-744B-A40B.sh` 里包含：

```bash
--spec "slime_plugins.models.glm5.glm5" "get_glm5_spec"
```

所以转换 GLM5.2 时，模型结构一定会走：

```text
slime_plugins/models/glm5/glm5.py
  -> get_glm5_spec
  -> DSAMLASelfAttention
  -> DSA indexer / SparseMLA / cross-layer top-k sharing
```

GLM5.2 的权重映射走：

```text
slime_plugins/mbridge/deepseek_v32.py
```

因为 HF config 的 `model_type` 是：

```text
glm_moe_dsa
```

旧 mbridge 注册里有：

```python
@register_model(["deepseek_v32", "glm_moe_dsa"])
class DeepseekV32Bridge(DeepseekV3Bridge):
    ...
```

这句话只表示：GLM5.2 的 HF 参数名/张量布局，可以基于 DeepSeek-V3.2 的 mbridge 映射来处理，并补充 DSA indexer 权重映射和 rope 半区重排。

它不表示 GLM5.2 的训练模型结构直接等于 DeepSeek-V3.2。

GLM5.2 的完整转换链是：

```text
HF GLM5.2 BF16 checkpoint
  |
  | tools/convert_hf_to_torch_dist.py
  |
  | 1. slime_plugins.models.glm5.glm5:get_glm5_spec
  |    构建 GLM5.2 的 Megatron 模型结构
  |
  | 2. slime_plugins.mbridge.deepseek_v32:DeepseekV32Bridge
  |    用旧 mbridge 把 HF 权重映射进这个模型
  |
  | 3. Megatron save_checkpoint
  v
Megatron torch_dist checkpoint
```

## 6. GLM5.2 为什么不能只靠 mbridge

因为 `mbridge` 只负责参数怎么对上，不负责 forward 怎么跑。

GLM5.2 的结构差异包括：

```text
DSA attention
SparseMLA
indexer projections:
  wq_b
  wk
  k_norm
  weights_proj
cross-layer top-k sharing
skip layer 不保存/不运行 indexer 权重
allgather-CP layout 下 index key/value gather
```

这些都在 `slime_plugins/models/glm5/glm5.py` 里实现。

`slime_plugins/mbridge/deepseek_v32.py` 只做类似下面的事情：

```python
"self_attention.wq_b.weight"
  -> "model.layers.{layer_number}.self_attn.indexer.wq_b.weight"

"self_attention.wk.weight"
  -> "model.layers.{layer_number}.self_attn.indexer.wk.weight"

"self_attention.weights_proj.weight"
  -> "model.layers.{layer_number}.self_attn.indexer.weights_proj.weight"

"self_attention.k_norm.weight"
  -> "model.layers.{layer_number}.self_attn.indexer.k_norm.weight"
```

以及对部分 DSA 权重做 rope 维度重排。

所以：

```text
mbridge 知道“这个 HF 权重应该塞到 Megatron 哪个参数里”。
models 知道“Megatron 里这个参数属于什么模块，以及 forward 怎么使用它”。
```

## 7. Qwen3.5 的对照

Qwen3.5 也类似，只是结构差异不同。

模型结构：

```text
scripts/models/qwen3.5-35B-A3B.sh
  -> --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"

slime_plugins/models/qwen3_5.py
  -> get_qwen3_5_spec
  -> 按 HF config 的 layer_types
  -> 把 linear_attention 层替换成 Qwen3.5 Attention / GatedDeltaNet
```

权重映射：

```text
slime_plugins/mbridge/qwen3_5.py
  -> @register_model(["qwen3_5", "qwen3_5_moe"])
  -> 定义 HF 权重名和 Megatron 权重名怎么对应
```

所以 Qwen3.5 也是：

```text
slime_plugins/models/qwen3_5.py
  负责结构

slime_plugins/mbridge/qwen3_5.py
  负责旧 mbridge 权重映射
```

## 8. `--megatron-to-hf-mode bridge` 又是什么

这个参数名容易误导。

Slime 里默认是：

```bash
--megatron-to-hf-mode raw
```

如果显式设置：

```bash
--megatron-to-hf-mode bridge
```

runtime 的 model provider 会走：

```python
from megatron.bridge import AutoBridge
import slime_plugins.megatron_bridge

bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
provider = bridge.to_megatron_provider(load_weights=False)
return provider.provide
```

这才是新 `Megatron-Bridge` 的 provider 路径。

但是注意：`tools/convert_hf_to_torch_dist.py` 这个 HF -> torch_dist 脚本顶部仍然 import 的是旧 mbridge：

```python
from mbridge import AutoBridge
```

所以在当前 Slime 代码里，至少这个转换脚本的 HF 权重 load 部分仍是旧 mbridge。

## 9. 最后用一张总图记住

```text
HF checkpoint
  |
  | tools/convert_hf_to_torch_dist.py
  |
  |-- 构建 Megatron 模型:
  |     get_model_provider_func(args)
  |       -> args.spec
  |       -> slime_plugins/models/<model>.py
  |       -> GPTModel(...)
  |
  |-- 加载 HF 权重:
  |     from mbridge import AutoBridge
  |     import slime_plugins.mbridge
  |     bridge.load_weights(...)
  |
  v
Megatron torch_dist checkpoint


训练时:
  get_model(get_model_provider_func(args), ...)
    -> raw/spec 路径:
         slime_plugins/models/<model>.py
       或 bridge 路径:
         from megatron.bridge import AutoBridge
         import slime_plugins.megatron_bridge
```

最稳的判断方法：

```text
想看模型结构 / forward:
  看 slime_plugins/models

想看 HF->Megatron 权重名怎么映射:
  当前很多例子看 slime_plugins/mbridge

想看新 Megatron-Bridge provider / export / VLM bridge mode:
  看 slime_plugins/megatron_bridge 和 from megatron.bridge import AutoBridge
```

