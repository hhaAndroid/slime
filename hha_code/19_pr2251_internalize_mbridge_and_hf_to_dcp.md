# PR #2251：内置 mbridge 与 HF → Megatron → DCP 链路

> 本文分析 THUDM/slime PR [#2251](https://github.com/THUDM/slime/pull/2251)：
> `Internalize mbridge and remove megatron-bridge`。
>
> 对应合入提交：`f655e13d9b262748441e836983deaddfe4715e22`。
>
> 本文描述 PR #2251 合入后的架构。`hha_code/11_bridge_and_model_plugins.md`
> 记录的是 PR 合入前的历史结构，其中提到的 `slime_plugins/mbridge`、
> `slime_plugins/megatron_bridge` 和 `--megatron-to-hf-mode bridge` 已被删除。

## 1. 核心结论

PR #2251 不是一个普通 bugfix，而是一次模型接入与权重转换体系的收缩和重构：

```text
删除：
  NVIDIA Megatron-Bridge 的实验性集成
  ISEEKYAN/mbridge 外部依赖
  raw / bridge 两套并存的转换路径

保留并内置：
  slime 实际需要的 HF <-> Megatron 参数名映射
  Tensor 布局变换
  TP / ETP 分片逻辑
  Megatron 原生 torch_dist / DCP 保存能力
```

合入后的 HF → DCP 流程是：

```text
Hugging Face safetensors
        │
        │  slime 根据 MODEL_ARGS 构建分布式 Megatron 模型
        ▼
各 TP / PP / EP / ETP rank 上的本地 Megatron model shard
        │
        │  slime.hf_to_megatron 做参数映射、Tensor 变换和本地切分
        ▼
每个 rank 上已经加载正确权重的 Megatron model shard
        │
        │  megatron.training.checkpointing.save_checkpoint()
        ▼
Megatron torch_dist / DCP checkpoint
```

需要特别强调：

```text
mbridge 的核心价值不是“保存 DCP”。

DCP 的 metadata 生成、分片写入和后续 reshard 一直由 Megatron-Core 负责。
mbridge 主要解决的是：如何把 HF 权重语义正确地装入 Megatron 分片模型。
```

## 2. PR 为什么要做这次调整

PR message 给出的背景是：

1. slime 当前难以及时支持所有模型；
2. NVIDIA Megatron-Bridge 接入只完成了一部分，尚未达到项目对代码质量和可维护性的要求；
3. 独立的 ISEEKYAN/mbridge 项目正在停止维护，后续方向迁移到了 NVIDIA-NeMo/Megatron-Bridge；
4. slime 决定只内置自身真正依赖的能力，以获得更清晰的代码所有权；
5. 如果用户更看重广泛的模型覆盖，PR 作者建议尝试 `radixark/miles`。

这次改动的规模也说明它不只是依赖替换：

```text
119 个文件变化
约增加 1975 行
约删除 9452 行
```

项目选择从：

```text
依靠通用 Bridge 尽量覆盖更多模型
```

转向：

```text
明确维护有限模型白名单
+ 每个模型显式实现结构和双向权重映射
```

## 3. 两个 Bridge 必须分开理解

PR #2251 同时处理了两个名字相近、职责不同的项目。

### 3.1 ISEEKYAN/mbridge

Python namespace：

```python
from mbridge import AutoBridge
```

它在 slime 中主要承担：

```text
HF config 识别
模型类型注册
HF 参数名 <-> Megatron 参数名映射
Tensor 布局变换
TP / EP 等并行切分适配
HF <-> Megatron 双向转换
```

PR 后不再安装这个包。slime 把当前需要的部分重写到：

```text
slime/backends/megatron_utils/hf_to_megatron/
slime/backends/megatron_utils/megatron_to_hf/
```

这不是把整个 mbridge 项目原样 vendor 进仓库，而是内置一个更小、更直接、
只覆盖 slime 支持模型的实现。

### 3.2 NVIDIA-NeMo/Megatron-Bridge

Python namespace：

```python
from megatron.bridge import AutoBridge
```

它的范围更大，除了权重转换，还试图提供：

```text
HF config
  -> Megatron TransformerConfig
  -> model provider / layer spec
  -> 构建 Megatron 模型
  -> 加载和导出权重
```

旧代码中可以通过：

```python
bridge.to_megatron_provider(load_weights=False)
```

直接获取 model provider。

PR 后这套实验集成被彻底删除，没有被完整内置。少数仍需支持的模型，
例如 Qwen3.5-VL，改成了 slime 自己维护的 native model provider。

## 4. PR 前的整体结构

PR 前存在：

```bash
--megatron-to-hf-mode raw
--megatron-to-hf-mode bridge
```

这个分支渗透在多条关键路径中：

```text
模型构建：
  model_provider.py

HF checkpoint 加载：
  checkpoint.py

训练中 Megatron -> SGLang 权重同步：
  update_weight/hf_weight_iterator_*.py

保存 HF checkpoint：
  hf_checkpoint_saver.py

参数校验和启动行为：
  slime/utils/arguments.py
```

### 4.1 旧 HF → DCP 工具

旧版 `tools/convert_hf_to_torch_dist.py` 的核心流程是：

```python
model = get_model(
    get_model_provider_func(args),
    ModelType.encoder_or_decoder,
    wrap_with_ddp=False,
)

bridge = AutoBridge.from_pretrained(hf_model_path, trust_remote_code=True)
bridge.load_weights(model, hf_model_path, memory_efficient=True)

save_checkpoint(1, model, None, None, 0)
```

这里实际是三层职责：

```text
slime model provider/spec：
  构建 Megatron 模型结构

ISEEKYAN/mbridge：
  将 HF 权重转换并装入 Megatron 模型

Megatron-Core：
  将模型保存为 torch_dist / DCP
```

### 4.2 旧 Megatron-Bridge 模型构建路径

当使用 `--megatron-to-hf-mode bridge` 时，`model_provider.py` 还会走：

```python
bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
provider = bridge.to_megatron_provider(load_weights=False)
```

随后 slime 再把 TP、PP、EP、ETP、CP 等配置手动写入 provider。

这导致：

1. `raw` 与 `bridge` 两条路径长期并存；
2. model provider、checkpoint、权重同步和保存都有条件分支；
3. Bridge 版本需要与 slime 当前 Megatron commit 精确兼容；
4. 一些模型还依赖专门 fork，例如旧 Qwen3.5-VL 示例；
5. 问题可能跨越 slime、mbridge、Megatron-Bridge 和 Megatron-Core 多层代码。

## 5. PR 后的 HF → Megatron → DCP 流程

### 5.1 初始化 distributed 和并行组

入口工具：

```text
tools/convert_hf_to_torch_dist.py
```

工具首先初始化 NCCL：

```python
dist.init_process_group(
    backend="nccl",
    world_size=world_size,
    rank=global_rank,
    device_id=torch.device(f"cuda:{local_rank}"),
)
```

随后通过 slime/Megatron 初始化逻辑建立：

```text
TP group
PP group
DP group
EP group
ETP group
CP group
```

### 5.2 构建当前 rank 的 Megatron 模型分片

转换工具调用：

```python
model = get_model(
    get_model_provider_func(args),
    ModelType.encoder_or_decoder,
    wrap_with_ddp=False,
)
```

模型结构来自：

```text
scripts/models/*.sh 中的 MODEL_ARGS
slime/backends/megatron_utils/model_provider.py
slime_plugins/models/*.py 中的 model spec/provider
```

此时每个 rank 上只存在当前并行布局对应的模型部分：

```text
PP：当前 rank 只拥有自己的 layer range
EP：当前 rank 只拥有自己的 local experts
TP：参数对象已经是本地 shard shape
ETP：expert 参数已经带有相应并行属性
```

模型参数仍然是初始化值，但模型结构、local shape 和分片属性已经确定。

### 5.3 根据 HF model_type 选择 loader

入口：

```text
slime/backends/megatron_utils/hf_to_megatron/__init__.py
```

核心是一个显式白名单：

```python
_LOADERS = {
    "deepseek_v3": deepseek_hf_tensor,
    "deepseek_v32": deepseek_hf_tensor,
    "glm4": glm4_hf_tensor,
    "glm4_moe": glm4_moe_hf_tensor,
    "glm4_moe_lite": deepseek_hf_tensor,
    "glm_moe_dsa": deepseek_hf_tensor,
    "kimi_k2": deepseek_hf_tensor,
    "llama": qwen_hf_tensor,
    "mimo": mimo_hf_tensor,
    "minimax_m2": minimax_m2_hf_tensor,
    "qwen2": qwen_hf_tensor,
    "qwen2_moe": qwen_moe_hf_tensor,
    "qwen3": qwen_hf_tensor,
    "qwen3_5": qwen3_5_hf_tensor,
    "qwen3_5_moe": qwen3_5_hf_tensor,
    "qwen3_moe": qwen_moe_hf_tensor,
    "qwen3_next": qwen3_next_hf_tensor,
}
```

调用过程：

```python
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
get_hf_tensor = _LOADERS[config.model_type]
load_model_hf_weights(args, model, path, config, get_hf_tensor)
```

不在白名单内的 `model_type` 会明确报错，而不是尝试通过通用 Bridge 自动适配。

### 5.4 读取 HF safetensors

`SafetensorReader` 支持：

```text
model.safetensors.index.json 分片索引
单个或多个 *.safetensors 文件
按参数名懒加载 tensor
部分 block-FP8 权重通过 *_scale_inv 反量化到 BF16
```

每个 rank 遍历自己本地模型参数：

```python
for name, parameter in named_params_and_buffers(args, model):
    tensor = get_hf_tensor(name, reader, config)
    tensor = shard_mcore_tensor(name, _pad_vocab(args, name, tensor), parameter)
    parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
```

### 5.5 参数名映射和 Tensor 语义转换

这是 HF → Megatron 最核心、也最容易出错的步骤。

#### QKV 融合

HF 通常分别保存：

```text
q_proj.weight
k_proj.weight
v_proj.weight
```

Megatron 可能保存：

```text
self_attention.linear_qkv.weight
```

对于 GQA，Megatron fused QKV 通常按 query group 组织，不能简单执行：

```python
torch.cat([q, k, v], dim=0)
```

loader 需要根据：

```text
num_attention_heads
num_key_value_heads
head_dim
```

先 reshape 到 group 结构，再按 Megatron 要求重新排列 Q/K/V。

#### Gate/Up 融合

HF：

```text
gate_proj.weight
up_proj.weight
```

Megatron：

```text
mlp.linear_fc1.weight
```

loader 需要将它们融合为：

```text
[gate; up]
```

TP 切分时也必须分别切 gate 和 up 后重新拼接，不能对整个 fused tensor
做无语义的平均切分。

#### MoE 和模型特有转换

还需要处理：

```text
per-expert <-> fused-experts 表示
global expert id <-> local EP expert id
shared expert 和 shared expert gate
MTP layer 的独立命名与布局
DeepSeek/GLM DSA 权重布局
Qwen3 Next linear attention 参数
Qwen3.5 的 model.language_model 嵌套
tied embedding/output weight
vocabulary padding
模型特有的 RoPE 或矩阵半区重排
```

这些逻辑才是过去 mbridge 的主要价值。

### 5.6 TP / ETP 本地切分

通用切分入口：

```text
slime/backends/megatron_utils/hf_to_megatron/common.py
  shard_mcore_tensor()
```

它读取 Megatron 参数对象上的属性：

```python
parameter.tensor_model_parallel
parameter.parallel_mode
parameter.partition_dim
parameter.partition_stride
```

普通参数使用 TP group；expert 参数使用 ETP group：

```python
if ".experts." in name:
    parallel_size = mpu.get_expert_tensor_parallel_world_size()
    parallel_rank = mpu.get_expert_tensor_parallel_rank()
else:
    parallel_size = mpu.get_tensor_model_parallel_world_size()
    parallel_rank = mpu.get_tensor_model_parallel_rank()
```

PP 和 EP 的选择更多由本地模型结构天然决定：

```text
当前 PP rank 没有的 layer，不会出现在本地参数遍历中；
当前 EP rank 没有的 expert，也不会出现在本地模型中；
loader 根据本地参数对应的 global layer/expert id 读取 HF tensor。
```

### 5.7 shape 校验并写入模型

切分后会做严格 shape 检查：

```python
if tensor.shape != parameter.shape:
    raise ValueError(...)
```

随后复制到目标参数：

```python
parameter.copy_(
    tensor.to(device=parameter.device, dtype=parameter.dtype)
)
```

需要注意：shape 正确仍不代表语义一定正确。例如 QKV 排列错误时，
Tensor shape 和 DCP 结构都可能完全合法，但模型输出会出错。因此当前代码通过
HF → Megatron → HF round-trip 测试覆盖关键映射。

### 5.8 Megatron-Core 保存 DCP

所有 rank 完成权重加载后，工具直接调用：

```python
save_checkpoint(1, model, None, None, 0)
```

该函数来自：

```python
from megatron.training.checkpointing import save_checkpoint
```

Megatron-Core 负责：

```text
从模型生成 sharded_state_dict
计算每个 local tensor 的 global shape 和 global offset
生成 torch_dist/DCP metadata
让各 rank 写入对应的 .distcp shard
保存 common state
维护 checkpoint tracker
```

转换工具最后会将 iteration 1 改成 release checkpoint，并把 tracker 写为：

```text
release
```

因此输出大致为：

```text
output/
├── latest_checkpointed_iteration.txt
└── release/
    ├── common.pt
    ├── metadata.json 或等价 metadata
    ├── _0_0.distcp
    ├── _0_1.distcp
    └── ...
```

具体文件名可能随 Megatron/PyTorch DCP 版本变化，但职责边界不变。

## 6. 为什么 DCP 可以换并行布局加载

旧 `torch` checkpoint 通常以 `mp_rank_xxx` 为主要组织方式，保存和加载时往往要求
TP/PP 等布局一致。

`torch_dist` / DCP 保存的不只是“某 rank 的 Tensor 文件”，还记录：

```text
全局 Tensor shape
当前 local shard 的 global offset
各维度的 fragmentation
replicated / sharded 关系
参数 key 与 sharding metadata
```

因此可以：

```text
转换时：TP=8, PP=2, EP=16
                 │
                 │ 保存全局 sharding metadata
                 ▼
训练时：TP=4, PP=4, EP=32
```

加载时，Megatron 根据新模型的 `sharded_state_dict` 重新规划读取并完成 reshard。

这也解释了为什么 HF loader 不需要直接生成 DCP metadata：

```text
HF loader 只需要把当前模型参数填正确；
模型本身已经知道自己的分片语义；
Megatron-Core 再根据模型生成 DCP metadata。
```

## 7. mbridge 过去具体承担了什么

当前流程看起来简单，是因为“从 HF 加载到 Megatron”被压缩成了一句话。
mbridge 曾经将这部分复杂度抽象成统一框架。

### 7.1 model_type 注册和自动选择

旧插件可以写成：

```python
@register_model("qwen3_next")
class Qwen3NextBridge(Qwen2MoEBridge):
    ...
```

调用方只需要使用 `AutoBridge`，不需要自己维护 loader 分发表。

### 7.2 HF config → Megatron config

mbridge 可以将常见字段映射为 Megatron 配置：

```text
num_hidden_layers    -> num_layers
num_key_value_heads  -> num_query_groups
intermediate_size    -> ffn_hidden_size
rms_norm_eps         -> layernorm_epsilon
num_experts          -> num_moe_experts
num_experts_per_tok  -> moe_router_topk
```

并补充模型特有选项，例如 MLA、QK norm、shared expert、MTP 和 attention 类型。

PR 后，slime 更倾向于将这些配置显式放在：

```text
scripts/models/*.sh
slime_plugins/models/*.py
```

### 7.3 参数名映射

mbridge 维护 Megatron 参数名与一个或多个 HF 参数名之间的关系，例如：

```text
Megatron linear_qkv
  <-> HF q_proj + k_proj + v_proj

Megatron linear_fc1
  <-> HF gate_proj + up_proj
```

它还处理 layer、expert、MTP 和多模态嵌套前缀。

### 7.4 Tensor 布局变换

mbridge 的 Bridge 类可以实现：

```python
_weight_to_mcore_format(...)
_weight_to_hf_format(...)
```

用于完成双向布局变换，而不只是参数改名。

### 7.5 并行与 memory-efficient 加载

mbridge 会结合本地 Megatron 参数形状和并行属性，将 HF tensor 分配到对应 rank，
并提供 memory-efficient 的加载编排，避免调用方自己组织完整转换流程。

### 7.6 双向转换复用

同一套 Bridge 可以同时服务：

```text
HF -> Megatron：初始化训练模型或生成 DCP
Megatron -> HF：同步给 SGLang 或保存 HF checkpoint
```

现在 slime 将两个方向拆成独立模块：

```text
hf_to_megatron/
megatron_to_hf/
```

好处是代码更直接；代价是两边的规则需要显式保持一致。

## 8. PR 后为什么不再需要 mbridge

不是因为 mbridge 的工作不重要，而是 slime 选择了不同的维护边界。

### 8.1 旧方案优势

```text
统一 AutoBridge API
模型族之间可以继承和复用映射
接入新模型更体系化
配置和双向转换由框架组织
潜在模型覆盖范围更大
```

### 8.2 旧方案成本

```text
外部项目生命周期不再由 slime 控制
需要与 slime 固定的 Megatron commit 保持兼容
很多模型仍需要 slime 自己继承、patch 和注册
raw / bridge 两条运行路径扩大测试矩阵
出错调用栈横跨多个仓库
slime 只使用框架的一部分，却需要跟随整个抽象演进
```

### 8.3 当前方案取舍

当前方案是：

```text
有限模型白名单
+ 直接 mapping 函数
+ slime 自己的 model spec/provider
+ Megatron 原生 DCP save/load
```

它的收益是：

```text
代码所有权清晰
依赖减少
调用链缩短
不再维护 raw / bridge 分支
可以针对当前 Megatron 版本直接调整
```

代价是：

```text
不再自动覆盖未知 HF 模型
每个新模型都需要显式接入
需要维护 HF -> Megatron 和 Megatron -> HF 两个方向
模型支持范围由 slime 白名单决定
```

## 9. PR 后的两种 HF 加载方式

### 9.1 离线转换为 DCP

推荐流程：

```bash
source scripts/models/qwen3-4B.sh

PYTHONPATH=/root/Megatron-LM \
python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist
```

大模型可以通过 `torchrun` 多卡/多机转换，并指定合适的 TP、PP、EP、ETP。

这条路径的结果是一个可被不同训练并行布局 reshard 的 `torch_dist` checkpoint。

### 9.2 训练启动时直接加载 HF

训练也可以把支持的 HF 目录直接传给 `--load`。

slime 会判断：

```text
目录存在
+ 包含 config.json
+ config.model_type 位于 _LOADERS
```

满足条件时直接执行：

```text
构建当前训练拓扑的 Megatron model shard
  -> load_hf_weights()
  -> 开始训练
```

这条路径不会预先在磁盘生成 DCP；第一次训练 checkpoint save 时，才会由
Megatron-Core 写出正常的 `torch_dist` checkpoint。

离线转换仍然适合：

```text
大模型初始化
多次复用同一个初始 checkpoint
OPD teacher 要求 Megatron checkpoint
希望提前验证转换结果
希望训练启动时避免重复读取完整 HF checkpoint
```

## 10. 模型支持变化和 breaking changes

PR 删除了以下依赖 Bridge 的实现或示例：

```text
Gemma4 模型、脚本、文档和测试
GPT-OSS 模型、脚本和预处理工具
GLM-4.6V 自定义 Megatron-Bridge
旧 Qwen3-VL Megatron-Bridge 示例
部分多模态/NPU 示例
gemma4 loss-mask 类型
bridge 版 checkpoint 导入、导出和在线权重同步
```

同时新增了 slime 原生的 Qwen3.5-VL provider：

```text
slime_plugins/models/qwen3_5_vl.py
slime_plugins/models/qwen3_5_vl_utils.py
scripts/models/qwen3.5-35B-A3B-vl.sh
```

它采用：

```text
Megatron language model
+ Transformers vision encoder
+ slime 自己的 MRoPE 和 multimodal embedding 注入
```

这是特定模型的原生实现，不是通用 Megatron-Bridge 的完整替代。

如果旧脚本包含：

```bash
--megatron-to-hf-mode bridge
```

升级后不能只删除参数。还需要确认：

1. 当前模型有可用的 Megatron model spec/provider；
2. `config.model_type` 已注册到 `_LOADERS`；
3. HF → Megatron 映射覆盖所有参数；
4. Megatron → HF 映射可用于 SGLang 权重同步和 HF 导出；
5. TP/PP/EP/ETP 下 global layer/expert 编号正确；
6. 有 round-trip 或 train/rollout logprob 对齐验证。

## 11. 新模型接入需要实现什么

在 PR #2251 后，新模型支持至少分为三部分。

### 11.1 模型结构

如果 Megatron 原生 GPTModel/spec 足够，只需配置 `MODEL_ARGS`。

如果存在特殊 attention、MLP、MTP、VLM 或 forward 行为，则需要：

```text
slime_plugins/models/<model>.py
```

提供 model spec 或 custom provider。

### 11.2 HF → Megatron

需要在：

```text
slime/backends/megatron_utils/hf_to_megatron/
```

实现：

```text
Megatron local parameter name
  -> HF tensor name(s)
  -> merge/reorder/transform
```

并把 `config.model_type` 注册到 `_LOADERS`。

### 11.3 Megatron → HF/SGLang

需要在：

```text
slime/backends/megatron_utils/megatron_to_hf/
```

实现反向映射，用于：

```text
训练中同步权重给 SGLang
保存 HF safetensors
离线 Megatron -> HF 转换
```

### 11.4 验证

至少应验证：

```text
HF -> Megatron -> HF round-trip
参数 key 完整性
参数 shape 完整性
TP/PP/EP/ETP 多布局 DCP load
初始 train/rollout logprob 对齐
必要时验证 MTP、VLM、shared expert 和量化 checkpoint
```

## 12. 最终职责边界

PR #2251 合入后，可以用下面四句话记住当前架构：

```text
slime model spec/provider
  决定 Megatron 模型长什么样、每个 rank 有哪些参数。

slime hf_to_megatron
  决定 HF 参数如何映射、变换并切到当前 rank。

Megatron-Core save_checkpoint
  根据模型 sharding metadata 保存 torch_dist / DCP。

slime megatron_to_hf
  把训练参数转换回 HF 命名和布局，供 SGLang 同步与 HF 导出使用。
```

最重要的认识是：

```text
“保存 DCP”并不是 HF -> Megatron 转换中最难的部分。

最难的是保证每一个 HF Tensor 在各种模型结构和 TP/PP/EP/ETP 布局下，
都被放到了正确的 Megatron 参数位置，并且内部语义排列完全一致。

mbridge 过去把这部分复杂度组织成通用 Bridge 框架；
PR #2251 后，slime 选择只为明确支持的模型直接维护这些映射。
```
