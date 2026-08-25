# Kimi K3 深度解读：架构、训练技术与基础设施

> 论文：[Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)，v1，2026-07-27。
>
> 本文只讨论模型架构、预训练/后训练技术与 infra；按要求不整理评测分数。
>
> 阅读约定：文中的“论文明确说明”与“我的判断”会尽量分开。后者是基于公开信息的工程解释，不代表 Kimi Team 的官方表述。

## 0. 先给结论

Kimi K3 最值得关注的不是某一个孤立模块，而是一条贯穿模型、训练和服务的统一设计原则：**把会随规模失控增长或高度波动的状态，改造成有界、静态或可暂停/迁移的状态。**

- 序列维度：用 3:1 的 KDA–MLA 混合注意力，把大部分层的 token mixing 从随上下文增长的 KV cache 改成固定大小的循环状态，同时周期性保留全局 softmax attention。
- 深度维度：用 Block Attention Residuals 让层可以选择性读取早期表示，而不是把所有历史层压进一条普通 residual stream；同时把保存/通信开销从按层数增长降到按少量 block 增长。
- 宽度维度：用 LatentMoE 把 routed expert 放进半宽 latent space，从而负担得起 896 个 routed experts、每 token 激活 16 个；再用归一化、有限幅激活和 Quantile Balancing 解决极稀疏 MoE 的数值稳定与负载问题。
- 训练系统：通过 MoonEP 把动态路由造成的不规则 token 负载，转成每个 EP rank 固定的计算量和静态 shape；通过统一 activation manager 把 recompute、FP8、CPU/远端 offload 变成可组合的 tensor storage policy。
- Agent RL：通过 partial rollout 把超长轨迹切跨多个 iteration；跨轮持久化的是 token/message history、RL bookkeeping 与 microVM 环境状态。MLA KV/KDA state 只在同一个 rollout phase、同一个 policy version 内通过 external pool 复用；rollout iteration 结束后 pool 整体释放，未完成轨迹在下一轮用新 policy 对历史重新 prefill。
- 在线服务：混合注意力带来的两类 cache 被统一管理；长会话通过 prefix affinity 保持 cache locality；请求成本横跨三个数量级的问题，通过分预算 admission control 隔离。

所以，K3 可以被概括为：

> **结构上扩展信息流，数值上限制极值，执行上消灭不规则 shape；逻辑与环境状态支持跨 iteration 恢复，模型派生 cache 则严格受 rollout phase 和 policy version 约束。**

这也是论文中“algorithm–system co-design”真正具体的含义。

---

## 1. 全景图：K3 到底由什么组成

### 1.1 主干规格

论文披露的核心配置如下（仅保留理解架构与 infra 所需的规格）：

| 项目 | Kimi K3 |
|---|---:|
| 总参数 | 2.78T（正文近似写作 2.8T） |
| 每 token 激活参数 | 104.2B |
| backbone 层数 | 93 |
| hidden size | 7,168 |
| vocabulary | 160K |
| attention heads | 96 |
| attention 组成 | 69 KDA + 24 Gated MLA |
| routed expert 数 | 896 / MoE layer |
| 每 token routed top-k | 16 |
| shared experts | 2 / MoE layer |
| latent MoE width | 3,584，即主干 hidden size 的 0.5 倍 |
| 每 expert FFN hidden | 3,072 |
| dense layer | 1 |
| MTP layer | 1 |
| 训练上下文上限 | 1M tokens |
| vision encoder | MoonViT-V2，401M，27 层，patch size 14，12 heads |

来源：[论文 Table 1，§2–3](https://arxiv.org/pdf/2607.24653#page=11)。

### 1.2 一条 token 的主要数据流

```text
文本 token ───────────────────────────────────────────────┐
                                                        │
图像/视频 → MoonViT-V2 → 2×2 pixel shuffle → MLP projector
                                                        │
                                                        ▼
                                              共享 embedding space
                                                        │
                  ┌─────────────────────────────────────┘
                  ▼
        [ KDA → Stable LatentMoE ] × 3
                  │
        [ Gated MLA → Stable LatentMoE ] × 1
                  │
          上述 4-layer pattern 重复
                  │
      额外一个末端 Gated MLA（保证最终全局交互）
                  │
                  ▼
               LM output

每一层的输入并非只来自上一层：
Block AttnRes 对 embedding、历史 block 输出和当前 block partial sum 做深度方向的选择性聚合。
```

93 层与 69/24 的关系是：23 组 `3 KDA + 1 MLA` 给出 92 层，再在末尾增加 1 个 MLA，共 69 个 KDA、24 个 MLA。[论文 §2.1](https://arxiv.org/pdf/2607.24653#page=4)

### 1.3 “三个信息流维度”是理解整篇论文的钥匙

| 维度 | 原始瓶颈 | K3 的机制 | 系统代价/配套 |
|---|---|---|---|
| sequence/token mixing | full attention 的长上下文计算与 KV cache | 3:1 KDA–MLA hybrid attention | FlashKDA、KDA Context Parallelism、双 cache 管理 |
| depth/layer mixing | 普通 residual 把所有历史层压进单一状态 | Block Attention Residuals | block cache、checkpoint、pipeline 增量通信、专用 kernel |
| width/channel mixing | 大 expert pool + 大 top-k 的权重/通信成本 | Stable LatentMoE | QB、MoonEP、latent projection/通信融合、专用 decode kernel |

K3 不是“线性注意力模型”，也不是“纯 MoE 扩容”。它保留 1/4 左右的全局 MLA 层、两个 full-width shared experts，并用 latent routed branch 扩大稀疏容量。这是一种明显的混合主义：昂贵路径负责保真，便宜路径负责规模。

---

## 2. 序列维度：Hybrid KDA–MLA

### 2.1 KDA 的直觉

单头 Kimi Delta Attention 维护循环状态：

$$
S_t = (I-\beta_t k_tk_t^\top)\,\mathrm{Diag}(\alpha_t)S_{t-1}
      + \beta_t k_tv_t^\top,
\qquad
\tilde{o}_t=S_t^\top q_t.
$$

可以把它拆成三件事：

1. `Diag(α_t)`：按 key channel 对旧状态做遗忘/保留，每个 channel 的保留率不同。
2. `I - β_t k_t k_t^T`：delta rule 的纠错项；在写入新 `(k_t, v_t)` 之前，先沿当前 key 方向消去旧映射中的误差。
3. `β_t k_t v_t^T`：把当前 key–value 关联写入状态。

这与普通 attention 的区别很本质：普通 attention 保存历史 token 的 K/V，查询时显式访问它们；KDA 把历史压缩进固定大小的矩阵状态 `S`。因此 decode state 不随序列长度线性增长，但历史被压缩，不能像 softmax attention 那样无损地逐 token 回看。

K3 采用 ShortConv + Swish 产生 Q/K/V，Q/K 再做 L2Norm；`β` 是标量写强度，`α` 是逐 channel 的保留率。输出端先做 per-head RMSNorm，再使用 input-dependent full-rank gate：

$$
y_t=W_o[\sigma(W_gx_t)\odot\mathrm{RMSNorm}(\tilde{o}_t)].
$$

相比 Kimi Linear 的低秩输出 gate，K3 让每个 token 可以更细粒度地控制读出的 channel。[论文 §2.1.1](https://arxiv.org/pdf/2607.24653#page=4)

### 2.2 为什么训练时仍能并行：chunkwise formulation

KDA 对 token 是递归的，直接逐 token 算会浪费 GPU。论文沿用 Kimi Linear 的 chunkwise 方法：

- chunk 之间传递循环状态，仍然串行；
- chunk 内把输出拆成 `inter-chunk` 与 `intra-chunk` 两部分；
- chunk 内的因果交互转为 triangular dense matrix multiplication，交给 Tensor Core；
- UT transform 把 delta-rule 的依赖改写成可并行形式。

所以它不是消除了递归，而是把递归的粒度从 token 提升到 chunk，并让 chunk 内有足够大的矩阵乘法。

#### 2.2.1 lower-bounded decay 是典型的算法—kernel 共设计

chunkwise 算法需要用累计 retention `Γ` 去缩放 key。若每步 `α∈(0,1)` 无下界，累计乘积可能极小，`1/Γ` 会溢出。Kimi Linear 为此把 chunk 再切成 16-token tile，并让对角 tile 走显式 position-pair 路径；这部分不是标准 dense GEMM，成为瓶颈。

K3 改成：

$$
g_t^h=g_{min}\,\sigma(e^{A_h}z_t^h)\in(g_{min},0),
\qquad \alpha_t^h=e^{g_t^h},
\qquad g_{min}=-5.
$$

于是单步 `α > e^-5`；16-token tile 的累计 log-decay 大于 `-80`，倒数小于 `e^80`，仍在 BF16 动态范围内。结果是：**对角和非对角 tile 都能走 dense Tensor Core GEMM，删除特殊 position-pair kernel。**

这项改动的重要性不只在数值稳定。它通过限制模型可学习 gate 的数值范围，换取了统一计算路径和更好的硬件效率。[论文 Fig.3 与 §2.1.1](https://arxiv.org/pdf/2607.24653#page=5)

### 2.3 为什么还要周期性插入 MLA

KDA 的固定状态是压缩记忆，不能完全替代全局内容寻址。K3 每个 block 使用 3 个 KDA 层后接 1 个 Gated MLA，并在 backbone 末尾再放一个 MLA，形成：

- KDA：位置敏感、recency-aware、长序列成本友好；
- MLA：不受循环压缩限制的全局 token-to-token 内容交互；
- MLA 的 latent KV 表示：相对传统 MHA/GQA 降低 KV cache。

K3 的 MLA 不使用任何显式位置编码（NoPE）。位置信息由相邻 KDA 层的递归 gate/decay 注入；MLA 只负责全局内容匹配。这也意味着扩展上下文时不用调整 RoPE base 或做 YaRN/interpolation。[论文 §2.1.2、§3.4](https://arxiv.org/pdf/2607.24653#page=5)

MLA 输出同样采用 full-rank sigmoid gate。训练时，为避免 flash attention 的有偏舍入误差，attention output 保持 FP32；因为 FP32 output tile 会加倍片上占用，kernel 又把它与 KV staging buffer 而不是 query tile overlap，以腾出 shared memory 做更深的 KV pipeline。这是另一个数值选择直接改变 kernel layout 的例子。

### 2.4 不能误读成“1M context 已经近似免费”

KDA 层的 recurrent state 固定大小，但 K3 仍有 24 个全局 MLA 层：

- MLA KV cache 仍随 sequence length 增长；
- MLA prefill 仍有全局 attention 的序列计算；
- KDA prefill 仍需要处理所有 token，只是计算/通信形式更易线性扩展；
- 混合模型必须在同一请求中同步维护 KDA state 与 MLA KV cache。

因此，KDA 显著改变了长上下文的常数、cache 组成和并行方式，但没有把整个模型变成严格意义上的端到端 O(n) full-context Transformer。论文后面大量 cache、CP 与调度设计，恰恰证明 1M context 仍是系统级问题。

---

## 3. 深度维度：Attention Residuals

### 3.1 普通 residual 的问题

标准 residual stream 可写成不断累加的单一状态。所有早期层的信息先被压进 `h_l`，后续层只能读取这一个混合结果。论文把这类比为时间维度上的 RNN 瓶颈：既然 Transformer 用 attention 让 token 选择历史 token，为什么不让 layer 选择历史 layer？

Full AttnRes 为第 `l` 层设置一个可学习、与 token 无关的 layer-specific pseudo-query `w_l`，对 embedding 和所有前序层输出做 attention。key 在计算权重前做 RMSNorm，避免幅值大的层仅凭 scale 支配 softmax。

这里需要注意：

- query 是“这一层想读什么深度”，不是当前 token 动态生成的 query；
- key/value 仍是每个 token 对应的历史层表示，因此不同 token 的权重会因 key 不同而变化；
- 它解决的是 depth mixing，不替代 token attention。

Full 版本算术量 `O(L²d)` 在不到 100 层时不算最大问题，真正困难的是保存所有历史层输出的 `O(Ld)` activation，以及 pipeline stage 间的通信。

### 3.2 Block AttnRes 如何降成本

K3 把层分块：block 内将已经完成的 layer output 累加成 partial sum；block 间只保留一个 block representation。某层的候选来源是：

- embedding；
- 所有已完成 block 的 representation；
- 若位于 block 内第二层及以后，再加当前 block 的 partial sum。

最终输出层聚合所有 block 表示。于是 memory/communication 从 `O(Ld)` 降到 `O(Nd)`，其中 `N` 是 block 数，而非总层数。

K3 把 layers 分成 8 个 layer blocks，目标 block size 为 12 层，最后一个 block 不满 12 层；再把 embedding 作为一个独立 source，因此最终共有 9 个 block-level sources。专用 online-softmax 可合并并行的 inter-block 结果和顺序产生的 intra-block partial sum。[论文 §2.2](https://arxiv.org/pdf/2607.24653#page=6)

### 3.3 AttnRes 的工程含义

AttnRes 不是一个可以只改 model.py 的小模块，它会影响：

- training activation 生命周期：历史 block 表示必须跨多个 layer 存活；
- PP 通信：后续 stage 需要前面生成的 block cache；
- checkpoint/recompute：需要确保保存项不会重新涨回逐层规模；
- prefill：若每个 TP rank 都 materialize block 表示，会产生重复显存；
- decode：inter-block 读 cache 与 intra-block 更新的调度不同。

K3 的配套实现是：block representation 仅在边界生成一次；AttnRes 整体 checkpoint；PP 只增量发送新 block，micro-batch 结束即释放；prefill 用 sequence-parallel activation 避免 TP rank 重复 materialize；decode 把 inter-block kernel 放 side stream，把 intra-block merge + RMSNorm 融入此前的 TP all-reduce。

---

## 4. 宽度维度：Stable LatentMoE

### 4.1 为什么是 latent expert

普通 MoE 把完整 `d` 维 token 发给每个被选 expert。若同时增加 expert 总数和 top-k：

- dispatch/combine 通信随 `d × top-k` 增长；
- 每 token 需要读取更多 expert weights；
- routed branch 的算力也快速增长。

LatentMoE 把公共能力和专门能力拆开：

- 两个 shared experts 在完整 `d=7168` 宽度处理 `x`；
- routed branch 先做 `W_down: 7168 → 3584`；
- 896 个 routed experts 都在 `ℓ=3584` latent space 工作，每 token 选 16 个；
- 聚合后做 RMSNorm，再用 `W_up: 3584 → 7168` 回到主干宽度。

可写成：

$$
u=\sum_{i\in T_k(x)}p_iE_i^{routed}(W_\downarrow x),
$$

$$
y=\sum_{j=1}^{2}E_j^{shared}(x)+W_\uparrow\mathrm{RMSNorm}(u).
$$

这里的“sparsity 56”是 `896/16=56`，即每 token 只走 routed expert pool 的 1/56；不能理解成整个模型每 token 只激活 1/56 参数，因为还有 shared experts、attention、projection 等始终激活的部分。

### 4.2 极端稀疏下的两个失败模式

论文指出两个主要问题：

1. routed path 形成接近四个连续矩阵乘法的链（down projection、GLU 多分支、up projection），在 2.8T 规模下条件数恶化并出现内部 activation explosion；
2. 近千 experts 下，旧式 auxiliary-loss-free bias 的固定步长更新在“响应慢”和“来回振荡”之间难以取舍，部分 expert 过热、部分濒死。

Stable LatentMoE 用三个部件共同处理：Normalized LatentMoE、SiTU-GLU、Quantile Balancing。

### 4.3 Normalized LatentMoE

被选 experts 及其 router weights 不同，聚合后的 `u` scale 会波动。如果直接接 `W_up`，这类变化会被放大，并与 full-width shared branch 相加。

K3 在聚合后、up projection 前插入 RMSNorm。这个位置很关键：它不是普通 block pre-norm，而是专门隔离 routed aggregation 的 scale variation，使 latent branch 回到 full-width 前有可控幅值。

### 4.4 SiTU-GLU：用软上限替代事后 clipping

SwiGLU 的 gate 与 up 两个乘法因子都无界，大坐标同时出现时容易制造 activation outlier，并放大低精度溢出风险。K3 定义：

$$
\mathrm{softcap}(x,\beta)=\beta\tanh(x/\beta),
$$

$$
\mathrm{SiTU\mbox{-}GLU}(x)=
[\beta_1\tanh(W_gx/\beta_1)\odot\sigma(W_gx)]
\odot[\beta_2\tanh(W_ux/\beta_2)].
$$

K3 使用 `β1=4, β2=25`。原点附近，tanh 近似线性，行为接近 SwiGLU；大幅值时两个分支都平滑饱和，标量输出绝对值有 `β1β2=100` 的上界。相比 hard clamp，它保持光滑梯度，也更适合低精度训练。[论文 §2.3.2](https://arxiv.org/pdf/2607.24653#page=7)

### 4.5 Quantile Balancing：router 学语义，bias 管容量

K3 采用 auxiliary-loss-free routing：

$$
s_i=\sigma(W_rx_i),\qquad
T_i=\operatorname{argtopk}(s_i+b).
$$

实际 mixture weight `p` 只由原始 `s` 归一化，不包含 bias `b`。因此：

- `s` 通过梯度学习 token–expert 适配；
- `b` 只改变 dispatch 决策，负责容量均衡；
- balancing 不通过辅助 loss 扭曲主训练目标。

旧方法根据 expert 当前 load 高低，用固定步长对 `b_j` 做 sign update。QB 则一次性解出让 expert 达到目标负载所需的分位点：

1. 对每 token 在带旧 bias 的 router score 上取 Top-(k+1)；前 k 个是真实 route，第 k+1 个是进入 Top-k 所需的 cutoff `α_i`。
2. 对 expert `j`，观察所有 token 的 margin `s_{i,j}-α_i`。
3. 全局 batch 有 `m` tokens、`n` experts、top-k 为 `k`，目标是每 expert 接收 `q=mk/n` 个 token。
4. 选择 margin 的 `(1-k/n)` 分位点并取负，作为新 bias；再对所有 bias 去均值，因为共同平移不改变 Top-k。
5. 新 bias 只在下一 training step 生效，避免用当前 batch 推导的 bias 反过来改变当前 batch，保持因果。

从优化角度，固定步长 sign update 只取“目标 load − 实际 load”的方向；QB 直接跳到同一个分段线性对偶子问题的 coordinate minimizer，因此没有 learning-rate-like 超参，能在近千 expert 规模迅速平衡。

#### 4.5.1 全局分位数如何落地

精确聚合数百万 token × 896 experts 的 margin 不现实。论文实现为每 expert 一个 histogram：

- 每个 rank、每个 accumulation micro-batch 只在本地 scatter-add bin counts；
- training step 末尾，对 `n × B` 个整数计数做一次 all-reduce；
- 从合并后的累计计数读出全局 quantile，并在命中 bin 内线性插值；
- `B=1000` 时误差不超过动态 bin width，论文称通信成本低于逐 micro-batch 交换 raw margins 的 1%。

这不是“平均每卡 quantile”，而是先把可加的直方图计数合并，再求 pooled global batch quantile；两者统计意义不同。[论文 §2.3.3 与 Appendix D](https://arxiv.org/pdf/2607.24653#page=44)

### 4.6 QB 与 MoonEP 解决的不是同一层问题

这是论文最容易混淆的一点：

- **QB 是学习/路由层面的均衡**：让 experts 长期接近目标 token load，防止过热或濒死，改善训练质量并降低常态 imbalance。
- **MoonEP 是执行/放置层面的均衡**：即使当前 micro-batch、当前 layer 的路由仍然任意倾斜，也通过动态复制 expert，让每个 EP rank 最终执行恰好相同数量的 token。

前者使分布更健康，后者给系统最坏情况保证。只做 QB 不能保证每一步每张卡完全相等；只做 MoonEP 则可以执行平衡，但不能防止某些 expert 在学习意义上长期收不到 token。

---

## 5. Native Vision：不是后挂一个视觉塔

K3 从预训练第一步起联合优化文本主干、MoonViT-V2 和 projector。图文 token 在一个 next-token prediction 目标中交错，而不是先训纯语言模型、再后接 vision encoder 做对齐。

### 5.1 MoonViT-V2

- 27 层，约 0.4B 参数，patch size 14，12 heads；
- 使用 RMSNorm，linear/attention projection 去除 bias，目的是提高从零训练稳定性；
- 图片和视频完全共享参数；
- attention 分解成 frame 内 spatial pass 与 frame 间 temporal pass；
- temporal pooling 压缩视频 token；
- projector 前做 2×2 pixel shuffle/downsampling，使视觉 token 数降到 1/4；
- 支持最高 3584×3584 输入，而不让视觉 token 过度挤占 1M context。

论文的一个反常规结论是：MoonViT-V2 从零开始、只受语言建模目标监督，也能匹配 SigLIP 初始化的 baseline；作者据此认为，在这种规模的 multimodal LM 中，对比学习初始化不是必要条件。这里应理解为该训练配方下的经验结果，不应外推为所有视觉语言模型都不需要 contrastive pretraining。[论文 §2.4](https://arxiv.org/pdf/2607.24653#page=9)

### 5.2 训练数据对 agentic vision 的针对性

视觉数据除 caption、interleaved image-text、OCR、perception、video 外，还显著扩展了“程序—渲染结果”配对：SVG、3D asset、网页、游戏、CAD schematic。这个分布不是只为静态 VQA，而是为“写代码 → 看渲染/截图 → 修改代码”的 vision-in-the-loop agent 闭环提供预训练基础。

坐标监督同时使用绝对坐标和归一化 `[0,1]` 坐标，以兼顾精确定位与跨分辨率泛化。

---

## 6. 预训练技术

### 6.1 数据管线

文本覆盖 Web、Code、Math、Knowledge 四个主域；各域组合规则过滤、质量 classifier、去重，并通过小模型消融确定 sampling rate。

Knowledge/Math 数据沿用 K2 的 rephrasing recipe：

- 用风格和视角多样的 prompt 改写；
- 长文档采用 chunk-wise autoregressive generation；
- 对照原文做 fidelity verification。

视觉数据则组合开源数据和内部过滤、合成、去重管线。论文没有公开训练 token 总量、精确 domain mixture、各阶段 token 数或合成数据比例。

### 6.2 Scaling-law 不是只拟合最终模型大小

架构、数据和 optimizer 都变了，因此团队重新搜索：batch size、learning rate、tokens-per-parameter、model shape。学习率 schedule 也为 cosine 和 WSD 分别独立搜索最佳超参，而不是拿一套超参硬比较；结果选择 cosine。

论文给出相对 K2 的“overall scaling efficiency 约 2.5×”，含义是拟合 scaling curve 后达到同等 validation loss 所需 FLOPs 的相对变化，不是训练吞吐提升 2.5×，也不是每 token 推理便宜 2.5×。这个数字不能用来直接反推集群规模或 wall-clock。

### 6.3 Optimizer 与基础 recipe

- matrix parameters 使用 Muon；
- Q/K/V projection 的 momentum matrix 按 attention head 分块，各自做 Newton–Schulz orthogonalization，即 Per-Head Muon；
- 直觉是避免大梯度 head 主导整块矩阵的归一化方向，让不同 head 的 update scale 更均衡；
- per-head tall matrix 的 Newton–Schulz 也略便宜于整块矩阵；
- 配合 K2 引入的 weight clipping；
- cosine LR schedule，1% linear warmup；
- weight decay 0.1；
- MoE balancing 使用 QB。

论文没有披露 peak LR、global batch、Muon/AdamW 的具体参数分工与系数、gradient clipping 阈值、训练精度全貌等，不能从公开文字补齐。[论文 §2.5、§3.3](https://arxiv.org/pdf/2607.24653#page=10)

### 6.4 1M 长上下文课程

上下文窗口分四阶段增长：

```text
pre-training:  8K  → 64K
cooldown:     256K → 1M
```

设计意图是把最昂贵的长序列计算集中在总训练预算的一小部分，而不是全程以 1M 训练。

长上下文数据做了专门处理：

- 自然长文档/视频：exact + fuzzy dedup；视频帧 perceptual hash；启发式和 classifier 质量过滤；结构合法性检查；
- 真实长样本稀缺，因此 cooldown 阶段 upsample；
- 仅仅把短文拼成长序列不等于学会长依赖，因此构造 permutation/concatenation 的 multimodal documents 与 sub-tasks，让任务答案必须整合散落在 1M 范围的信息。

NoPE 使窗口扩展不需要修改位置编码，但真正获得 1M 能力仍依赖长程训练任务和 progressive curriculum。**“可以数值外推”与“学会使用远距离信息”是两件事。**[论文 §3.4](https://arxiv.org/pdf/2607.24653#page=12)

---

## 7. 后训练总流程

K3 的后训练可以画成：

```text
                         ┌→ General expert × {low, high, max}
Pretrained K3 → SFT → RL ├→ General-agent expert × {low, high, max}
                         └→ Coding-agent expert × {low, high, max}
                                      │
                                      ▼
                         9 teachers + on-policy student rollout
                                      │
                                      ▼
                         Multi-Teacher On-Policy Distillation
                                      │
                                      ▼
                           单一、多领域、多 effort 模型

从 SFT 开始到 RL：routed-expert MXFP4 / activation MXFP8 QAT
另一路：预训练 MTP layer → EAGLE-3-style draft fine-tuning
```

### 7.1 SFT：为 RL 建立 agent cold start

SFT 不是只做通用 instruction tuning。团队用以往 Kimi 系列的 domain-specialized models 合成长 agent trajectory，再经过多阶段 verification 与 human-in-the-loop annotation。目标是预置：

- adaptive reasoning；
- 精确 tool calling；
- 长程执行基本能力。

所有轨迹统一序列化为 XTML（eXtensible Token Markup Language）chat template。XTML 显式区分 think、response、tools channel，支持带 index 的并行 tool call 和 typed arguments；global options 放在历史输入前，而 one-shot request options 放在输入后，以避免每次改变 request option 都让历史 KV cache 失效。动态工具也可在 session 中途注入。[论文 §4.1.1、Appendix F](https://arxiv.org/pdf/2607.24653#page=46)

### 7.2 RL：3 个领域 × 3 个 reasoning effort

三个大域是：

1. General：通用体验、视觉、推理、faithfulness、搜索、knowledge work；
2. General agents：长程 assistant、deep research、段落级写作；
3. Coding agents：SWE、coding experience、GPU kernel、web development。

每个域训练 low/high/max 三个 effort，得到九个 teacher experts。这里的 expert 是九份领域/effort 专门化 policy，不是模型内部的 MoE expert。

#### 7.2.1 Partial rollout

同步 RL 的主要敌人是长尾：同一批轨迹可能相差几十到几千个 tool calls。每轮对 `N` 个 prompts 各采样 `K` 条 completion，共 `NK` 条活动轨迹；当其中比例 `λ` 完成时就停止等待，立即进入 policy optimization。未完成轨迹进入队列，在下一 iteration 优先恢复。

一条 trajectory 因而可以跨多个 policy iteration，产生极端 stale/off-policy 数据。论文只说明沿用 K2.5 的 optimization algorithm，并用 per-token regularization 把 policy update 限制在局部邻域以容忍 staleness；没有给出本报告可独立复现所需的完整 objective、系数和 `λ/N/K` 配置。

partial rollout 的关键不是“丢弃慢样本”，而是暂停慢轨迹并保留其**逻辑状态与环境状态**：完整 token/message history、prompt/group identity、policy/logprob 等 RL metadata，以及 resumable sandbox。旧 policy 产生的 MLA KV/KDA state 不跨随后的 policy update 直接复用；下一 iteration 恢复 unfinished trajectory 时，需要用当前 policy 对历史重新做 long prefill。这也是 trajectory 跨多个 policy version 后变得 stale/off-policy 的系统来源之一。[论文 §4.1.2](https://arxiv.org/pdf/2607.24653#page=13)

#### 7.2.2 Reasoning-effort RL

每个问题先由 cold-start model 估计初始 token budget `b0(x)`。若轨迹总 token 数 `T(y)` 超过 `τ·b0(x)`，任务 reward 被覆盖为 `-1`：

- general task 的 `T(y)` 是 thinking tokens；
- agentic task 的 `T(y)` 是所有累计输出 token，包括 reasoning trace 与 tool-call arguments。

课程先用较大的 `τ` 训练 max-effort，同时仍设总上限防止无休止 overthinking；随后逐步减小 `τ` 得到 high/low experts。`τ` 按 domain 配置并有人参与指导。

这不是仅在 inference 时截断 token，而是在 RL reward 中把“解题质量—计算预算”权衡学进 policy。

#### 7.2.3 非可验证任务的 Agentic GRM

对没有确定性 verifier 的 general task，使用生成式 judge 做 tournament-style group binary comparison。judge 被强制执行：读产物 → 生成 rubric → 按 rubric 逐个打分 → 写 scorepad。

为防止 reward model 偏爱冗长答案，若 candidate 超过 cold-start verbosity `ℓ0` 的 `σ` 倍，自动输掉 binary comparison。这与 reasoning budget control 同构：都把预算越界变成明确负反馈。

### 7.3 Multi-Teacher On-Policy Distillation（MOPD）

九个专门 policy 最终要合并回一个模型。对采样的 domain `d` 和 effort `e`，选择相应 teacher；student 用自己的 policy 生成 token，在同一 prefix 上计算逐 token dense reward：

$$
r_{opd}^{d}(y_t)=
\operatorname{clip}\left(
\operatorname{sg}\left[
\log\frac{\pi_{teacher}^{(d,e)}(y_t\mid x,y_{<t})}
{\pi_\theta(y_t\mid e,x,y_{<t})}
\right],-R_{max},R_{max}\right).
$$

直觉上：若 student 实际采样的 token 在对应 teacher 下概率更高，就获得正的 dense signal；反之为负。stop-gradient 和 clipping 控制极端 advantage。由于数据来自当前 student 的 on-policy rollout，它比只在 teacher 离线轨迹上做 token CE 更直接地纠正 student 自己会访问的状态分布；又因为 reward 是逐 token 的，可与 partial rollout 基础设施兼容。

论文称试过更细粒度的 top-k distillation objective，但未观察到收敛速度或最终表现优势。[论文 §4.1.3](https://arxiv.org/pdf/2607.24653#page=13)

### 7.4 面向部署的后训练

#### 7.4.1 全程 QAT

从 SFT 开始贯穿 RL：

- routed MoE expert weights：MXFP4；
- routed expert input activations：MXFP8；
- attention projections、latent MoE projections、shared experts、router 等非 routed-expert 模块：更高精度。

rollout inference 与 RL training 使用相同量化方案，避免 train–inference precision mismatch。选择只量化 routed experts 很合理：它们占绝大部分参数内存，同时不把对状态演化、路由或跨维投影敏感的模块一起压到 4 bit。[论文 §4.1.4](https://arxiv.org/pdf/2607.24653#page=14)

#### 7.4.2 从 MTP 变成 EAGLE-3 draft

预训练时有一个结构类似 backbone block 的 MTP layer。后训练将它改造成单层 EAGLE-3-style draft：

- target model 冻结，只更新 draft layer 与 feature-fusion projection；
- 训练时 unroll 7 steps；第一步之后无法获得最新位置的 target feature，draft 改吃自己的先前输出，贴近 inference 时的递归 drafting；
- 输入融合第 1、第 4、最终 AttnRes block 的低/中/高层 feature；
- projection 初始化为 `[0, 0, I]`，初始行为等同原 MTP 所见的 high-level feature，再逐渐学会使用低/中层信息；
- 不用普通 KL surrogate，而直接最小化 lossless speculative sampling 接受率的负对数：

$$
L_{LK}=-\log\sum_{x\in V}\min(p(x),q(x)).
$$

这直接优化 target/draft 分布重叠，而不是假设最小 KL 对小 draft 一定等价于最大接受率。

---

## 8. RL 任务与环境设计

模型变成 agent 的关键训练单元，不再是一问一答，而是：

```text
objective + initial world state + action/tool space + budget + verifier
       ↓
reason → act → observe → verify → adapt → ... → terminal state
```

### 8.1 Unified white-box environment

固定在单一 harness 上训练会过拟合 tool schema、system prompt、context management 和交互协议。K3 把 harness 拆成可组合模块：tools、system prompts、context strategy、skills、memory、subagents 等，再按 task group 动态配置，可模拟 Kimi Code、Claude Code、Codex、OpenClaw、Hermes 或新 harness。

“white-box”在这里强调训练者能控制 harness 组成和环境，不是让 agent 看见 verifier 内部实现。其目标是 scaffold generalization。

### 8.2 知识图谱引导的任务合成

系统从粗粒度 seed nodes 出发，由 agents 递归搜索并扩展成有向无环知识图谱；加新节点前先检索现有图，合并等价/相关概念，边始终从粗概念指向细概念，直到节点足够 atomic。

合成时按目标 domain/task distribution 从不同粒度抽节点或相关节点组合，将节点关键词和 ancestor context 组成 web query，取回论文、博客、代码库等真实材料，再选择 coding、knowledge、vision 等 task type 生成任务。

它解决两个数据覆盖问题：图的层次控制 granularity；跨节点采样控制 diversity，而不是让生成模型凭空反复产出高频题型。

### 8.3 可验证环境的几类代表任务

- 多步搜索：逐步收集网页证据，答案可核验；
- professional workflow：投行、数据分析、法律等，在 sandbox 中跨数十至数百步完成 deliverable；
- visual reasoning：Python sandbox 中反复 crop/zoom/transform、计算并检查生成图；
- kernel optimization：CUDA、Triton、CuTe DSL、Gluon、ThunderKittens、TileLang，覆盖 BF16/FP8/FP4；先以 PyTorch reference 检查数值正确性，再按相对专家实现与 hardware roofline 给性能 reward，并检测 CUDA graph replay、input caching、私自降低精度等 reward hacking；
- personal assistant：Gmail/Notion/Slack/Canvas 的 mock apps，跨多个模拟日和跨应用事件；单 rollout 可到数千 tool calls、累计数百万 context tokens；
- Autonomous Execution Tasks：只给初始状态、目标、约束、工具、预算和独立 verifier，不给 reference trajectory；reward 基于最终环境状态；public verifier 提供诊断、hidden verifier 查 held-out scenarios，并限制提交次数；
- web development：container 中构建网页、游戏、3D/WebGL、可视化、SVG、full-stack app；确定性功能/结构/pixel checks 加 model judge；构建失败、运行报错或伪造产物直接归零。

共同点是：reward 尽量落在**可检查的外部世界状态**，而不是模型声称“已完成”。

---

## 9. KDA 的系统实现

### 9.1 单卡：FlashKDA

chunkwise KDA 的 token-parallel chunk 内计算与 head-parallel 跨 chunk recurrence 有不同并行特征。朴素交替会在串行 state propagation 时让大量 SM 空闲。

FlashKDA 用 CUTLASS 实现，把两部分拆成独立调度/调优的 stages，并 overlap intra-chunk compute 与 cross-chunk state propagation。它同时用于 training 和 inference prefill，并作为 `flash-linear-attention` 的自动选择 backend。K3 lower-bounded gate 进一步让所有 tile 统一为 Tensor Core dense matmul。[论文 §5.1.1](https://arxiv.org/pdf/2607.24653#page=17)

### 9.2 卡内 context parallelism

纯 tensor parallel 只切 heads，不缩短每个 head 的 recurrence；当每 rank 只剩少量 heads 时，超长 prefill 仍无法铺满 SM。

KDA 的一个序列 segment 可以先独立求出“该 segment 对任意输入状态的 transition”，之后精确组合。自动 SM-level CP planner 因而可以在**同一 GPU 内**把 sequence 切给多个 SM 并行计算各 segment transition，再合并得到精确 initial state；不产生跨设备通信。

### 9.3 跨卡 KDA Context Parallelism（KCP）

普通 additive linear attention 的本地状态可以直接做 prefix sum；KDA 不行，因为 delta rule 让 segment 输出依赖进入 segment 的状态。

对 rank `i` 的 segment，KCP 本地计算两个量：

- `M_i`：该 segment 对输入 recurrent state 的累计线性 transition；
- `S̃_i`：从零状态开始，仅由本 segment tokens 生成的 state。

segment 可表示为仿射变换：

$$
F_i(S)=M_iS+\tilde S_i.
$$

两个 segment 的组合仍是同类仿射变换：

$$
F_b(F_a(S))=(M_bM_a)S+(M_b\tilde S_a+\tilde S_b),
$$

而且组合满足结合律。因此各 rank 可先独立算 `(M_i,S̃_i)`，用一次 fixed-size all-gather 交换，再按文档顺序做 prefix scan 重建各 rank 的 incoming state。通信 payload 只与 state shape 有关，不随 local sequence length 增长；计算可随 CP 线性扩展。[论文 §5.1.2](https://arxiv.org/pdf/2607.24653#page=18)

这是 KDA 能支撑 1M training 的核心系统性质：不是简单“把 sequence 切开”，而是找到了可结合的 segment transition 表示。

---

## 10. 3T 预训练 infra

### 10.1 并行组合

论文列出的组合是：

- Pipeline Parallelism（PP）+ virtual pipeline stages（VP）；
- Expert Parallelism（EP）；
- ZeRO-1 Data Parallelism；
- Pipeline ZeRO-2 gradient sharding；
- Context Parallelism（CP/KCP）。

共享 experts 在 EP ranks 上复制。expert dispatch/combine 的 all-to-all 与计算 overlap。论文没有给出生产训练的 PP/VP/EP/DP/CP degree、GPU 型号/数量、网络拓扑、MFU、step time 或故障恢复体系，因此无法判断实际集群布局，也不应臆测用了多少卡。

### 10.2 MoonEP：动态冗余 expert 实现 rank 级完美平衡

普通 EP 中，router 决定的 token 分布动态且倾斜：最热 rank 决定 step 时间，activation shape 每层每步变化还会制造显存碎片。

MoonEP 的方法：

1. 读取当前 micro-batch、当前 layer 的 router output；
2. 在线规划需要临时复制到其他 rank 的 redundant experts；
3. forward expert compute 前预取这些 expert weights；
4. 把 tokens 迁到新放置位置，使每个 rank 恰好收到 `S×K` 个 routed token；
5. backward 在本地 reduce buffer 暂存 replica gradients，计算完后归并回 home rank 的 gradient buffer。

论文证明每 rank 最多预留 `E/R` 个 redundant-expert slots，就对任意路由分布都存在完美平衡方案，且上界基本 tight。精确 ILP 只离线求代表 case 作 reference；线上用近最优 GPU planner，保证不超过此上界且开销可忽略。

#### 10.2.1 完美平衡带来的二阶收益

- zero-copy permute/unpermute：planner 预先知道每个 token 最终位置，token 直接写入远端 expert-grouped buffer，计算直接拿 buffer view；
- communication buffer 从 DeepEP 最坏情况下的 `S×K×R` 降为固定 `S×K`；
- 每 rank token 总数固定，所有 layer 的 compute shape 静态，删除 host 每层读 token count 后再 launch 的同步；
- 静态 shape 也减少 allocator fragmentation；
- rank 内不同 expert 的 token 数仍不均，因此 group GEMM 另有 workload-aware scheduler；参数在 launch 前根据分布与离线校准 cost model 选定，执行中固定；
- shared-expert GEMM 放独立 stream，与其他 kernels overlap。

MoonEP 已公开源码：[MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP)。仓库 README 还说明训练时 redundant slot `B` 必须取 `E/R`，其跨层共享预取池使额外权重槽成本按 process 计，而非每层重复分配。

### 10.3 统一 activation manager

每个 backward 需要的 tensor 都关联可插拔 storage backend：

- recomputation；
- quantization；
- local CPU offload；
- remote offload 到别的 PP rank。

这些策略能在 tensor 粒度组合，只需 annotation，与模型代码解耦；recompute 是 function 粒度，支持跨层。所有 GPU 内存从 main compute stream 分配并进入单一 pool，避免多 stream allocator fragmentation/host overhead；prefetch 按 layer 粒度与 compute overlap。

K3 多数 activation 使用 block-wise FP8，再结合 local/remote offload；elementwise operator 主要选择 recompute。

### 10.4 模块级省显存

#### MoE

- 通过代数变换，让 permuted routing probability 的 gradient 只依赖中间 activation 与 upstream gradient，不再依赖保存 forward output；
- group GEMM forward 只保存 dispatch 前输入；backward 重新 dispatch 恢复 GEMM input；
- recompute 带来的通信与部分 group-GEMM backward overlap。

核心思想是：宁可重做便宜且可 overlap 的 dispatch/elementwise，也不长期保存巨大的 routed activation。

#### AttnRes

- block representation 只在边界生成一次并留在 GPU，供后续层共享；
- AttnRes 全部 checkpoint，使每层为 backward 保存的 activation 与普通 residual 架构相同；
- PP 使用 cache-based incremental communication，只发送新生成 block，micro-batch 结束即释放，达到论文所称理论显存下界。

### 10.5 跨 PP rank 平衡 activation

interleaved 1F1B 中，pipeline warmup 使低编号 PP rank 同时驻留更多 micro-batch activation，高编号 rank 更少。如果所有 rank 按最坏峰值独立配置，会由前部 rank 先 OOM、后部 rank 显存闲置。

K3 用 Mooncake Transfer Engine 把 activation remote-offload 到其他 PP ranks 的空闲显存，从“每层内均衡计算”进一步做到“pipeline 时序上的显存均衡”。

### 10.6 Gradient 与 Muon state

- Pipeline ZeRO-2 在 DP ranks 间 shard gradients；
- sharded gradients 常驻 CPU，GPU 保留 double grad buffer；reduce 后累加进 CPU shards；
- Muon 的 Newton–Schulz 需要完整 parameter matrix，但 naive all-gather 全参数会占巨大 buffer；
- 每个 rank 改为 P2P 只取自己负责更新的参数所缺 shards；按 model-chunk buffer 流水化 communication 与 computation，避免每 rank materialize 全参数 buffer。

### 10.7 Multimodal encoder 如何隐藏

大图/长视频使各 sample 的 ViT 成本差异很大。K3：

- 对单个大图沿 patch 维做 dynamic CP，跨 CP ranks gather KV；
- 一个 CP group 再分多个 sub-CP groups，把多张大图按负载分发，避免规模增大时通信占比同步增大；
- 将 ViT 与 text training 解耦；必须位于最前/最后的少量 ViT forward/backward 同步执行，其余 ViT 计算填进 interleaved 1F1B 的 pipeline bubbles。

这使 vision encoder 从主 critical path 上被大幅隐藏。所谓“原生多模态”不仅是数据和 objective，也是一个专门的 pipeline scheduling 问题。[论文 §5.2](https://arxiv.org/pdf/2607.24653#page=19)

---

## 11. 1M-token Agentic RL infra

### 11.1 为什么采用 co-located

论文称每个 1M-context K3 RL experiment 控制在几百张 GPU 内，因此采用 co-located training：同一批 GPU 在 rollout 与 training 阶段间复用，而不是为 generation 和 update 永久各配一套大集群。

这提高资源利用率，却让 rollout cache 与 training state 争用同一批 GPU/CPU memory。K3 只在 rollout phase 内为当前 policy version 维护 GPU KV/KDA 与 CPU external pool；phase 结束时释放整个 pool，把资源交回 training。未完成轨迹跨轮保留的是历史与 sandbox，而不是旧 policy 的 KV/KDA activation。

### 11.2 External KV cache pool

partial rollout 会让下一轮开头同时恢复许多 unfinished requests。上一轮 rollout pool 已释放，而且 policy optimization 后模型版本发生变化，所以这些请求必须用当前 policy 对完整历史重新 prefill，重建 MLA KV cache 与 KDA recurrent states；1M long prefill 的代价极高。speculative decoding 又会让同一 rollout phase 内更快抵达 tool-call boundary，提高 request turnover，加剧 prefix-block churn。

K3 使用 write-back，而不是 write-through：

- active decoding blocks 留在 GPU；
- 只有从 GPU 被 evict、但以后还会复用的 idle prefix 才写到 CPU DRAM；
- 恢复前 prefetch 回 GPU；
- KDA states 与对应 MLA KV blocks 一起 offload/prefetch，生命周期对齐；
- training iteration 后把 model weights 与 optimizer states offload 到 NVMe，腾出 CPU DRAM 给 rollout external cache pool；
- rollout iteration 结束后释放整个 external pool，再把 CPU DRAM 交回训练；co-located GPU 也切回 training，不继续保留 rollout KV/KDA。

这里形成 GPU → CPU DRAM → NVMe 的分层状态管理，而且存储用途随 training/rollout phase 切换。[论文 §5.3.1](https://arxiv.org/pdf/2607.24653#page=21)

external pool 的生命周期必须明确区分：

```text
training iteration i 结束
  training weights / optimizer states → NVMe
  policy π_i（更新后的当前版本）进入 rollout
                    │
                    ▼
rollout iteration i
  GPU：π_i 的 active MLA KV / KDA states
  CPU：被 GPU evict、但本 rollout 内仍可能复用的 idle prefixes
  tool-call 前后的 requests 可通过 write-back / prefetch 复用 cache
                    │
                    ▼
rollout iteration i 结束
  所有样本对应的 external pool entries 一并释放
  ├── completed trajectory：已经终止，通常无需下一轮恢复
  └── unfinished trajectory：只保留 history + RL metadata + sandbox
                    │
                    ▼
policy optimization：π_i → π_{i+1}
                    │
                    ▼
rollout iteration i+1
  unfinished trajectory 恢复逻辑/环境状态
  用 π_{i+1} 对完整 history 做 long prefill
  重新生成 MLA KV / KDA states 后继续 rollout
```

因此 write-back pool 解决的是**单个 rollout phase 内**的 GPU eviction、tool-call 周期和 request preemption，不消除 iteration boundary 的第一次 long prefill。所有 cache entry 都随 pool 一起清空；区别只是 completed sample 不再继续，而 unfinished sample 下一轮必须重建。论文明确说明 pool 在 rollout iteration 后释放，也明确说上一轮 unfinished requests 会在下一轮开头形成 long-prefill burst。权重更新导致旧 KV/KDA 失效，是由这些 state 对 policy parameters 的依赖得到的工程结论；即使 token history 相同，也不能把 `cache(prefix; π_i)` 接到 `π_{i+1}` 上继续 decode。

### 11.3 Auto-throttling

agent trajectory 的 context 随交互逐步增长。按完整轨迹平均长度设固定 concurrency：前期过于保守；若按短 prefix 设高 concurrency：后期 KV pressure 会触发 preemption。

scheduler 根据 active request count、queued count、KV utilization 动态控制送入 inference engine 的请求数：前期尽量吃满，cache 压力上升时逐步降并发。它控制的是 admission 到 engine 的速率，而不是要求上层预先准确估计每条轨迹最终长度。

### 11.4 复用 gradient buffer 跑 reference model

RL loss 可能要 forward reference/non-policy models，但 2.8T 模型不能常驻额外一份 GPU weights。实现方式：

- reference weights 常驻 CPU；
- 需要时按 chunk stream 到 policy 的 FP32 gradient-buffer storage；
- forward-only 阶段该 buffer 尚未用于真实 gradient，因此可以安全复用；
- 每 GPU 在 ZeRO-2/offload 后只保留两个 VPP chunk 的 grad slots：一个做当前 forward，一个预取下一 chunk，形成 double buffer。

这是典型的“按生命周期复用显存”：同一物理内存在不同 phase 分别承载 reference parameter 与 policy gradient。

### 11.5 AgentENV：环境状态也必须可恢复

跨 iteration 不能依赖旧模型 KV：它会随 rollout pool 释放，并在 policy 更新后失效。真正必须持久化的是完整 token/message history，以及代码、文件、进程、应用数据库等 sandbox world state；两者共同定义 unfinished trajectory 的恢复点。

K3 使用 container sandbox、GPU sandbox，以及最关键的 AgentENV microVM runtime。AgentENV 基于 Firecracker，设计目标是：

#### 高隔离 + 高保真

作者早期 container sandbox 遇到 agent 意外操作引发 kernel panic/deadlock；但又不希望为了安全禁止 mount disk、run container、甚至 nested VM 等真实任务。microVM 把 guest kernel 也隔离，允许更宽的动作面而不直接共享 host kernel。

#### 灵活生命周期

- 增量 checkpoint 只保存自上次 checkpoint 后 dirty 的 memory pages；
- 论文报告 checkpoint/resume 最低延迟 133ms/49ms；
- Pause/Resume：等待 LLM inference 时 sandbox 不占 CPU/内存；论文称等待可占 sandbox lifetime 的 98%；
- Fork：从完全相同状态复制独立 sandbox，可让 reward judge 检查而不污染原环境；
- Snapshot：定期保存以便错误恢复。

#### 高密度启动

OverlayBD image、定制 ublk driver、storage-layer sharing、P2P transport 支持大量异构 image 亚秒级启动；copy-on-write memory 和 page-cache optimization 在真实 workload 达到最高 6.5× memory overcommit。

论文称 K3 训练与评测期间共创建 51,219,741 个 sandboxes、涉及 1,505,678 个 images。这个数字更重要的意义是：sandbox 已是训练 data plane，而不是偶尔调用的测试容器。

AgentENV 已开源：[kvcache-ai/AgentENV](https://github.com/kvcache-ai/AgentENV)。项目 README 明确提醒当前 server 自身不提供 authorization，只应部署在可信网络或认证代理之后；这属于开源版本落地时必须补齐的 control-plane 安全边界。

### 11.6 一条 partial rollout 真正需要保存什么

```text
Trajectory identity / prompt group / policy version / reward bookkeeping
                              │
           ┌──────────────────┴──────────────────┐
           ▼                                     ▼
跨轮逻辑状态                                  跨轮环境状态
完整 token/message history                 microVM memory dirty pages
old logprobs / policy version              writable filesystem layers
sampling / tool-call position              processes / app databases
           │                                     │
           └──────── 同一恢复点一致提交 ──────────┘
                              │
                              ▼
                    下一 RL iteration 恢复
                              │
                              ▼
              用当前 policy 对完整 history long prefill
                              │
                              ▼
                   重建 MLA KV + KDA states
                              │
                              ▼
                           继续生成

旧 policy 的 MLA KV / KDA states：
只在当前 rollout phase 内作为派生 cache 存活，phase 结束随 pool 释放。
```

论文没有明确描述 history 与 sandbox checkpoint 的事务协议、失败恢复语义、exactly-once tool execution、policy-version metadata schema 等。它说明了关键部件和 pool 生命周期，但没有给出可复现的完整 RL control plane。

---

## 12. 推理与在线服务 infra

### 12.1 双 cache 的统一物理管理

每个 hybrid block 有三份 KDA recurrent states 和一份 MLA KV cache：

- MLA KV：按 token 增长、适合 page；
- KDA state：每 request 固定一份、只在选定 prefix boundary 保存 checkpoint。

K3 把二者放进相同 byte size 的 paged block pool，共享 allocation、reference counting、eviction 逻辑。KDA page 内按 head 连续存储，使单 head byte stream 成为跨节点传输最小单位。prefill/decode disaggregation 若使用不同 TP degree，在传输路径重排，不做 GPU-side reshuffle。

### 12.2 为什么物理 page 和 prefix hash 粒度必须解耦

KDA checkpoint 很大，只适合稀疏保存；如果 hash block 必须等于物理 page，page 会被迫大到 1024–6144 tokens。这样短于一页的请求完全不能命中，chunked prefill 在填满页前也没有可复用 prefix。

K3 将：

- physical allocation block 保持粗粒度，例如 6144 tokens；
- prefix hash block 细化到例如 512 tokens；
- KDA checkpoint 只在 MLA hash endpoints 的稀疏子集保存，尤其 conversation turn boundary；
- partially filled MLA page 也按最后一个完整 hash endpoint 注册；
- lookup 先找 MLA 最长匹配，再要求所有 KDA cache groups 在同一 boundary 都有 checkpoint；
- 命中后把 read-only checkpoint copy 到 request-private running state，后续不原地修改共享 cache。

并发一致性还要求：命中的所有 cache groups 先一起 pin，再分配 private block；当轮刚分配/注册但 GPU copy 尚未完成的 block 不参与匹配；任一 KDA group 的 checkpoint eviction 必须原子失效 sibling groups。最终可以在任意 512-token hash boundary 复用，而不受 6144-token physical block 对齐约束。[论文 §5.4.1](https://arxiv.org/pdf/2607.24653#page=23)

### 12.3 KDA speculative decode：重放小输入，不快照大状态

KDA state 每个 decode step 原地更新。MTP draft 若部分 token 被拒绝，state 已前进，不能简单 rollback。若每个 draft position 都保存 state snapshot，state memory traffic 会成为大 batch serving 的瓶颈。

观察是：接受任意 draft prefix 后的 state，可由 draft tokens 的 projected inputs 重建；projected inputs 远小于完整 state。因此 K3：

- 只 cache projected inputs；
- verification 后在片上 replay 被接受 tokens，重建 state；
- 写回 verified token 和 bonus token 的状态；
- replay、bonus、下一 draft window 进入同一个 fused recurrent loop，融合 short conv、input norm、gate、KDA recurrence、output norm。

这把“rollback 大状态”改成“replay 小输入”。

### 12.4 Stable LatentMoE serving kernels

latent projections：

- down projection 与 router 融成一个 GEMM；
- latent weights 跨 ranks shard；
- output all-gather 用 multimem store 融进 GEMM epilogue；
- 通信与 shared-expert compute overlap。

routed expert decode 在小 batch 下通常不是 compute-bound，而是流式读取权重的 memory-bound workload。传统 tile-centric GEMM 预处理重、为算力密集场景设计；K3 基于 WarpDecode 的 token-centric kernel：每 warp 负责一个 output neuron 并流式读相应 weights，再把 warp 分成更小 lane teams 分别处理不同 experts，最后 warp-wide reduce。weights 离线重排一次，降低运行时 dequantization 开销。[论文 §5.4.2](https://arxiv.org/pdf/2607.24653#page=24)

### 12.5 Fleet scheduling

#### Cache-aware affinity

典型 coding session 可能已有 400K prefix，而本轮只新增 4K。把请求发到无 cache 集群会重算整个 prefix；把 cache 跨集群搬走又受较慢 inter-cluster link 限制。

因此 session 固定到持有 prefix cache 的 primary cluster。同时用 consistent hashing 预先指定 secondary；secondary 平时不复制 cache，primary 失败后才重新 prefill。不同 session 的 secondary 均匀分散，使单集群故障的 re-prefill 风暴被摊到全 fleet。这是“常态 cache locality”和“故障域上界”之间的折中，而不是做昂贵的双活 cache replication。

#### Budget-based admission control

请求从不足 2K 到 1M，成本约跨三个数量级；按 request count 或“平均请求”做 capacity planning 都会失真。K3 给不同 request class 独立 resource budget，长上下文突发最多耗尽自己的份额，不能挤占短请求并全面恶化 TTFT/SLO。

这是一种资源隔离，不等价于优先级队列：重点是为不同成本类别设硬预算边界。

---

## 13. 把整套系统串起来

### 13.1 预训练 critical path

```text
DataLoader / multimodal packing
        │
        ├─ ViT：大样本 dynamic CP；多数计算填 PP bubbles
        ▼
PP + VP backbone
        │
        ├─ KDA：FlashKDA + KCP
        ├─ AttnRes：block cache + checkpoint + incremental PP comm
        └─ LatentMoE：QB routing → MoonEP plan/prefetch
                             → zero-copy all-to-all
                             → workload-aware group GEMM
        │
        ▼
Backward：FP8/offload/recompute activation policies
        │
        ├─ remote activation balancing across PP ranks
        ├─ Pipeline ZeRO-2 → CPU gradient shards
        └─ P2P gather needed shards → Per-Head Muon update
```

### 13.2 Agent RL 的 phase 交替

```text
Rollout phase
  GPU: 当前 policy version 的 active KV/KDA state
  CPU DRAM: 当前 rollout 内被 evict 的 reusable prefixes
  NVMe: training weights / optimizer state
  AgentENV: running or paused microVM world states
       │
       ├─ λ fraction done → completed prompt groups enter update
       └─ unfinished trajectories: checkpoint history/metadata/sandbox and queue
       ▼
Rollout boundary
  release all external KV/KDA pool entries
  completed 与 unfinished samples 的 cache 都清空
       │
       ▼
Training phase
  GPU: policy train states / grad buffers
       + streamed reference weights temporarily occupying grad slots
  π_i → π_{i+1}; CPU/NVMe resources change role
       │
       ▼
Next rollout
  unfinished only: restore history/metadata + sandbox
  long-prefill history under π_{i+1}
  rebuild MLA KV + KDA state, then continue
```

### 13.3 最核心的跨层设计模式

| 不规则性 | 转换方式 | 得到的系统收益 |
|---|---|---|
| KDA 逐 token recurrence | chunk transition / associative segment composition | chunk 内并行、跨卡 fixed-size state exchange |
| 无界 decay 数值 | lower-bound gate | 所有 tile 统一 Tensor Core path |
| MoE router imbalance | QB + dynamic redundant experts | 学习负载健康 + 执行 rank 完美平衡 |
| MoE 动态 token shape | 固定每 rank `S×K` | 无逐层 host sync、少碎片、静态 buffer |
| 巨量 activation | policy 化 storage：FP8/offload/recompute | tensor 粒度按成本选择保存方式 |
| 长 RL trajectory 尾延迟 | partial rollout | 不让慢轨迹阻塞 iteration |
| partial rollout 状态跨轮 | history/metadata checkpoint + resumable microVM | 环境与任务连续；KV/KDA 不跨 policy update，下一轮需 long prefill |
| draft rejection 后 KDA rollback | replay projected inputs | 避免逐 draft token 复制大 state |
| 1M 请求成本方差 | cache affinity + class budgets | cache locality 与 SLO 隔离 |

---

## 14. 我的工程判断

本节是基于论文的分析，不是论文原文。

### 14.1 K3 的创新密度主要在接口，而不只在算子

单看 KDA、AttnRes、LatentMoE，每个都可以被视为新 layer；真正困难的是它们改变了框架接口：

- attention cache 从单一 KV 变成 recurrent state + KV；
- residual state 从单一 hidden 变成 block-level history；
- MoE dispatch 从“token 去 expert home rank”变成“planner 动态改变可执行位置”；
- RL sample 从一次性 sequence 变成长寿命、跨 iteration 的 distributed object；
- sandbox 从一次性容器变成可 pause/fork/snapshot 的训练状态。

因此复现 K3 不能只拿 Hugging Face model implementation 跑通 forward；训练与服务框架必须原生理解这些新状态。

### 14.2 K3 在多处用“少量昂贵路径 + 大量便宜路径”

- 3 KDA + 1 MLA：多数层便宜长程混合，少数层全局精确寻址；
- latent routed experts + full-width shared experts：专业容量走窄路径，公共变换走宽路径；
- sparse KDA checkpoints + fine MLA hashes：少量大状态快照配合细粒度 prefix index；
- primary cache affinity + 无 cache secondary：常态局部性优先，故障时才付 re-prefill 成本；
- exact ILP offline + heuristic GPU planner online：精确解用于校准，热路径用有保证的近优算法。

这比“所有东西统一一种机制”更贴近大规模系统现实。

### 14.3 训练稳定性与 infra 效率被共同优化

一些模块同时有算法收益和系统收益：

- lower-bounded KDA gate：防溢出，并删除慢的 diagonal special path；
- SiTU-GLU：抑制 outlier，也让 FP8/FP4 路线更可控；
- QB：防 dying experts，也降低常态 EP imbalance；
- Block AttnRes：改善深度信息访问，也把保存/通信压到 block 级；
- Per-Head Muon：平衡 head update scale，也降低 Newton–Schulz 开销；
- 从 SFT 起 QAT：模型适应量化，同时 rollout 与 train 一致。

这说明 K3 的结构搜索目标不只是 validation loss，而是“在目标硬件和生命周期内可训练、可 rollout、可部署”。

### 14.4 1M agent RL 的瓶颈已经从 token 生成扩展为状态编排

长 agent trajectory 中，GPU 生成只占整个闭环的一部分，sandbox 甚至可能 98% 时间在等模型。系统效率取决于能否正确区分可持久状态与派生状态：

- 同一 rollout phase 内，模型 cache 可从 GPU write-back 到 CPU，保留 prefix 并缓解 churn；
- rollout boundary 后，模型 cache 全部释放，unfinished history 在新 policy 下重新 prefill；
- environment state 可 pause/checkpoint，在不持续占用 host CPU/DRAM 的情况下保留世界进度；
- scheduler 可以不等待 straggler；
- 恢复时 history 与 sandbox 回到一致边界，再由当前 policy 重建模型派生状态。

这与传统 RLHF “prompt → 一次 completion → reward”已经是不同系统范式，更接近分布式 workflow engine + model trainer 的结合。

### 14.5 MoonEP 的价值不只是更快 all-to-all

完美 rank balance 把动态 MoE 变成静态 shape 后，连锁收益包括：固定 buffer、少同步、少碎片、可预测 kernel launch、训练不中断。它改变的是执行模型，而不仅是通信 primitive。

公开 MoonEP README 的 benchmark 结论也显示，其延迟随 router imbalance 基本稳定，而 DeepEP 的最热 rank 会随 imbalance 恶化；但这些结果来自项目自己的 H20 benchmark，仍需在不同拓扑、expert size 和 batch 下独立验证。

### 14.6 最大技术风险仍是混合状态复杂度

KDA–MLA hybrid 同时维护两类 cache，AttnRes 又增加 depth cache；MTP speculative decode 要考虑 KDA state rollback；prefill/decode TP degree 不同还要重排 state。论文给出了成熟的具体解法，但也意味着：

- correctness bug 很可能表现为 silent cache corruption，而不是立即 crash；
- cache group 的边界原子性和生命周期管理成为核心；
- 常规 full-attention serving engine 很难通过少量 patch 完整支持；
- 调试必须把 token boundary、KDA checkpoint version、MLA hash chain、request ownership 一起观测。

论文提到用不同 page 内存布局让 type-confused access 直接产生 garbage，作为零成本 sanity check；这从侧面说明双 cache 类型混淆是实际开发中的重要风险。

---

## 15. 论文没有告诉我们的关键信息

若目标是复现或评估工程可行性，以下信息仍缺失：

### 模型/预训练

- KDA 的 head dimension、value dimension、short-conv kernel size、实际 chunk size；
- MLA latent/rope-less 具体投影维度；
- 总训练 tokens、各 domain/vision mixture、每个 context stage 的 tokens；
- batch size、peak LR、Muon 详细超参、不同参数组 optimizer；
- PP/VP/EP/DP/CP degrees、GPU 型号/数量、网络拓扑、MFU、训练 wall-clock；
- FP8 activation policy 的 block size、哪些 tensor 例外；
- checkpoint、故障检测与大规模训练容错。

### 后训练

- SFT 数据量与 mixture；
- RL 完整 policy objective、advantage 估计、per-token regularizer 形式和系数；
- `N/K/λ`、rollout temperature、每域采样比例；
- 九个 expert policies 是从共同 checkpoint 独立分叉还是阶段性继承的全部细节；
- MOPD 的 domain/effort sampler、与 task reward 的组合权重；
- QAT fake-quant/scale 更新、MX block size 等实现细节。

### RL infra / serving

- rollout state 与 sandbox state 的一致性协议；
- CPU DRAM/NVMe 容量规划、带宽需求、cache eviction policy；
- auto-throttling controller 的具体规则；
- AgentENV control plane、scheduler 和 object-store topology；
- prefix cache 的完整 eviction/admission 策略与 KDA checkpoint 间隔选择；
- fleet request classes 与 budget enforcement 算法。

因此，这篇报告足以理解设计，但不足以从零复刻生产系统。公开的 MoonEP、AgentENV 与 FLA KDA 实现可以补部分执行细节，仍不能补齐内部训练 orchestration。

---

## 16. 建议后续追问的切入点

如果基于本文继续讨论，我建议按以下几条线深入：

1. **KDA 数学与 kernel**：delta rule 如何做 UT transform；KCP 的 backward 如何实现；固定 state 到底多大。
2. **Stable LatentMoE**：latent width 如何改变参数/FLOPs/通信；QB 与 loss-free routing、BIP 的关系；MoonEP planner 的构造。
3. **slime 对照**：partial rollout、stale data、external KV、reference model forward 在 slime 架构里分别对应什么，缺什么。
4. **训练并行推演**：给定 GPU/显存/网络，如何选择 PP×EP×DP×CP，以及 MoonEP redundant slots 的实际内存代价。
5. **Agent RL 状态机**：怎样定义跨轮 trajectory checkpoint，使 history、policy version、sandbox、reward bookkeeping 一致恢复，并在 pool 清空后由当前 policy 安全重建 KV/KDA。
6. **Serving**：hybrid cache page layout、fine hash/coarse allocation、MTP reject 后的 KDA replay，如何落到 vLLM/SGLang 类引擎。

---

## 17. 资料索引

- Kimi Team, [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653)
- 论文 PDF：[arXiv PDF](https://arxiv.org/pdf/2607.24653)
- KDA/linear attention 实现：[fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- Perfectly balanced EP：[MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP)
- Resumable microVM sandbox：[kvcache-ai/AgentENV](https://github.com/kvcache-ai/AgentENV)

最后更新：2026-08-05。
