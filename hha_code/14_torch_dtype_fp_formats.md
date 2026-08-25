# PyTorch dtype 与低精度浮点格式学习笔记

整理时间：2026-07-31  
本地校验环境：`torch 2.11.0+cu129`  

这份笔记的目标不是背 API，而是建立一个稳定的理解框架：一个数值格式到底用多少 bit 存一个值；哪些是 PyTorch 里的真实 `torch.dtype`；哪些是训练/推理框架里的量化 recipe；以及 FP32、FP16、BF16、FP8、MXFP8、MXFP4、NVFP4/FP4 之间到底差在哪里。

## 0. 先建立几个基本概念

### bit、byte、element_size

1 byte = 8 bit。

在 PyTorch 里通常用：

```python
torch.empty((), dtype=some_dtype).element_size()
```

查看一个元素占多少 byte。大部分 dtype 很直观：

- `torch.float32`：4 byte
- `torch.float16`：2 byte
- `torch.bfloat16`：2 byte
- `torch.float8_e4m3fn`：1 byte
- `torch.float8_e5m2`：1 byte
- `torch.int64` / `torch.long`：8 byte
- `torch.int32` / `torch.int`：4 byte
- `torch.int16` / `torch.short`：2 byte
- `torch.int8`：1 byte
- `torch.uint8`：1 byte
- `torch.bool`：通常 1 byte

但 4-bit dtype 会特殊一些。`torch.float4_e2m1fn_x2` 在 PyTorch 里是 packed dtype：两个 4-bit 数打包进一个 byte。因此 PyTorch 的 `element_size()` 返回 1 byte，因为存储和 shape/stride 的操作按 byte 边界处理；从“单个数值槽位”的理论占用看，它是 4 bit，也就是 0.5 byte。

还要注意：`torch.float4_e2m1fn_x2` 更像一个 packed storage dtype / shell dtype。它可以承载 FP4 E2M1 packed byte，但不等于 PyTorch 普通算子都能像处理 `float16`、`bfloat16` 那样直接处理它。

### S-E-M 是什么

浮点数常用 `S-E-M` 描述 bit 分配：

- `S`：sign bit，符号位，通常 1 bit。
- `E`：exponent bits，指数位，决定动态范围。
- `M`：mantissa/fraction bits，尾数位，决定精度。

通俗理解：

- 指数位多：能表示非常大或非常小的数，但相邻数之间更粗。
- 尾数位多：同一数量级内更细，但动态范围可能小。

例如：

- FP32：`S-E-M = 1-8-23`
- FP16：`1-5-10`
- BF16：`1-8-7`
- FP8 E4M3：`1-4-3`
- FP8 E5M2：`1-5-2`
- FP4 E2M1：`1-2-1`



### dtype、format、recipe 不是一回事

这是低精度里最容易混淆的点。

- `torch.float16`、`torch.bfloat16`、`torch.float8_e4m3fn` 是 PyTorch dtype。
- `E4M3`、`E5M2`、`E2M1` 是底层数值编码格式。
- `FP8 delayed scaling`、`MXFP8`、`MXFP4`、`NVFP4` 是“低精度元素 + scale 策略 + kernel/硬件约定”的工程 recipe。

所以不要简单说“MXFP8 是一种新的 8-bit dtype”。更准确地说：MXFP8 通常用 8-bit FP8 元素，再加上 block scale 元数据；它是一种 block-scaled 表示方式。

## 1. 常见 PyTorch dtype 速查

下表以 PyTorch 官方 dtype 命名为主，并结合本地 `torch 2.11.0+cu129` 的 `element_size()` 校验。


| 类别            | PyTorch dtype            | 别名              | 每元素存储       | 典型用途                            |
| ------------- | ------------------------ | --------------- | ----------- | ------------------------------- |
| float         | `torch.float64`          | `torch.double`  | 8 byte      | 高精度数值计算，深度学习很少全量使用              |
| float         | `torch.float32`          | `torch.float`   | 4 byte      | 默认训练/推理精度，稳定但显存大                |
| float         | `torch.float16`          | `torch.half`    | 2 byte      | GPU 混合精度训练/推理                   |
| float         | `torch.bfloat16`         | 无常用短别名          | 2 byte      | 大模型训练常用，动态范围接近 FP32             |
| float8        | `torch.float8_e4m3fn`    | 无               | 1 byte      | FP8 权重/激活，精度比 E5M2 好，范围较小       |
| float8        | `torch.float8_e5m2`      | 无               | 1 byte      | FP8 梯度/反向中更常见，范围更大，精度更粗         |
| float8        | `torch.float8_e4m3fnuz`  | 无               | 1 byte      | 无负零变体，主要用于特定后端/互操作              |
| float8        | `torch.float8_e5m2fnuz`  | 无               | 1 byte      | 无负零变体，主要用于特定后端/互操作              |
| float8 scale  | `torch.float8_e8m0fnu`   | 无               | 1 byte      | 只表示 2 的幂，常用于 microscaling scale |
| float4 packed | `torch.float4_e2m1fn_x2` | 无               | 1 byte 存两个值 | packed FP4，两个 E2M1 值塞进 1 byte   |
| int           | `torch.int64`            | `torch.long`    | 8 byte      | 索引、token id、默认整数张量              |
| int           | `torch.int32`            | `torch.int`     | 4 byte      | 普通整数计算                          |
| int           | `torch.int16`            | `torch.short`   | 2 byte      | 较少用于训练主路径                       |
| int           | `torch.int8`             | 无               | 1 byte      | INT8 量化                         |
| uint          | `torch.uint8`            | 无               | 1 byte      | 图像、字节、量化权重                      |
| uint          | `torch.uint16`           | 无               | 2 byte      | 有限 eager 支持，更多服务编译/导出场景         |
| uint          | `torch.uint32`           | 无               | 4 byte      | 有限 eager 支持，更多服务编译/导出场景         |
| uint          | `torch.uint64`           | 无               | 8 byte      | 有限 eager 支持，更多服务编译/导出场景         |
| bool          | `torch.bool`             | 无               | 1 byte      | mask                            |
| complex       | `torch.complex64`        | `torch.cfloat`  | 8 byte      | 两个 FP32 分量                      |
| complex       | `torch.complex128`       | `torch.cdouble` | 16 byte     | 两个 FP64 分量                      |


PyTorch 新版本中还有一些 shell dtype 或 packed/bit dtype，例如 `uint1`、`uint2`、`uint4`、`int1`、`int2`、`int4`、`bits1x8`、`bits2x4`、`bits4x2`。这些类型往往是为了编译、量化、导出或特定 kernel 服务的，不等于常规 eager 模式下所有算子都能直接用。

## 2. FP32：标准基准线

`torch.float32` 是传统深度学习里的基准格式。


| 属性              | 值                               |
| --------------- | ------------------------------- |
| bit 数           | 32 bit                          |
| byte 数          | 4 byte                          |
| S-E-M           | 1-8-23                          |
| PyTorch dtype   | `torch.float32` / `torch.float` |
| 本地 `finfo.max`  | 约 `3.40e38`                     |
| 本地 `finfo.tiny` | 约 `1.17e-38`                    |
| 本地 `finfo.eps`  | 约 `1.19e-7`                     |


理解方式：

- FP32 的动态范围和精度都比较充足。
- 训练最稳定，但显存、带宽、算力成本高。
- 很多低精度训练仍会保留 FP32 master weight、optimizer state、accumulation 或 loss scaling 相关状态。

例子：如果一个模型有 10B 参数，只按参数本身算：

- FP32：`10B * 4 byte = 40 GB`
- FP16/BF16：`10B * 2 byte = 20 GB`
- FP8：`10B * 1 byte = 10 GB`
- FP4：理论元素位宽 `10B * 0.5 byte = 5 GB`，但真实工程里还要加 scale 元数据和 padding/layout 开销。



## 3. FP16：传统半精度

`torch.float16` 也叫 half precision。


| 属性              | 值                              |
| --------------- | ------------------------------ |
| bit 数           | 16 bit                         |
| byte 数          | 2 byte                         |
| S-E-M           | 1-5-10                         |
| PyTorch dtype   | `torch.float16` / `torch.half` |
| 本地 `finfo.max`  | `65504`                        |
| 本地 `finfo.tiny` | 约 `6.10e-5`                    |
| 本地 `finfo.eps`  | 约 `9.77e-4`                    |


和 FP32 相比：

- 显存减半。
- Tensor Core 支持好，吞吐高。
- 尾数 10 bit，精度还可以。
- 指数只有 5 bit，动态范围远小于 FP32，容易 overflow/underflow。

因此 FP16 训练常需要：

- loss scaling
- FP32 optimizer state
- 某些归一化、累加、softmax 等操作用更高精度



## 4. BF16：大模型训练常用半精度

`torch.bfloat16` 是 Brain Floating Point。


| 属性              | 值                |
| --------------- | ---------------- |
| bit 数           | 16 bit           |
| byte 数          | 2 byte           |
| S-E-M           | 1-8-7            |
| PyTorch dtype   | `torch.bfloat16` |
| 本地 `finfo.max`  | 约 `3.39e38`      |
| 本地 `finfo.tiny` | 约 `1.17e-38`     |
| 本地 `finfo.eps`  | `0.0078125`      |


BF16 的关键设计：保留 FP32 的 8 bit exponent，但减少 mantissa。

通俗理解：

- BF16 和 FP32 一样“看得远”，不容易溢出。
- BF16 看得不够“细”，局部精度比 FP16 粗。
- 大模型训练里，动态范围常比局部尾数精度更关键，所以 BF16 很受欢迎。

FP16 vs BF16：


| 格式   | byte | exponent | mantissa | 优点          | 风险    |
| ---- | ---- | -------- | -------- | ----------- | ----- |
| FP16 | 2    | 5        | 10       | 精度比 BF16 细  | 动态范围小 |
| BF16 | 2    | 8        | 7        | 动态范围接近 FP32 | 局部精度粗 |


如果只记一句话：FP16 更细但容易爆；BF16 更稳但更粗。

## 5. TF32：不是普通 PyTorch 存储 dtype

TF32 经常和 FP32/FP16/BF16 一起出现，但它不是普通张量存储 dtype。

TF32 是 NVIDIA Ampere 之后 Tensor Core 上用于矩阵乘的计算格式。输入通常仍是 FP32 存储，但矩阵乘内部以 TF32 精度执行，累加通常更高精度。

可把它理解为：

- 存储：还是 FP32，4 byte。
- 计算：Tensor Core 使用类似 `1-8-10` 的有效精度。
- 目的：让 FP32 matmul 更快。

在 PyTorch 中常见控制项是：

```python
torch.backends.cuda.matmul.allow_tf32
torch.set_float32_matmul_precision(...)
```

所以 TF32 不应该和 `torch.float16`、`torch.bfloat16` 一样放进“张量 dtype 占用字节”表里。

## 6. FP8：一字节浮点，但必须重视 scale

FP8 表示一个数只用 8 bit，也就是 1 byte。PyTorch 当前常见 FP8 dtype：

- `torch.float8_e4m3fn`
- `torch.float8_e5m2`
- `torch.float8_e4m3fnuz`
- `torch.float8_e5m2fnuz`
- `torch.float8_e8m0fnu`



### 6.1 E4M3


| 属性              | 值                     |
| --------------- | --------------------- |
| bit 数           | 8 bit                 |
| byte 数          | 1 byte                |
| S-E-M           | 1-4-3                 |
| PyTorch dtype   | `torch.float8_e4m3fn` |
| 本地 `finfo.max`  | `448`                 |
| 本地 `finfo.tiny` | `0.015625`            |
| 本地 `finfo.eps`  | `0.125`               |


特点：

- mantissa 3 bit，比 E5M2 多 1 bit，局部精度更好。
- exponent 4 bit，动态范围较小。
- 常用于 forward 的 activation/weight。

这里要注意命名：`E4M3` 通常只是口语简称，表示 `1-4-3` 这种 bit 分配；PyTorch 里没有裸的 `torch.float8_e4m3` dtype，而是具体到 `torch.float8_e4m3fn` 或 `torch.float8_e4m3fnuz` 这样的编码变体。

理解 E4M3 的关键是：它不是每隔固定的 `0.125` 放一个数，而是在不同数量级上使用不同的绝对间隔。`eps = 0.125` 只表示 1.0 附近的间隔。

先只看正的 normal 数，忽略符号位。E4M3FN 的 normal 数可以按这个直觉公式理解：

```text
value = 2^k * (1 + mantissa / 8)
```

其中：

- `k = exponent`bias  `- bias`
- E4M3FN 的 exponent bias 是 7
- mantissa 有 3 bit，所以 `mantissa` 可以是 `0..7`
- 因为 `2^3 = 8`，所以每个 `[2^k, 2^(k+1))` 区间会被切成 8 份

例如 `k = 0`，也就是 `[1, 2)` 这个区间：

```text
mantissa = 0: 1 * (1 + 0/8) = 1.000
mantissa = 1: 1 * (1 + 1/8) = 1.125
mantissa = 2: 1 * (1 + 2/8) = 1.250
mantissa = 3: 1 * (1 + 3/8) = 1.375
mantissa = 4: 1 * (1 + 4/8) = 1.500
mantissa = 5: 1 * (1 + 5/8) = 1.625
mantissa = 6: 1 * (1 + 6/8) = 1.750
mantissa = 7: 1 * (1 + 7/8) = 1.875
```

所以 1.0 附近相邻可表示数的间隔是：

```text
1 / 8 = 0.125
```

这就是 `torch.finfo(torch.float8_e4m3fn).eps = 0.125` 的含义：

```text
eps = 大于 1.0 的下一个可表示数 - 1.0
    = 1.125 - 1.0
    = 0.125
```

再看 `k = 1`，也就是 `[2, 4)`：

```text
2 * (1 + 0/8) = 2.00
2 * (1 + 1/8) = 2.25
2 * (1 + 2/8) = 2.50
2 * (1 + 3/8) = 2.75
...
```

此时间隔变成：

```text
2 / 8 = 0.25
```

再看 `k = -1`，也就是 `[0.5, 1)`：

```text
0.5 * (1 + 0/8) = 0.5000
0.5 * (1 + 1/8) = 0.5625
0.5 * (1 + 2/8) = 0.6250
...
```

此时间隔是：

```text
0.5 / 8 = 0.0625
```

因此 E4M3 normal 数的间隔规律是：

```text
某个 2^k 数量级附近，间隔 = 2^k / 8 = 2^(k-3)
```


| 数值区间          | k   | normal 间隔 |
| ------------- | --- | --------- |
| `[0.25, 0.5)` | -2  | `0.03125` |
| `[0.5, 1)`    | -1  | `0.0625`  |
| `[1, 2)`      | 0   | `0.125`   |
| `[2, 4)`      | 1   | `0.25`    |
| `[4, 8)`      | 2   | `0.5`     |
| `[8, 16)`     | 3   | `1.0`     |


所以更准确的理解是：

```text
浮点数的相邻间隔不是固定的；
数值绝对值越大，相邻可表示数的绝对间隔越大；
数值绝对值越小，相邻可表示数的绝对间隔越小。
```

这也是为什么 `eps` 要专门定义在 1.0 附近：它给了一个统一位置，用来比较不同浮点格式的局部精度。

`tiny = 0.015625 = 2^-6` 表示最小正 normal 数。注意它不是 E4M3FN 能表示的最小正数，因为 E4M3FN 还有 subnormal 区域。更小的正 subnormal 可以到：

```text
2^-9 = 0.001953125
```

也就是说，从 0 往上数，E4M3FN 的下一个正数不是 `0.015625`，而是最小正 subnormal：

```text
0
0.001953125   # 最小正 subnormal
0.00390625
0.005859375
0.0078125
0.009765625
0.01171875
0.013671875
0.015625      # 最小正 normal，也就是 tiny
```

subnormal 数没有 normal 数里的隐藏 leading 1，靠近 0 的这段使用固定间隔。可以把它理解成：

```text
normal 区域：间隔随 2^k 数量级变化
subnormal 区域：靠近 0，间隔固定，但相对精度变差
```



### 6.2 E5M2


| 属性              | 值                   |
| --------------- | ------------------- |
| bit 数           | 8 bit               |
| byte 数          | 1 byte              |
| S-E-M           | 1-5-2               |
| PyTorch dtype   | `torch.float8_e5m2` |
| 本地 `finfo.max`  | `57344`             |
| 本地 `finfo.tiny` | `6.1035e-5`         |
| 本地 `finfo.eps`  | `0.25`              |


特点：

- exponent 5 bit，动态范围大。
- mantissa 2 bit，局部精度更粗。
- 常用于梯度或 backward 中动态范围更大的张量。
- 普通 `torch.float8_e5m2` 保留了 infinity/NaN 这类特殊值编码，所以名字里不需要额外写 `fn`。

E5M2 的间隔计算方式和 E4M3 一样，只是 mantissa 从 3 bit 减少到 2 bit。也就是说，每个 `[2^k, 2^(k+1))` normal 区间不再切成 8 份，而是切成 `2^2 = 4` 份。

normal 数可以按这个直觉公式理解：

```text
value = 2^k * (1 + mantissa / 4)
```

其中：

- `k = exponent - bias`
- E5M2 的 exponent bias 是 15
- mantissa 有 2 bit，所以 `mantissa` 可以是 `0..3`
- 因为 `2^2 = 4`，所以每个 `[2^k, 2^(k+1))` 区间会被切成 4 份

例如 `k = 0`，也就是 `[1, 2)`：

```text
mantissa = 0: 1 * (1 + 0/4) = 1.00
mantissa = 1: 1 * (1 + 1/4) = 1.25
mantissa = 2: 1 * (1 + 2/4) = 1.50
mantissa = 3: 1 * (1 + 3/4) = 1.75
```

所以 1.0 附近相邻可表示数的间隔是：

```text
1 / 4 = 0.25
```

这就是 `torch.finfo(torch.float8_e5m2).eps = 0.25` 的含义。

再看几个区间：

```text
[0.5, 1):  0.5, 0.625, 0.75, 0.875，间隔 0.125
[1, 2):    1.0, 1.25, 1.5, 1.75，间隔 0.25
[2, 4):    2.0, 2.5, 3.0, 3.5，间隔 0.5
[4, 8):    4.0, 5.0, 6.0, 7.0，间隔 1.0
[8, 16):   8.0, 10.0, 12.0, 14.0，间隔 2.0
```

因此 E5M2 normal 数的间隔规律是：

```text
某个 2^k 数量级附近，间隔 = 2^k / 4 = 2^(k-2)
```


| 数值区间          | k   | E5M2 normal 间隔 | 对应 E4M3 normal 间隔 |
| ------------- | --- | -------------- | ----------------- |
| `[0.25, 0.5)` | -2  | `0.0625`       | `0.03125`         |
| `[0.5, 1)`    | -1  | `0.125`        | `0.0625`          |
| `[1, 2)`      | 0   | `0.25`         | `0.125`           |
| `[2, 4)`      | 1   | `0.5`          | `0.25`            |
| `[4, 8)`      | 2   | `1.0`          | `0.5`             |
| `[8, 16)`     | 3   | `2.0`          | `1.0`             |


这个表能看出 E5M2 的核心取舍：在同一个数量级附近，E5M2 的间隔通常是 E4M3 的 2 倍，也就是更粗；但 E5M2 多了 1 个 exponent bit，所以能覆盖更小的 `tiny` 和更大的 `max`。

### 6.3 E4M3 vs E5M2


| 格式   | byte | exponent | mantissa | 动态范围 | 局部精度 | 典型位置                      |
| ---- | ---- | -------- | -------- | ---- | ---- | ------------------------- |
| E4M3 | 1    | 4        | 3        | 小    | 较好   | forward weight/activation |
| E5M2 | 1    | 5        | 2        | 大    | 较粗   | backward/gradient         |


如果只记一句话：E4M3 更准，E5M2 更能扛大范围。

更精确地说，`E4M3` 和 `E5M2` 只描述 bit 分配；真实 dtype 还要说明特殊值编码。PyTorch 中常见的是 `float8_e4m3fn`、`float8_e4m3fnuz`、`float8_e5m2`、`float8_e5m2fnuz`，而不是一个裸的 `float8_e4m3` 或 `float8_e5m2fn`。

### 6.4 `fn`、`fnu`、`fnuz` 后缀

PyTorch 官方说明里这些后缀大致表示：

- `f`：finite only，没有 infinity 编码。
- `n`：NaN 编码和 IEEE 常规形式不完全一样。
- `u`：unsigned，用在某些无符号格式语义中。
- `uz`：unsigned zero，只有一个零，没有负零。

这里的 `finite only` 需要单独解释。普通 IEEE 浮点会保留一些 bit pattern 表示特殊值：

```text
+inf
-inf
NaN
```

例如 FP32 里，某些 exponent/mantissa 组合不表示普通有限数，而是表示 infinity 或 NaN。E4M3FN 的 `FN` 表示 finite only，意思是这个格式不专门保留 `+inf/-inf` 编码，而是尽量把有限的 8 bit 编码空间留给普通有限数值。

所以对 `torch.float8_e4m3fn` 来说：

```text
最大有限正数是 448
最小有限负数是 -448
没有一个比 448 更大的 +inf 档位
```

如果计算或转换结果超过范围，通常会按具体实现进行饱和、截断或产生 NaN 等处理，但它不是像普通 IEEE 浮点那样自然落到 `+inf/-inf` 编码上。

这也是为什么 `E4M3FN` 的名字必须写 `fn`：它不是单纯说明 `1-4-3` 的 bit 分配，还说明了这个格式牺牲 infinity 编码来换更多有限值编码。

所以：

- `float8_e4m3fn`：E4M3，finite-only，NaN 特殊。
- `float8_e4m3fnuz`：E4M3，finite-only，NaN 特殊，且没有负零。
- `float8_e5m2`：E5M2，保留常规 infinity/NaN 特殊值编码，所以名字里不写 `fn`。
- `float8_e5m2fnuz`：E5M2 的 finite-only、无负零变体，和普通 `float8_e5m2` 不是同一个格式。
- `float8_e8m0fnu`：E8M0，无 sign/mantissa，主要表达 2 的幂 scale。

这解释了一个容易疑惑的命名差异：

```text
torch.float8_e4m3fn  带 fn，因为 E4M3 常用变体为了扩大有限数范围，不保留 infinity。
torch.float8_e5m2    不带 fn，因为普通 E5M2 保留 infinity/NaN 特殊值。
```

所以学习时可以把名称分成两层：

```text
E4M3 / E5M2：描述 sign、exponent、mantissa 的 bit 分配。
FN / FNUZ：描述 finite-only、NaN、负零等特殊编码约定。
```



## 7. Shell dtype：名字存在，不代表所有算子都成熟

PyTorch 官方把 FP8、FP4、部分 unsigned/bit dtype 标成 shell dtype 或有限支持 dtype。

直观理解：

- 可以创建 tensor。
- 可以做一些不需要解释元素数值含义的操作，比如 reshape/view/cat。
- 真正读写数值、矩阵乘、cast、isnan/isinf 等操作，支持程度取决于具体 dtype、设备、kernel 和 PyTorch 版本。

这对低精度很重要：看到 dtype 名字不等于“所有普通 PyTorch 算子都能跑”。低精度训练/推理通常通过 Transformer Engine、cuDNN、TensorRT-LLM、torchao、vendor kernel 或框架内置 quantizer 来使用。

## 8. 整数类型：int、uint、quantized int 和 packed bit

整数 dtype 和浮点 dtype 的核心区别：整数没有 exponent/mantissa，它不能表达小数，也没有动态范围/精度的折中；它只是固定 bit 宽里的离散整数。

### 8.1 常规 signed integer

signed integer 有符号，能表示负数。常见范围遵循二进制补码：

```text
intN: -2^(N-1) 到 2^(N-1)-1
```


| PyTorch dtype | 别名            | byte | bit | 取值范围                                          |
| ------------- | ------------- | ---- | --- | --------------------------------------------- |
| `torch.int8`  | 无             | 1    | 8   | `[-128, 127]`                                 |
| `torch.int16` | `torch.short` | 2    | 16  | `[-32768, 32767]`                             |
| `torch.int32` | `torch.int`   | 4    | 32  | `[-2147483648, 2147483647]`                   |
| `torch.int64` | `torch.long`  | 8    | 64  | `[-9223372036854775808, 9223372036854775807]` |


深度学习里最常见的是：

- `torch.int64`：token id、embedding index、label、shape/index 相关张量。PyTorch 很多索引 API 默认要求 long。
- `torch.int32`：一些 kernel、导出、数据处理场景。
- `torch.int8`：量化权重/激活的底层存储，通常不直接拿来当“真实数值”训练。



### 8.2 常规 unsigned integer

unsigned integer 无符号，只表示非负数：

```text
uintN: 0 到 2^N-1
```


| PyTorch dtype  | byte | bit | 取值范围                        | 典型用途                      |
| -------------- | ---- | --- | --------------------------- | ------------------------- |
| `torch.uint8`  | 1    | 8   | `[0, 255]`                  | 图像、字节、量化存储                |
| `torch.uint16` | 2    | 16  | `[0, 65535]`                | 有限 eager 支持，更多用于编译/导出/互操作 |
| `torch.uint32` | 4    | 32  | `[0, 4294967295]`           | 有限 eager 支持，更多用于编译/导出/互操作 |
| `torch.uint64` | 8    | 64  | `[0, 18446744073709551615]` | 有限 eager 支持，更多用于编译/导出/互操作 |


PyTorch 里 `uint8` 是历史上最成熟的 unsigned dtype。`uint16/uint32/uint64` 在官方文档中被标注为有限 eager 支持，很多时候是为了 `torch.compile`、导出或特定后端。

### 8.3 `torch.bool`

`torch.bool` 逻辑上只表达 `False/True`，但 PyTorch 存储上通常是 1 byte/元素，而不是 1 bit/元素。

所以一个 `[N]` 的 bool mask 近似占：

```text
N * 1 byte
```

这点在大 batch、大 sequence length 的 attention mask 或 loss mask 中很重要。bool 看起来只有一位信息，但默认 dense tensor 并不 bit-pack。

### 8.4 Quantized integer dtype：`qint8`、`quint8`、`qint32`

PyTorch 还有 quantized dtype：

- `torch.qint8`
- `torch.quint8`
- `torch.qint32`
- `torch.quint4x2`
- `torch.quint2x4`

这些不是普通整数计算 dtype，而是 PyTorch 量化系统里的“整数存储 + scale/zero_point 解释”的 dtype。

典型反量化公式：

```text
real_value = scale * (integer_value - zero_point)
```

也就是说，一个 `qint8` 张量底层每个值是 8-bit integer，但它代表的真实数值不是 `-128` 到 `127` 这些整数本身，而是通过 scale/zero_point 映射出来的浮点近似值。

这和 FP8 的区别很关键：


| 对比项  | INT8 quantization  | FP8                        |
| ---- | ------------------ | -------------------------- |
| 底层编码 | 整数                 | 浮点，带 exponent/mantissa     |
| 小数表达 | 靠 scale/zero_point | 编码本身可表达不同数量级的小数            |
| 动态范围 | 强依赖 scale 粒度       | dtype 自带一部分动态范围，也常配合 scale |
| 常见用途 | 推理量化，权重/激活压缩       | 训练/推理低精度 GEMM              |




### 8.5 Sub-byte int/uint 和 bits dtype

本地 `torch 2.11.0+cu129` 可以看到这些 dtype 名字：

- `torch.uint1` 到 `torch.uint7`
- `torch.int1` 到 `torch.int7`
- `torch.bits1x8`
- `torch.bits2x4`
- `torch.bits4x2`
- `torch.bits8`
- `torch.bits16`

这些类型的重点是 packed storage 或 bit-level storage，而不是常规 eager 数学计算。例如：

- `bits1x8`：1 个 byte 里放 8 个 1-bit 槽位。
- `bits2x4`：1 个 byte 里放 4 个 2-bit 槽位。
- `bits4x2`：1 个 byte 里放 2 个 4-bit 槽位。

本地环境里这些 dtype 的 `element_size()` 多数返回 1 byte，因为 PyTorch 存储和 stride 仍按 byte 边界管理。它们更适合理解为“底层打包容器”或编译/量化后端接口，不要默认当成 `torch.int32` 那样完整可算。

### 8.6 INT4、INT8、FP4、FP8 的直觉差异


| 格式   | 每值主存储    | 是否浮点 | 是否需要 scale 才像真实模型数值 | 通俗理解                         |
| ---- | -------- | ---- | ------------------- | ---------------------------- |
| INT8 | 1 byte   | 否    | 通常需要                | 256 个均匀整数格点，通过 scale 映射到真实范围 |
| INT4 | 0.5 byte | 否    | 通常需要                | 16 个均匀整数格点，更依赖 scale         |
| FP8  | 1 byte   | 是    | 常配合 scale           | 自带指数，非均匀格点，能覆盖多个数量级          |
| FP4  | 0.5 byte | 是    | 强依赖 scale           | 自带极少指数和尾数，格点很少               |


整数低 bit 量化更像“拿一把均匀刻度尺量某个范围”；浮点低 bit 更像“刻度在 0 附近密一些、远处疏一些”。但无论 INT4/INT8 还是 FP4/FP8，只要进入大模型低精度工程，scale 粒度、outlier 处理和 kernel 支持都会变成核心问题。

工程里说“用 INT4 权重”时，常见做法并不是直接创建一个普通 `torch.int4` 张量。更常见的是：

```text
用 torch.uint8 作为物理存储；
每个 uint8 byte 里塞两个 4-bit 值；
再由专门的 CUDA/Triton/C++ kernel、量化库或推理框架解包，或者在 GEMM kernel 内边解包边计算。
```

例如 unsigned INT4 的 packed 表示可以这样理解：

```python
import torch

x = torch.tensor([1, 2, 3, 15], dtype=torch.uint8)  # 每个值都在 0..15

lo = x[0::2]
hi = x[1::2]
packed = lo | (hi << 4)  # tensor([0x21, 0xf3], dtype=torch.uint8)

unpacked_lo = packed & 0x0F
unpacked_hi = (packed >> 4) & 0x0F
unpacked = torch.stack([unpacked_lo, unpacked_hi], dim=-1).reshape(-1)
```

真实模型权重量化还会保存 scale，常见粒度包括 per-tensor、per-channel、per-group。推理时不一定真的先把整个权重解包成 `int8` 或 `float16`，更高效的实现会让 matmul kernel 直接读取 packed `uint8`，在寄存器里解包、反量化并参与计算。

## 9. FP4：4-bit 浮点

FP4 通常指 E2M1。


| 属性                       | 值                              |
| ------------------------ | ------------------------------ |
| bit 数                    | 4 bit                          |
| 理论每值 byte                | 0.5 byte                       |
| PyTorch packed dtype     | `torch.float4_e2m1fn_x2`       |
| PyTorch `element_size()` | 1 byte，因为两个 4-bit 值 packed 在一起 |
| S-E-M                    | 1-2-1                          |
| 常见值域直觉                   | 大约覆盖 `[-6, 6]`                 |


E2M1 只有 4 bit，总共 16 种编码。它非常省空间，但单个值的信息量极少。没有 scale 时，直接用 FP4 表达真实模型权重/激活通常误差很大。因此工程上几乎一定要配合 block scaling。

一个常见可理解的 E2M1 数值集合是：

```text
0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6
```

为什么只有 16 种？因为 E2M1 的 bit 分配就是：

```text
S E E M
1 2 2 1
```

也就是：

```text
1 bit sign
2 bit exponent
1 bit mantissa
总共 4 bit
```

所以全部 bit pattern 数量是：

```text
2^4 = 16
```

如果固定符号位，例如只看 `sign = 1` 的负数侧，那么只剩：

```text
2 bit exponent + 1 bit mantissa = 3 bit
2^3 = 8 种编码
```

因此 FP4 E2M1 的表达能力真的非常少：正数侧最多 8 种编码，负数侧最多 8 种编码。它厉害的地方不是“单个 FP4 值很精细”，而是：

```text
每个 block 用一个 scale 改变整把尺子的单位长度；
FP4 在这个 block 内只负责选择有限的几个刻度。
```

例如某个 block 的 scale 是 `0.25`，那么前面的 E2M1 刻度整体乘以 `0.25`：

```text
0, ±0.125, ±0.25, ±0.375, ±0.5, ±0.75, ±1, ±1.5
```

换一个 block scale，这 16 个刻度会整体缩放到另一个范围。

为什么 `uint8` 可以编码 FP4 E2M1？因为 `uint8` 本质只是 8 个 bit 的容器。它平时可以解释成整数 `0..255`，但也可以把这 8 个 bit 拆成两个 4-bit code：

```text
一个 uint8 byte:

bit7 bit6 bit5 bit4 | bit3 bit2 bit1 bit0
      高 4 bit       |       低 4 bit
      一个 FP4 code   |       一个 FP4 code
```

例如：

```text
byte = 0b0011_1010

按 uint8 解释：
  0b00111010 = 58

按 packed FP4 解释：
  低 4 bit = 1010 -> 第一个 FP4 E2M1 code
  高 4 bit = 0011 -> 第二个 FP4 E2M1 code
```

同一批 bit 没变，变的是解释规则。`uint8` 不负责说明“这是整数还是两个 FP4”，它只负责装 8 个 bit。真正知道这些 bit 是 FP4 E2M1 的，是 checkpoint 的量化 config、框架 loader 和推理 kernel。

所以：

```text
uint8 storage byte
  只是物理存储

FP4 E2M1
  是每个 4-bit nibble 的逻辑解释方式

MXFP4
  是 packed FP4 E2M1 + E8M0 block scale 的整体 recipe
```

具体编码细节会受 `fn`、是否有负零、是否保留 NaN 等约定影响。学习时可以先抓住核心：FP4 的原生刻度非常少，scale 策略决定它能不能用。

## 10. E8M0：不是普通数值格式，更像 scale 格式

`torch.float8_e8m0fnu` 是 8-bit，`S-E-M = 0-8-0`。

它没有符号位，也没有 mantissa。它表达的是 2 的幂，常用于 microscaling 里的 scale。

例如 block-scaled 表示的核心公式可以写成：

```text
真实近似值 = 低精度元素值 * block_scale
```

当 `block_scale` 是 E8M0 时，这个 scale 基本就是某个 2 的幂。这样硬件实现简单、乘法/缩放便宜，但 scale 不能取任意小数，拟合能力比 E4M3 scale 粗。

### 10.1 E8M0 和整数有什么区别

E8M0 和整数都没有 mantissa，但它们不是一类东西。

最核心区别是：

```text
整数：
  bit pattern 直接解释成一个线性数值。
  例如 uint8 的 127 就是数值 127。

E8M0：
  bit pattern 解释成“指数编码”。
  它表达的是 2 的某个幂，而不是编码值本身。
```

可以这样理解：

```text
uint8:
  code = 127  -> value = 127
  code = 128  -> value = 128
  相邻 code 的差值通常是 1

E8M0:
  code = 127  -> scale = 2^0 = 1
  code = 128  -> scale = 2^1 = 2
  code = 126  -> scale = 2^-1 = 0.5
  相邻 code 的倍率通常是 2 倍，而不是相差 1
```

所以 E8M0 的“0 个 mantissa bit”不是说它像整数一样只能表示 `0, 1, 2, 3, ...`，而是说：

```text
同一个指数下没有额外小数刻度；
它只能表示 2 的幂级别的 scale。
```

对比表：


| 格式      | bit 如何解释   | 数值分布    | 典型用途                      |
| ------- | ---------- | ------- | ------------------------- |
| `uint8` | 直接解释成整数    | 线性      | 索引、普通整数、packed 容器         |
| `int8`  | 直接解释成有符号整数 | 线性      | INT8 量化权重/激活、普通整数         |
| `E8M0`  | 解释成指数      | 对数/2 的幂 | MXFP8/MXFP4 的 block scale |


这也是为什么 Kimi-K3 里 `weight_scale` 物理上可能显示成 `U8`，但它不能简单理解成整数 scale。它的 byte 存储是 8 bit，语义却是 E8M0 scale。推理 kernel 会按 E8M0 的规则解释这些 byte。

## 11. Microscaling：MXFP8 和 MXFP4

MX 是 Microscaling 的缩写。核心思想：

```text
把 tensor 切成很多小 block；
每个 block 共享一个 scale；
block 内每个元素用很低 bit 的格式存；
还原时用 element * scale。
```



### 11.1 MXFP8

MXFP8 常见定义：


| 项          | 值                                       |
| ---------- | --------------------------------------- |
| 元素格式       | E4M3 或 E5M2，实践中 Blackwell MXFP8 常用 E4M3 |
| 元素 bit     | 8 bit                                   |
| block size | 32 个连续元素                                |
| scale 格式   | E8M0                                    |
| scale bit  | 8 bit                                   |
| 公式         | `x ≈ x_fp8 * s_block`                   |


只算元素本身：

```text
每个元素 8 bit = 1 byte
```

把 scale 元数据摊进去：

```text
32 个元素共享 1 个 8-bit scale
摊销 scale = 8 bit / 32 = 0.25 bit / element
总计约 8.25 bit / element = 1.03125 byte / element
```

注意：真实 kernel 还会有 padding、tile layout、对齐、转置副本等开销，不能把 1.03125 byte 当成所有场景的精确显存占用。

为什么 MXFP8 有意义：

- 普通 FP8 如果一个 tensor 共享一个 scale，tensor 内 outlier 会影响所有值。
- MXFP8 每 32 个值一个 scale，局部适配更好。
- 因为局部动态范围压力更小，很多场景可以用 E4M3 覆盖 forward/backward，而不必大量依赖 E5M2。



### 11.2 MXFP4

MXFP4 常见定义：


| 项          | 值                     |
| ---------- | --------------------- |
| 元素格式       | E2M1                  |
| 元素 bit     | 4 bit                 |
| block size | 32 个连续元素              |
| scale 格式   | E8M0                  |
| scale bit  | 8 bit                 |
| 公式         | `x ≈ x_fp4 * s_block` |


只算元素本身：

```text
每个元素 4 bit = 0.5 byte
```

把 scale 元数据摊进去：

```text
32 个元素共享 1 个 8-bit scale
摊销 scale = 8 bit / 32 = 0.25 bit / element
总计约 4.25 bit / element = 0.53125 byte / element
```

MXFP4 比裸 FP4 更可用，但 scale 是 E8M0，只能表示 2 的幂。它简单、硬件友好，但 scale 拟合能力有限。

### 11.3 Kimi-K3 checkpoint 里的 MXFP4 例子

以 Kimi-K3 某个 MoE expert 权重为例，checkpoint 中可能看到类似：

```text
language_model.model.layers.15.block_sparse_moe.experts.0.w2.weight_packed  [3584, 1536]  U8
language_model.model.layers.15.block_sparse_moe.experts.0.w2.weight_scale   [3584, 96]    U8
```

这两个 tensor 的 shape 可以这样拆：

```text
1536 * 2 = 3072
96 * 32 = 3072
```

含义是：

```text
weight_packed 每行有 1536 个 uint8 byte。
如果每个 uint8 打包两个 4-bit 值，那么逻辑上每行就是 1536 * 2 = 3072 个 4-bit 权重。

weight_scale 每行有 96 个 scale。
如果每 32 个逻辑权重共享一个 scale，那么每行 scale 数就是 3072 / 32 = 96。
```

所以从 shape 可以可靠推断：

```text
物理 packed shape: [3584, 1536] U8
逻辑 unpacked shape: [3584, 3072] 个 4-bit 值
group size: 32
每 group 一个 U8 scale
```

这里的乘 2 和除以 2 要按方向区分：

```text
unpacked -> packed：3072 个 4-bit 值 / 2 = 1536 个 uint8
packed -> unpacked：1536 个 uint8 * 2 = 3072 个 4-bit 值
```

但只看 `U8 packed` 和 shape，不能单独证明这 4-bit 值一定是 FP4 E2M1。4-bit packed 也可能表示：

```text
INT4
UINT4
NF4
FP4 E2M1
其他自定义 4-bit codebook
```

判断 Kimi-K3 这里是 MXFP4，需要看模型文档或 config。Kimi-K3 的公开信息中写的是：

```text
Quantization: MXFP4 weights / MXFP8 activations
format: mxfp4-pack-quantized
num_bits: 4
group_size: 32
scale_dtype: torch.uint8
type: float
```

因此完整判断链条应该是：

```text
从 shape 看：这是 4-bit packed + group size 32 scale。
从 Kimi-K3 config/README 看：这个 packed 4-bit 的语义是 MXFP4。
从 MXFP4 定义看：元素格式是 FP4 E2M1，scale 是 E8M0。
```

也就是说，`weight_packed` 物理上是 `uint8`，但它不是 INT8 权重；它是 packed 容器。真正的数值语义由 MXFP4 recipe 决定：

```text
real_weight ≈ fp4_e2m1_value * e8m0_block_scale
```

高效推理实现通常不会先把整个 `weight_packed` 解包成大号 FP16/BF16 权重，而是在专用 GEMM kernel 中读取 packed `uint8`，解出 FP4 E2M1，应用 E8M0 scale，然后参与矩阵乘。

### 11.4 SGLang 里 Kimi-K3 的 MXFP4 是怎么跑的

这一节基于本地 SGLang 源码：

```text
/mnt/shared-storage-user/huanghaian/code/slime_package/sglang
```

先说结论：

```text
Kimi-K3 checkpoint:
  权重主数据：packed MXFP4，物理上用 uint8/int8 byte 装两个 4-bit FP4 E2M1 值
  scale：每 32 个 FP4 权重一个 8-bit scale，语义上是 E8M0

H200 / H100:
  推荐 MoE backend 是 Marlin
  走 W4A16 路径：weight 是 4-bit，activation 通常是 BF16/FP16 级别
  不会把整份权重先解包成一个巨大的 torch.float16 / torch.bfloat16 tensor 再算

B200 / GB200 / Blackwell:
  推荐不显式设置 --moe-runner-backend，自动优先 FlashInfer MXFP4
  有 SiTU cubin pool 时走 FlashInfer MXFP4 / TRT-LLM fused MoE
  默认更接近 W4A8：weight 是 MXFP4，activation 会量化成 MXFP8 后进入 kernel
  如果设置 bf16 precision，则 activation 保持 BF16，让 kernel 内部处理
```

这里的 `W4A16`、`W4A8` 是推理框架里常见的简称：

```text
W4A16 = weight 4-bit，activation 16-bit，通常 BF16/FP16
W4A8  = weight 4-bit，activation 8-bit，Kimi-K3/FlashInfer 这里是 MXFP8 activation
```

注意它们不是 PyTorch dtype 名字，而是 kernel 计算路径的描述。

#### 11.4.1 SGLang 文档里的硬件分叉

SGLang 的 Kimi-K3 文档明确写了：

```text
weights ship in MXFP4
Blackwell 上用 FlashInfer MXFP4 / trtllm-gen SiTU runner
其他平台用 Marlin W4A16
H100/H200 pin Marlin
```

对应文件：

```text
docs_new/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx
docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx
```

其中 Kimi-K3 配置片段里，MoE runner 的选项把 `flashinfer_mxfp4` 标成 Blackwell-only，把 `marlin` 标成 `Marlin (W4A16)`。H200/H100 的 recipe 里显式带有：

```bash
--moe-runner-backend marlin
```

B200/GB200 这类 Blackwell recipe 通常不显式指定 `--moe-runner-backend`，由 SGLang 自动选择 FlashInfer MXFP4；如果相关 cubin pool 不可用，再 fallback 到 Marlin。

#### 11.4.2 加载后并不是变成一个 torch.float4 大 tensor

SGLang 的通用 MXFP4 MoE quant method 在创建参数时，用的是 byte tensor：

```python
weight_dtype = torch.uint8
scale_dtype = torch.uint8
mxfp4_block = 32
```

对应的 expert 权重 shape 也体现 packed 关系：

```text
w13_weight:       [E, 2 * intermediate, hidden / 2]       uint8
w13_weight_scale: [E, 2 * intermediate, hidden / 32]      uint8
w2_weight:        [E, hidden, intermediate / 2]           uint8
w2_weight_scale:  [E, hidden, intermediate / 32]          uint8
```

`/2` 是因为一个 byte 放两个 FP4 E2M1 值；`/32` 是因为 MXFP4 每 32 个逻辑权重共享一个 scale。

所以 checkpoint 里看到的：

```text
weight_packed  U8
weight_scale   U8
```

加载进推理框架后，本质仍然是 packed byte + scale。后面只是根据后端需要做 layout reorder、interleave、shuffle、padding，而不是把它变成“普通 torch.float4 矩阵”再交给通用 matmul。

#### 11.4.3 H200：Marlin W4A16 路径

H200 是 Hopper SM90。SGLang 对 Kimi-K3 的 H200 recipe 里 pin 了：

```bash
--moe-runner-backend marlin
```

Marlin 的 MXFP4 MoE 实现里，权重仍然是 packed 4-bit：

```text
w13_qweight / w2_qweight: packed 4-bit weight
w13_scales / w2_scales: E8M0 scale
weight_bits = 4
```

Marlin runner 最后调用 fused kernel：

```python
fused_marlin_moe(
    hidden_states=...,
    w1=w13_qweight,
    w2=w2_qweight,
    w1_scale=w13_scales,
    w2_scale=w2_scales,
    num_bits=4,
)
```

这说明 H200 上并不是：

```text
先把全部 MXFP4 权重反量化成 BF16 大矩阵 -> 再普通 matmul
```

而更接近：

```text
packed FP4 weight + E8M0 scale 保持压缩布局
Marlin fused MoE kernel 内部读取 packed weight
kernel 内部完成 unpack / scale / GEMM
activation 走 16-bit 路径
```

源码里还有一个细节：如果输入 activation 是 `float16`，而权重 scale 是 `torch.float8_e8m0fnu`，Marlin runner 会把 activation 转成 `bfloat16`，因为这条 MXFP4(E8M0) Marlin kernel 在数值上只支持 BF16 activation 路径。算完再 cast 回原来的 hidden dtype。

所以 H200 可以理解为：

```text
权重：MXFP4 packed W4
scale：E8M0
activation：BF16/FP16 级别，实际 kernel 对 E8M0 路径使用 BF16 activation
计算：Marlin fused MoE W4A16
```

更细一点看，H200 上确实没有 Blackwell 那种原生 FP4 Tensor Core 路径，所以 Marlin 的 `W4A16` 不是“硬件直接拿 FP4 做 MMA”。它做的是：

```text
global memory 里：
  weight 仍然是 packed FP4 E2M1
  scale 仍然是 E8M0

kernel 内部：
  1. 读取 packed FP4 byte
  2. 在 register fragment 里把 FP4 E2M1 解码成 BF16 fragment
  3. 把 E8M0 scale 解码成 BF16 scale
  4. 对 weight fragment 应用 scale
  5. 用 BF16 Tensor Core MMA 做：
       BF16 activation * BF16 dequantized-weight-fragment -> FP32 accumulate
  6. 最后写回 BF16/原 hidden dtype 输出
```

所以它不是：

```text
MXFP4 weight -> FP8 weight -> BF16 activation x FP8 weight
```

而是更接近：

```text
MXFP4 packed weight -> kernel register 内临时 BF16 fragment -> BF16 MMA
```

这里“临时 BF16 fragment”很重要：它不是把整个专家权重完整展开成 BF16 tensor 存在显存里，而是边读、边解码、边乘。这样显存带宽和权重存储仍然享受 W4 压缩，计算阶段则复用 Hopper 已经支持得很成熟的 BF16 Tensor Core。

那为什么 H200/Marlin 不把它变成 FP8 来算？

主要原因有几个：

```text
1. Marlin 这条 kernel 设计就是 W4A16：
   W4A16 = weight 低 bit 存储，activation 保持 16-bit。

2. Hopper 支持 FP8 Tensor Core，但不支持 Blackwell 那种原生 FP4/MXFP4 Tensor Core 路径。
   MXFP4 权重仍然必须先从 FP4 E2M1 + E8M0 scale 解码出来。

3. 如果为了用 FP8 MMA，把权重 fragment 再压到 FP8：
   packed FP4 -> 解码/scale -> FP8 fragment
   这会多一次 FP8 量化/舍入，而且 activation 也通常要量化到 FP8。
   对 accuracy-preserving runner 来说，这不划算。

4. BF16 activation + BF16 weight fragment 的路径简单、稳定、精度更好。
   权重存储仍然是 4-bit，所以主要显存带宽收益还在。
```

所以 H200/Marlin 的思路不是追求“全程 FP8”，而是：

```text
用 W4 减少权重存储和读带宽；
用 BF16 Tensor Core 保持计算兼容性和精度；
把解包和 scale 融在 GEMM kernel 里，避免显存里产生完整 BF16 权重。
```

这里还要区分“能加载 FP4 E2M1 编码”和“硬件原生支持 FP4 计算”：

```text
硬件不支持 FP4 E2M1 原生计算
!=
软件不能加载 FP4 E2M1 编码
```

FP4 E2M1 首先是一套 4-bit 编码规则。只要框架知道这套规则，就可以用 `uint8/int8` 把它加载进来：

```text
每个 byte 存两个 FP4 E2M1 code；
再配合每 32 个值一个 E8M0 scale；
checkpoint / parameter 里仍然可以保持 packed byte 形态。
```

这一步只是读 byte、保存 byte、按 shape 组织内存，不要求 GPU 有 FP4 Tensor Core。

真正受硬件限制的是计算阶段：

```text
硬件支持 FP4 Tensor Core：
  可以更直接地让低精度 kernel 消费 FP4/MXFP4。

硬件不支持 FP4 Tensor Core：
  仍然可以加载 packed FP4；
  但 kernel 里要先解码/反量化成 FP16/BF16/FP32 fragment；
  然后用硬件支持的 FP16/BF16/FP32/INT 路径计算。
```

所以 H200/Marlin 支持 Kimi-K3 MXFP4，不等于 H200 原生支持 FP4 计算。更准确地说，是 SGLang/Marlin 写了能读这种 packed MXFP4 格式并在线反量化的 kernel。

如果 checkpoint 里的 `uint8` byte 本来就已经是 FP4 E2M1 packed 编码，并且 nibble 顺序符合 PyTorch 的 `float4_e2m1fn_x2` 约定，那么可以把它 view 成 PyTorch 的 packed FP4 dtype：

```python
w_u8 = torch.empty((3584, 1536), dtype=torch.uint8)
w_fp4_packed = w_u8.view(torch.float4_e2m1fn_x2)

print(w_fp4_packed.dtype)         # torch.float4_e2m1fn_x2
print(w_fp4_packed.element_size()) # 1 byte，因为一个元素槽位仍是 packed byte
```

但这个 `view` 只是 reinterpret 同一批 byte，不是数值解码，也不是 scale 还原。它不会自动应用 E8M0 scale。

更重要的是，`torch.float4_e2m1fn_x2` 不能按普通浮点 dtype 期待：

```text
可以：
  用作 packed FP4 存储表示；
  在某些 kernel 里作为输入格式；
  和 uint8 之间做 byte-level reinterpret/view。

不要默认期待：
  torch.float16.to(torch.float4_e2m1fn_x2) 一定可用；
  torch.float4_e2m1fn_x2.to(torch.bfloat16) 一定可用；
  torch.matmul / torch.mm 能直接消费它；
  H100/H200 能原生 FP4 Tensor Core 计算它。
```

在当前本地环境 `torch 2.11.0+cu129` 里，`torch.float4_e2m1fn_x2` dtype 存在，`uint8.view(torch.float4_e2m1fn_x2)` 可以工作，但普通 `to()` 转换仍可能报 `copy_ not implemented for 'Float4_e2m1fn_x2'`。因此实际推理框架通常还是保留 packed byte，然后交给专门 kernel 解释和计算。

#### 11.4.4 B200：FlashInfer MXFP4 / TRT-LLM fused MoE 路径

B200 是 Blackwell。SGLang 的 Kimi-K3 文档建议 Blackwell 上不要手动 pin `--moe-runner-backend`，让它优先用 FlashInfer MXFP4。通用 MXFP4 MoE 代码里，启用 `flashinfer_mxfp4` 后按 GPU 架构分派：

```text
SM100 Blackwell -> trtllm_fp4_block_scale_moe
SM120 Blackwell -> cutlass_fused_moe(MXFP8 x MXFP4)
SM90 Hopper     -> cutlass_fused_moe(use_w4_group_scaling=True)
```

B200/GB200 这类数据中心 Blackwell 通常对应 SM100 路径，因此会进入 TRT-LLM / SiTU fused MoE kernel。

FlashInfer TRT-LLM MXFP4 实现里，权重参数仍然是 packed byte：

```text
w13_weight: int8，最后一维 hidden / 2
w2_weight:  int8，最后一维 intermediate / 2
```

scale 会被转成 E8M0：

```python
w13_scale = w13_scale.to(torch.float8_e8m0fnu)
w2_scale = w2_scale.to(torch.float8_e8m0fnu)
```

然后根据 FlashInfer / TRT-LLM kernel 的 ABI 做 shuffle 和 interleave。这里有时会看到 scale 被 `.view(torch.float8_e4m3fn)`，这不代表原始 MXFP4 scale 语义突然变成普通 E4M3 数值；更准确地说，这是为了匹配 kernel 需要的 byte-level 布局和 dtype view。理解上仍然应该把 Kimi-K3 的 MXFP4 block scale 看成 E8M0。

真正 apply 时有两种 activation precision：

```text
precision == "default":
  hidden_states 先通过 flashinfer_mxfp8_quantize 量化成 MXFP8 activation
  然后调用 trtllm_fp4_block_scale_routed_moe
  这就是 W4A8 路径

precision == "bf16":
  hidden_states 保持 BF16
  x_scale = None
  让 TRT-LLM kernel 内部处理
  这更接近 W4A16/BF16 activation 路径
```

所以 B200 默认更像：

```text
权重：MXFP4 packed W4
weight scale：E8M0
activation：MXFP8
计算：FlashInfer / TRT-LLM fused MoE W4A8
```

不是：

```text
先把 MXFP4 权重反量化为 FP8 权重 -> 再 FP8 GEMM
```

更准确的描述是：

```text
权重保持 packed MXFP4；
activation 可能被量化成 MXFP8；
专用 fused MoE kernel 同时消费 packed weight、weight scale、activation、activation scale。
```

这里不要把硬件理解成“有一个单独的 MXFP4 core”。更准确的说法是：

```text
硬件单元仍然是 Blackwell Tensor Core；
FP4 是元素格式；
MXFP4 是 FP4 E2M1 + E8M0 block scale + 特定 layout/kernel ABI；
Blackwell Tensor Core 支持新的低精度和 microscaling 路径，
所以专用 kernel 可以直接消费 MXFP4/MXFP8 这种带 scale 的格式。
```

也就是说，B200 上的 MXFP4 不是：

```text
普通 FP4 core
+ 完全由软件手动 scale
```

而更像：

```text
Blackwell Tensor Core 的 FP4 低精度能力
+ 硬件加速的 block scaling / microscaling 支持
+ FlashInfer / TRT-LLM 对应的 fused kernel 和 layout
```

NVIDIA 的资料里也把 Blackwell 支持的 4-bit 浮点路径分成 `FP4`、`MXFP4`、`NVFP4`，其中裸 `FP4` 只有 E2M1 元素和软件 scale，而 `MXFP4` 是 E2M1 元素加每 32 个值一个 power-of-two scale，并标注有 accelerated hardware scaling。Transformer Engine 文档也写到 Blackwell 新增了 `NVFP4` 和 `MXFP8`，并介绍 MXFP8 使用每 32 个值一个 E8M0 scale；NVIDIA Blackwell 架构介绍里则把这种能力称为 micro-tensor scaling / microscaling 的低精度 Tensor Core 能力。

#### 11.4.5 什么时候会真的反量化成 BF16

SGLang 的 MXFP4 通用代码里确实有兜底逻辑：

```python
from triton_kernels.numerics_details.mxfp import upcast_from_mxfp

w13_weight = upcast_from_mxfp(..., target_dtype=torch.bfloat16)
w2_weight = upcast_from_mxfp(..., target_dtype=torch.bfloat16)
```

这表示在某些没有命中特化后端的路径下，框架可以把 MXFP4 权重 upcast 成 BF16 存起来再算。

但对 Kimi-K3 的 H200/B200 推荐部署来说，这不是主路径：

```text
H200: Marlin W4A16，packed weight 交给 Marlin kernel
B200: FlashInfer MXFP4，packed weight 交给 TRT-LLM / FlashInfer kernel
```

因此回答“正常推理引擎咋加载”：

```text
不是加载成 torch.float4；
也不是通常先全量反量化成 FP8；
而是加载成 packed byte + scale，
再按具体后端重排成 kernel 需要的布局，
最后由专用 fused kernel 在计算时消费 packed FP4 和 scale。
```

源码定位：

```text
Kimi-K3 硬件/runner 说明:
  /mnt/shared-storage-user/huanghaian/code/slime_package/sglang/docs_new/cookbook/autoregressive/Moonshotai/Kimi-K3.mdx
  /mnt/shared-storage-user/huanghaian/code/slime_package/sglang/docs_new/src/snippets/configs/moonshotai/kimi-k3.jsx

MXFP4 通用 MoE loader:
  /mnt/shared-storage-user/huanghaian/code/slime_package/sglang/python/sglang/srt/layers/quantization/mxfp4.py

H200/H100 Marlin 路径:
  /mnt/shared-storage-user/huanghaian/code/slime_package/sglang/python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py
  /mnt/shared-storage-user/huanghaian/code/slime_package/sglang/python/sglang/srt/layers/moe/moe_runner/marlin.py

B200/Blackwell FlashInfer 路径:
  /mnt/shared-storage-user/huanghaian/code/slime_package/sglang/python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py
  /mnt/shared-storage-user/huanghaian/code/slime_package/sglang/python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py
  /mnt/shared-storage-user/huanghaian/code/slime_package/sglang/python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py
```



### 11.5 DSKV4 Pro checkpoint 里的 MXFP4 例子

DeepSeek-V4 Pro / DSKV4 Pro 一类 checkpoint 里，也可能看到这种字段命名：

```text
layers.7.ffn.experts.0.w3.scale   [3072, 224]   F8_E8M0
layers.7.ffn.experts.0.w3.weight  [3072, 3584]  I8
```

这和 Kimi-K3 示例里的 `weight_packed U8 + weight_scale U8` 是同一类问题：不要先被字段名里的 `I8` 带偏，要先看 shape 和 scale。

按 shape 反推：

```text
3584 * 2 = 7168
224 * 32 = 7168
```

含义是：

```text
w3.weight 每行有 3584 个 int8 byte。
如果每个 byte 打包两个 4-bit 值，那么逻辑上每行有 7168 个 4-bit 权重。

w3.scale 每行有 224 个 E8M0 scale。
如果每 32 个逻辑权重共享一个 scale，那么 7168 / 32 = 224。
```

所以这个字段组合可以理解为：

```text
物理 weight：I8 byte 容器，每 byte 两个 4-bit code
逻辑 weight：每行 7168 个 4-bit 值
scale：F8_E8M0，每 32 个逻辑权重一个 scale
group size：32
```

这里的 `I8` 不要理解成普通 INT8 权重。对 packed 4-bit 来说，`I8` / `U8` 很多时候只是“8-bit 容器 dtype”的显示差异；真正参与解释的是 byte 里的两个 nibble。kernel 往往会把它当 raw byte / `uint8` view 来读。

再结合 `scale` 明确是 `F8_E8M0`，这强烈说明它是 MXFP4 风格：

```text
real_weight ≈ fp4_e2m1_value * e8m0_block_scale
```

仅从 `weight I8` 本身不能证明一定是 FP4；但是 `weight I8 packed + scale F8_E8M0 + group size 32` 这三个信息合在一起，就基本是在描述 MXFP4 packed weight。

### 11.6 DSKV4 Pro 在 SGLang 里是否要单独 kernel

结论：

```text
底层 FP4/MXFP4 MoE 计算 kernel：
  不需要因为 DSKV4 Pro 再从零写一套。
  它可以复用 Marlin / FlashInfer MXFP4 这类 backend。

模型适配层：
  需要 DSKV4 专用逻辑。
  因为 DSKV4 的 checkpoint 字段命名、expert 布局、topk、shared expert、
  routed scale、swiglu_limit 等细节不一定和 Kimi-K3 完全一样。
```

SGLang 里 DSKV4 的入口逻辑大致是：

```text
1. DeepSeekV4 config 探测 routed expert weight dtype。

   U8 / I8 / F4 -> is_fp4_experts = True
   F8_E4M3      -> is_fp4_experts = False

2. ModelConfig 只对 DeepSeekV4 设置这个 is_fp4_experts 标志。

3. model_loader 把 is_fp4_experts 塞进 Fp8Config。

4. Fp8Config.get_quant_method() 根据 MoE backend 分发：

   is_fp4_experts + marlin
     -> Mxfp4MarlinMoEMethod

   is_fp4_experts + flashinfer_mxfp4
     -> Mxfp4FlashinferCutlassMoEMethod
        或 Mxfp4FlashinferTrtllmMoEMethod

   否则
     -> 普通 Fp8MoEMethod
```

所以 DSKV4 Pro 和 Kimi-K3 的关系可以这样理解：

```text
相同点：
  都是 MXFP4 packed weight + E8M0 scale；
  H100/H200 可以走 Marlin W4A16；
  B200/Blackwell 可以走 FlashInfer MXFP4 W4A8/W4A16；
  底层计算 kernel 属于同一类 MXFP4 backend。

不同点：
  Kimi-K3 更直接走 mxfp4 quant config 的通用路径；
  DSKV4 Pro 在 SGLang 里更像 fp8 quant config + is_fp4_experts 标志；
  DSKV4 Pro 需要额外 adapter 把自己的 checkpoint 布局整理成 Marlin/FlashInfer 能消费的布局。
```

一句话：

```text
DSKV4 Pro 不需要因为自己是 DSKV4 就重写一套 FP4 GEMM kernel；
但需要模型专用适配层，把 DSKV4 的 checkpoint/layout 接到已有 MXFP4 kernel 上。
```

源码定位：

```text
DSKV4 routed expert dtype 探测：
  /mnt/shared-storage-user/huanghaian/code/slime_package/sglang/python/sglang/srt/configs/deepseek_v4.py

ModelConfig 设置 is_fp4_experts：
  /mnt/shared-storage-user/huanghaian/code/slime_package/sglang/python/sglang/srt/configs/model_config.py

loader 把 is_fp4_experts 传入 Fp8Config：
  /mnt/shared-storage-user/huanghaian/code/slime_package/sglang/python/sglang/srt/model_loader/loader.py

Fp8Config 按 backend 分发到 MXFP4 MoE method：
  /mnt/shared-storage-user/huanghaian/code/slime_package/sglang/python/sglang/srt/layers/quantization/fp8.py

DSKV4/FP4 expert 可复用的 backend adapter：
  /mnt/shared-storage-user/huanghaian/code/slime_package/sglang/python/sglang/srt/layers/quantization/mxfp4_marlin_moe.py
  /mnt/shared-storage-user/huanghaian/code/slime_package/sglang/python/sglang/srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py
  /mnt/shared-storage-user/huanghaian/code/slime_package/sglang/python/sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py
```



## 12. NVFP4：NVIDIA 的 FP4 block scaling recipe

NVFP4 是 NVIDIA Blackwell 上重点支持的 4-bit 低精度格式/recipe。它不是 PyTorch 里的单个 `torch.dtype` 名字，而是一个完整的量化表示策略。

常见定义：


| 项                | 值                                                  |
| ---------------- | -------------------------------------------------- |
| 元素格式             | E2M1 FP4                                           |
| 元素 bit           | 4 bit                                              |
| micro-block size | 16 个连续元素                                           |
| block scale      | FP8 E4M3                                           |
| tensor scale     | FP32                                               |
| 公式直觉             | `x ≈ x_fp4 * block_scale_e4m3 * tensor_scale_fp32` |


只算元素本身：

```text
每个元素 4 bit = 0.5 byte
```

摊销 block scale：

```text
16 个元素共享 1 个 8-bit E4M3 scale
摊销 scale = 8 bit / 16 = 0.5 bit / element
元素 + block scale = 4.5 bit / element = 0.5625 byte / element
```

再加上 per-tensor FP32 scale：

```text
每个 tensor 额外 32 bit
如果 tensor 很大，这个成本几乎可以忽略
如果 tensor 很小，这个成本就不该忽略
```

NVFP4 相比 MXFP4 的关键变化：


| 格式    | block size | block scale     | 直觉                |
| ----- | ---------- | --------------- | ----------------- |
| MXFP4 | 32         | E8M0，2 的幂 scale | 简单，但 scale 粗      |
| NVFP4 | 16         | E4M3，FP8 scale  | block 更小，scale 更细 |


为什么 NVFP4 通常比 MXFP4 准：

- block 从 32 缩小到 16，一个 outlier 影响的值更少。
- scale 从 E8M0 换成 E4M3，可以表示非 2 的幂小数 scale。
- 额外 per-tensor FP32 scale 用于全局归一化，帮助 block scale 更好覆盖局部分布。



## 13. “NVFP8”这个词怎么理解

严格从 PyTorch dtype 和 NVIDIA Transformer Engine 常见 API 看，主线名称是：

- FP8：E4M3/E5M2 等 8-bit 浮点元素格式。
- MXFP8：Blackwell 上的 microscaling FP8 block scaling recipe。
- NVFP4：NVIDIA 的 FP4 block scaling recipe。

“NVFP8”有时会出现在消费级应用、模型 checkpoint 或宣传材料里，用来表示“面向 NVIDIA RTX/GPU 优化的 FP8 量化权重/推理路径”。但它不是当前 PyTorch 官方 dtype 名称，也不是像 NVFP4 那样在 Transformer Engine API 中非常明确的一类 recipe 名称。

如果别人说 `NVFP8`，建议先追问或检查：

- 它的底层元素是 E4M3 还是 E5M2？
- 是 per-tensor scaling、per-channel scaling，还是 block scaling？
- scale 是 FP32、E8M0、E4M3，还是其他？
- checkpoint 文件里真实 dtype 是什么，比如 safetensors metadata、框架加载日志、kernel 文档。

在没有更多上下文时，保守理解：

```text
NVFP8 ≈ NVIDIA 平台上的 FP8 量化/推理称呼，不等于一个标准 torch.dtype。
```



## 14. 显存占用对比：只看元素 vs 加 scale

假设有 N 个数值。


| 表示                | 元素 bit   | 元素 byte | scale 元数据                                            | 摊销后近似                             |
| ----------------- | -------- | ------- | ---------------------------------------------------- | --------------------------------- |
| FP32              | 32       | 4       | 无                                                    | 4 byte/值                          |
| FP16              | 16       | 2       | 无                                                    | 2 byte/值                          |
| BF16              | 16       | 2       | 无                                                    | 2 byte/值                          |
| INT64             | 64       | 8       | 无                                                    | 8 byte/值                          |
| INT32             | 32       | 4       | 无                                                    | 4 byte/值                          |
| INT16             | 16       | 2       | 无                                                    | 2 byte/值                          |
| INT8/UINT8        | 8        | 1       | 普通 int 无；量化 int 常有 scale/zero_point                  | 至少 1 byte/值                       |
| BOOL              | 逻辑 1 bit | 通常 1    | PyTorch dense bool 默认不 bit-pack                      | 1 byte/值                          |
| FP8 E4M3/E5M2     | 8        | 1       | 可能有 per-tensor/per-block scale                       | 至少 1 byte/值                       |
| MXFP8             | 8        | 1       | 每 32 值 1 个 8-bit E8M0 scale                          | 约 1.03125 byte/值                  |
| INT4/UINT4 packed | 4        | 0.5     | 量化时通常有 scale/zero_point                              | 至少 0.5 byte/值 + 元数据               |
| FP4 E2M1          | 4        | 0.5     | 裸 FP4 通常不可直接好用                                       | 0.5 byte/值                        |
| MXFP4             | 4        | 0.5     | 每 32 值 1 个 8-bit E8M0 scale                          | 约 0.53125 byte/值                  |
| NVFP4             | 4        | 0.5     | 每 16 值 1 个 8-bit E4M3 scale，另有 per-tensor FP32 scale | 约 0.5625 byte/值 + 很小 tensor scale |


这些数字适合理解“理论下限和主要元数据”，但真实显存还可能包含：

- 对齐和 padding
- scale tensor 的 tiled layout
- rowwise/columnwise 双份 scale
- 转置副本
- amax/history buffer
- optimizer state
- activation checkpoint
- kernel workspace

所以实际训练显存不能只用参数量乘这个表。

## 15. 数值格式对比：范围和精度直觉


| 格式       | S-E-M  | byte | 动态范围         | 局部精度       | 通俗理解             |
| -------- | ------ | ---- | ------------ | ---------- | ---------------- |
| FP32     | 1-8-23 | 4    | 很大           | 很细         | 稳，但贵             |
| FP16     | 1-5-10 | 2    | 中等           | 较细         | 快且省，但容易 overflow |
| BF16     | 1-8-7  | 2    | 接近 FP32      | 比 FP16 粗   | 大模型训练常用，稳        |
| INT8     | 无      | 1    | scale 决定真实范围 | 均匀整数格点     | 常用于推理量化          |
| INT4     | 无      | 0.5  | scale 决定真实范围 | 更少均匀格点     | 极致压缩量化           |
| FP8 E4M3 | 1-4-3  | 1    | 小            | FP8 中较细    | 常用于权重/激活         |
| FP8 E5M2 | 1-5-2  | 1    | 大            | 更粗         | 常用于梯度/反向         |
| FP8 E8M0 | 0-8-0  | 1    | 只表达 2 的幂     | 无 mantissa | 常用作 scale        |
| FP4 E2M1 | 1-2-1  | 0.5  | 很小           | 很粗         | 必须依赖 scale       |


一个非常实用的判断：

- 需要训练稳定性，优先 BF16/FP16 混合精度。
- 需要更高吞吐和更低显存，考虑 FP8，但要依赖成熟 kernel 和 scaling recipe。
- 需要极致推理压缩，考虑 FP4/NVFP4，但必须关注精度损失和硬件支持。



## 16. PyTorch 中如何自查



### 16.1 查 element_size

```python
import torch

for dtype in [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.int64,
    torch.int32,
    torch.int8,
    torch.uint8,
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e8m0fnu,
    torch.float4_e2m1fn_x2,
]:
    x = torch.empty((), dtype=dtype)
    print(dtype, x.element_size())
```

本地 `torch 2.11.0+cu129` 输出要点：

```text
torch.float32             4
torch.float16             2
torch.bfloat16            2
torch.int64               8
torch.int32               4
torch.int8                1
torch.uint8               1
torch.bool                1
torch.float8_e4m3fn       1
torch.float8_e5m2         1
torch.float8_e8m0fnu      1
torch.float4_e2m1fn_x2    1
```

最后一个返回 1 的原因是 packed：一个 byte 中有两个 FP4。

### 16.2 查 finfo

```python
import torch

for dtype in [
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]:
    f = torch.finfo(dtype)
    print(dtype, f.bits, f.eps, f.tiny, f.min, f.max)
```

注意：低精度 shell dtype 的 `finfo` 或转换操作可能不是所有版本都完整支持。例如本地环境里 `torch.float4_e2m1fn_x2` 的 `torch.finfo()` 会报 `NotImplementedError`。

### 16.3 查整数范围

```python
import torch

for dtype in [
    torch.int8,
    torch.uint8,
    torch.int16,
    torch.int32,
    torch.int64,
]:
    info = torch.iinfo(dtype)
    print(dtype, info.bits, info.min, info.max)
```

`torch.iinfo` 用于整数 dtype，`torch.finfo` 用于浮点 dtype。`torch.bool` 不支持 `torch.iinfo`，因为它不是整数范围意义上的 numeric integer。

## 17. 训练和推理里怎么选



### FP32

适合：

- debug
- 小模型
- 对数值稳定性要求极高的部分

不适合：

- 大模型全量训练，显存和吞吐成本太高



### FP16

适合：

- GPU 混合精度训练
- 推理
- 对局部精度有一定要求，且动态范围可控

风险：

- 梯度 overflow/underflow
- 需要 loss scaling 或框架自动处理



### BF16

适合：

- LLM 训练主路径
- 希望减少 loss scaling 复杂度
- A100/H100/B200 等硬件支持良好的环境

风险：

- 局部精度比 FP16 粗



### FP8

适合：

- 有成熟 FP8 kernel 的 Transformer 层
- Hopper、Ada、Blackwell 等支持路径
- 追求吞吐、显存、通信带宽收益

风险：

- 不是所有 op 都适合 FP8
- scale 策略和 amax 统计很关键
- shape、对齐、kernel 支持可能有限制



### MXFP8

适合：

- Blackwell 上的低精度训练
- 希望比 per-tensor FP8 scaling 更细粒度

风险：

- 依赖硬件和框架支持
- scale layout、block 方向、转置语义会影响实现



### FP4/MXFP4/NVFP4

适合：

- 极致低显存推理
- Blackwell 上的 NVFP4 训练/推理实验路径
- 对吞吐和显存非常敏感的模型部署

风险：

- 单值信息量极低
- 量化误差主要由 scale 策略决定
- attention、softmax、norm 等敏感部分常常需要保留更高精度



### INT8/INT4

适合：

- 推理量化
- 权重-only quantization
- 激活量化配合校准数据
- embedding/table、KV cache 等特定压缩路径

风险：

- 真实值解释依赖 scale/zero_point
- per-tensor scale 容易被 outlier 影响，per-channel/per-group scale 更常见
- 低 bit 整数量化通常需要校准、误差分析和专门 kernel



## 18. 一张总表


| 名称                | 是否 PyTorch dtype               | 是否单纯元素格式  | 是否带 block scale recipe | 每值主存储                             | 额外元数据                                        |
| ----------------- | ------------------------------ | --------- | ---------------------- | --------------------------------- | -------------------------------------------- |
| FP32              | 是，`torch.float32`              | 是         | 否                      | 4 byte                            | 无                                            |
| FP16              | 是，`torch.float16`              | 是         | 否                      | 2 byte                            | 无                                            |
| BF16              | 是，`torch.bfloat16`             | 是         | 否                      | 2 byte                            | 无                                            |
| INT64             | 是，`torch.int64`                | 是         | 否                      | 8 byte                            | 无                                            |
| INT32             | 是，`torch.int32`                | 是         | 否                      | 4 byte                            | 无                                            |
| INT16             | 是，`torch.int16`                | 是         | 否                      | 2 byte                            | 无                                            |
| INT8/UINT8        | 是，`torch.int8` / `torch.uint8` | 是         | 量化时常配合 scale           | 1 byte                            | 普通 int 无；quantized tensor 有 scale/zero_point |
| INT4/UINT4 packed | 部分 shell/packed dtype          | 是，packed  | 量化时常配合 scale           | 0.5 byte/值                        | recipe 决定                                    |
| BOOL              | 是，`torch.bool`                 | 是         | 否                      | 通常 1 byte                         | 无                                            |
| FP8 E4M3          | 是，`torch.float8_e4m3fn` 等      | 是         | 可配合 scale              | 1 byte                            | recipe 决定                                    |
| FP8 E5M2          | 是，`torch.float8_e5m2` 等        | 是         | 可配合 scale              | 1 byte                            | recipe 决定                                    |
| E8M0              | 是，`torch.float8_e8m0fnu`       | 更常作 scale | 是，常见于 MX               | 1 byte                            | 通常自身就是 scale                                 |
| FP4 E2M1          | 是，`torch.float4_e2m1fn_x2`     | 是，packed  | 常配合 scale              | 0.5 byte/值，PyTorch packed byte 边界 | recipe 决定                                    |
| MXFP8             | 不是单个 torch dtype               | 否         | 是                      | 1 byte/值                          | 每 32 值 1 byte scale                          |
| MXFP4             | 不是单个 torch dtype               | 否         | 是                      | 0.5 byte/值                        | 每 32 值 1 byte scale                          |
| NVFP4             | 不是单个 torch dtype               | 否         | 是                      | 0.5 byte/值                        | 每 16 值 1 byte scale + per-tensor FP32        |
| NVFP8             | 不是 PyTorch 官方 dtype 名          | 语境依赖      | 语境依赖                   | 通常 FP8 级别                         | 需要看具体实现                                      |




## 19. 建议后续提问路线

如果要继续深入，建议按这个顺序问：

1. 为什么 BF16 动态范围接近 FP32，但精度比 FP16 粗？
2. E4M3 和 E5M2 的可表示数值集合分别是什么？
3. INT8/INT4 量化里的 scale、zero_point、group size 是怎么工作的？
4. FP8 training 里的 amax、scale、delayed scaling 是怎么工作的？
5. MXFP8 为什么 block size 是 32，scale 为什么用 E8M0？
6. NVFP4 为什么 block size 改成 16，scale 为什么用 E4M3？
7. 真实 LLM 训练里哪些 tensor 能量化到 FP8/FP4/INT8/INT4，哪些必须保留 BF16/FP32？
8. 参数、梯度、optimizer state、activation 的显存应该怎么分别估算？



## 20. QAT：Kimi-K3 和 DSKV4 Pro 如何把 MXFP4 用到训练里

QAT 是 Quantization-Aware Training。它和普通 PTQ 的区别是：

```text
PTQ:
  训练完高精度模型后，再离线量化。
  模型参数本身没有在训练中适应量化误差。

QAT:
  训练阶段就把量化误差放进 forward。
  模型在 SFT/RL 中逐步适应低精度带来的舍入、截断、scale 误差。
```

一个常见 QAT 抽象流程是：

```text
optimizer 维护高精度 master weight
  通常 FP32，也可能有 BF16/FP32 混合实现

forward 时：
  master weight -> quantize 到低精度
  低精度 -> dequantize 到计算 dtype
  用这个“带量化误差”的权重算 loss

backward 时：
  quantize / round / clamp / pack 这些操作不可微
  通常用 STE：
    backward 假装 quantize-dequantize 是 identity
    把 dL/d(dequant_weight) 近似传回 master weight

optimizer step：
  更新高精度 master weight

部署时：
  丢掉训练用 master weight
  保存真正 packed 低精度权重 + scale
```

STE 是 Straight-Through Estimator。直觉是：

```text
forward:
  真的使用量化后的权重，让模型看到量化误差。

backward:
  不去求 round/pack 的真实梯度。
  近似认为量化操作的梯度是 1，让梯度直接穿过去。
```



### 20.1 Kimi-K3 的 QAT

Kimi-K3 的公开描述可以拆成四点：

```text
1. 只量化 MoE expert weights。
   MoE experts 占模型参数内存的大头，所以优先量化它们。

2. expert weights 用 MXFP4。
   也就是 packed FP4 E2M1 + E8M0 block scale。

3. activations computed in MXFP8。
   expert 计算相关 activation 走 MXFP8。

4. 非 expert 组件保留高精度。
   attention projections、latent MoE projections、shared experts、MoE routers
   都不跟 routed experts 一起降到 MXFP4。
```

它的 QAT 覆盖整个 post-training：

```text
SFT 阶段：
  就开始按 MXFP4 weight / MXFP8 activation 的方案训练。

RL 阶段：
  rollout 和 training 使用同一套 quantization scheme。
```

这句话很关键：

```text
rollout and training share the same quantization scheme
```

RL 里如果 rollout 用高精度模型采样，但 training/update 用量化模型算 loss，或者反过来，就会出现 train-inference mismatch。Kimi 的思路是让采样行为和训练时看到的模型尽量一致：

```text
rollout 看到的 logits:
  带 MXFP4/MXFP8 量化影响

training 计算 logprob/loss:
  也带同一套量化影响

部署:
  仍然是这套 MXFP4/MXFP8 行为
```

Kimi 报告没有展开每个 backward kernel 的实现细节，所以这里不能断言它是不是和 DSKV4 一样“FP4 先 dequant 到 FP8 再训练”。保守理解是：

```text
master expert weight
  -> MXFP4 fake/native quant
  -> dequant 到训练计算路径可用的格式
  -> activation 按 MXFP8 路径参与 forward
  -> loss
  -> 梯度通过 STE 或等价近似回到 master weight
```

也就是说，Kimi 的重点不是“训练全程只保存 4-bit 权重”，而是“训练 forward 里暴露部署时的 MXFP4/MXFP8 误差，让模型适应”。

### 20.2 DSKV4 Pro 的 QAT

DeepSeek-V4 / DSKV4 Pro 的公开描述更细。它对两类东西做 FP4/MXFP4 QAT：

```text
1. MoE expert weights
   这是 GPU 参数显存占用的大头。

2. CSA indexer 的 QK path
   QK activations 会被 cache、load，并且完全在 FP4 下相乘。
   目标是加速长上下文 attention score 计算。
```

另外它还把 index scores `I[:, :]` 从 FP32 量化到 BF16。论文描述这能让 top-k selector 加速，同时保持较高 KV entry recall。

#### 20.2.1 DSKV4 的 MoE expert weight QAT

DSKV4 对 MoE expert weights 的训练流程可以写成：

```text
FP32 master weight
  -> quantize 到 FP4/MXFP4
  -> dequantize 回 FP8 E4M3
  -> 用已有 FP8 training framework 计算 forward/backward
  -> 梯度直接回传到 FP32 master weight
```

这里最重要的是这句：

```text
FP4-to-FP8 dequantization is lossless
```

它的意思不是“FP4 比 FP8 信息更多”，而是：

```text
FP4 E2M1 本身只有 4 bit，非常粗；
但 MXFP4 还带有每 32 个值一个 E8M0 scale；
把 FP4 值乘上这些 fine-grained scale 后，
在一定 scale ratio 条件下，可以无额外损失地放进 FP8 E4M3 的表示范围。
```

DeepSeek 的说法是：FP8 E4M3 比 FP4 E2M1 多 2 个 exponent bit，动态范围更大。只要同一个 FP8 quantization block 内，不同 FP4 sub-block 的 scale 比值没有超过阈值，FP4 sub-block 的 fine-grained scale 信息就能被 FP8 的更大动态范围吸收。它们经验验证当前权重满足这个条件。

所以训练时它可以复用 FP8 训练框架：

```text
forward 看到的是从 FP4/MXFP4 转过来的 FP8 weight；
backward 也是对这个 FP8 weight 计算梯度；
梯度通过 STE 直接传回 FP32 master weight。
```

这避免了两件事：

```text
1. 不需要单独写一整套 FP4 backward training framework。
2. 不需要在 backward 里重新量化转置权重。
```



#### 20.2.2 DSKV4 的 CSA indexer QK path

DSKV4 还把 CSA indexer 里的 QK path 做 FP4 QAT：

```text
Q/K activations:
  cache 时是 FP4；
  load 时是 FP4；
  乘法时也是 FP4。

index scores I:
  从 FP32 降到 BF16。
```

这部分目标不是 MoE 权重显存，而是长上下文下 attention indexer 的读写和 score 计算成本。长上下文场景里 QK path 的 cache/load 很频繁，把它降到 FP4 可以直接减少 memory traffic。

#### 20.2.3 DSKV4 的 rollout / inference

训练有 backward，所以 DSKV4 选择：

```text
FP4/MXFP4 -> dequant 到 FP8 -> 复用 FP8 training framework
```

但 rollout 和 inference 没有 backward，因此可以直接用 native FP4 quantized weights：

```text
rollout / inference:
  直接读取 packed FP4/MXFP4 weight；
  用推理 kernel 消费 packed FP4 和 scale；
  不走训练时的 simulated quantization。
```

这样 RL sampling 行为和线上部署更一致，同时也能拿到真实的显存和带宽收益。

### 20.3 Kimi-K3 和 DSKV4 Pro 的差异

可以用这张表理解：


| 项目               | Kimi-K3                                   | DSKV4 Pro                                                        |
| ---------------- | ----------------------------------------- | ---------------------------------------------------------------- |
| 量化对象             | MoE expert weights                        | MoE expert weights + CSA indexer QK path                         |
| expert weight 格式 | MXFP4                                     | FP4/MXFP4                                                        |
| activation       | expert activations computed in MXFP8      | MoE 训练中 FP4 -> FP8 计算；QK path 使用 FP4                             |
| 非 expert 模块      | attention、shared experts、router 等保持高精度    | 大部分非 expert 参数是 FP8 mixed，另有 index score FP32 -> BF16            |
| QAT 覆盖阶段         | post-training 全程，SFT + RL                 | post-training QAT                                                |
| RL rollout       | rollout 和 training 共享 quantization scheme | rollout/inference 直接用 native FP4 quantized weights               |
| backward 描述      | 公开描述未展开，按 QAT 常规应使用 STE 或等价近似             | 明确说对 forward 中同一个 FP8 weight 求梯度，并直接传回 FP32 master weight，等价 STE |
| 训练框架复用           | 未明确说明                                     | 明确复用已有 FP8 training framework                                    |




### 20.4 梯度到底怎么流

以 MoE expert weight 为例，QAT 的梯度可以这样理解：

```text
真实数学上：
  quantize() 里面有 round/clamp/pack
  这些操作几乎处处不可导或梯度为 0

工程上：
  forward 真的做 quantize-dequantize
  backward 假装这个操作对梯度是透明的
```

伪代码：

```python
w_master = fp32_parameter

# forward
w_q = quantize_mxfp4(w_master)       # round / clamp / pack / scale
w_compute = dequantize(w_q)          # Kimi: 计算格式未完全公开；DSKV4: dequant 到 FP8
y = matmul_or_moe(x, w_compute)
loss = criterion(y, target)

# backward, STE intuition
# dloss/dw_compute 近似直接当成 dloss/dw_master
loss.backward()
optimizer.step()
```

这不是说量化操作真的可导，而是说训练时故意使用一个有偏但实用的梯度估计。模型会逐步学到：

```text
哪些权重更新方向在量化后仍然有效；
哪些权重变化会被 FP4/MXFP4 的刻度吃掉；
如何在低精度误差存在时保持输出分布稳定。
```



### 20.5 STE 在代码里通常怎么写

最常见的 STE 写法是：

```python
w_qdq = dequantize(quantize(w))
w_ste = w + (w_qdq - w).detach()
```

这行代码的行为要分 forward 和 backward 看：

```text
forward:
  detach() 不改变数值。
  w_ste = w + (w_qdq - w) = w_qdq
  所以前向真的使用量化-反量化后的权重。

backward:
  detach() 里面的路径不传梯度。
  d(w_ste) / d(w) = d(w) / d(w) + 0 = 1
  所以后向梯度像 identity 一样直接传回 w。
```

因此在这个最朴素的 STE 版本里，下列操作都不参与真实梯度链路：

```text
compute_e8m0_block_scale
round / clamp
quantize_to_fp4_e2m1
pack
dequantize_fp4_e2m1
乘 scale
```

它们只决定 forward 用什么数值；backward 时，autograd 直接把 `dL/d(w_ste)` 当成 `dL/d(w_master)`。

以 MXFP4 为例，可以写成：

```python
def mxfp4_fake_quant_ste(w):
    # 1. 每 32 个值算 block scale，比如 E8M0 power-of-two scale
    scale = compute_e8m0_block_scale(w, block_size=32)

    # 2. 映射到 FP4 E2M1 code
    q_code = quantize_to_fp4_e2m1(w / scale)

    # 3. dequant 回训练计算 dtype，比如 FP8/BF16
    w_qdq = dequantize_fp4_e2m1(q_code) * scale

    # 4. STE
    return w + (w_qdq - w).detach()
```

这意味着：

```text
forward:
  loss 是用 w_qdq 算出来的。
  模型真的看到量化误差。

backward:
  不对 quantize/dequantize/pack/scale 这一整条链路求真实梯度。
  直接把量化 forward 下产生的梯度传回 master weight。
```

一个容易误解的点是：STE 不是完全忽略量化。

```text
不是：
  loss = f(w_master)

而是：
  loss = f(w_qdq)
```

所以梯度虽然直通回 `w_master`，但这个梯度来自“量化后的 forward 行为”。这也是 QAT 有意义的原因。

scale 是否有梯度取决于实现：

```text
常见简单实现：
  scale 由 amax/statistics/rule 计算；
  scale 不作为可学习参数；
  scale 路径也被 detach。

更复杂实现：
  scale 可以学习；
  需要给 scale 单独设计近似梯度；
  例如 LSQ 一类方法。
```

DeepSeek-V4 描述里的：

```text
gradients are computed with respect to the same FP8 weights in the forward pass
and directly propagated back to the FP32 master weights
```

翻成代码语义，就很像：

```python
w_fp4 = quantize_mxfp4(w_master)
w_fp8 = dequant_fp4_to_fp8(w_fp4)
w_train = w_master + (w_fp8 - w_master).detach()
```

然后用 `w_train` 跑 forward。loss 对 `w_train` 的梯度最后近似回到 `w_master`。

### 20.6 能看出是在 H 卡还是 B 卡上训练的吗

仅凭这两段公开描述，不能确定训练硬件。

可以做的合理判断是：

```text
Kimi-K3:
  描述了 MXFP4/MXFP8 QAT 和 rollout/training scheme 对齐；
  没有明确训练硬件；
  不能仅凭这段判断是 H100/H200 还是 B200/GB200。

DSKV4 Pro:
  明确说 MoE expert weight 训练时 FP4 -> FP8，复用 FP8 training framework；
  这说明它不一定依赖 Blackwell 原生 FP4 backward/training；
  H100/H200 这类支持 FP8 Tensor Core 的 Hopper 机器理论上也能承载这种 QAT 训练路径；
  native FP4 更明确出现在 rollout / inference 这种无 backward 阶段。
```

也就是说：

```text
Hopper/H 卡：
  可以做“FP4 作为量化约束，训练计算复用 FP8/BF16”的 QAT。

Blackwell/B 卡：
  更适合 native FP4/MXFP4 inference，甚至部分低精度训练路径；
  但公开描述本身不足以证明 Kimi 或 DSKV4 的 post-training 一定在 B 卡完成。
```

最保守的一句话：

```text
这些描述能说明它们的部署目标强烈面向 FP4/MXFP4；
但不能单独证明 post-training 硬件是 H 卡还是 B 卡。
```

## 21. Miles / Kimi-K2.5 INT4 QAT 全流程

[https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/int4/readme.md](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/int4/readme.md)

[https://www.zhihu.com/question/1969558404759544488/answer/1970539327902679960](https://www.zhihu.com/question/1969558404759544488/answer/1970539327902679960)

这一节只讨论 Miles 里已经落地的 INT4 QAT 路径，先不讨论 NVFP4。

Miles 的 INT4 QAT 可以理解成一条 RL 低比特闭环：

```text
训练侧:
  BF16 runtime/model weight
  -> INT4 fake quant / dequant
  -> BF16 GEMM forward/backward
  -> STE 让梯度穿过 fake quant
  -> optimizer 更新 FP32 main_param

权重同步 / 导出:
  当前训练权重
  -> real INT4 quantization
  -> pack / permute 成推理 kernel 需要的格式
  -> 发给 SGLang rollout engine

rollout / inference:
  SGLang 加载 packed INT4 weight
  -> Marlin W4A16 MoE kernel
  -> INT4 weight x BF16 activation
  -> BF16 Tensor Core 计算
  -> 采样结果回流训练
```

### 21.1 先纠正“BF16 主权重”这个说法

中文 INT4 文档里会写“训练侧维护 BF16 主权重”。这句话按工程语境可以理解，但按源码更严谨地说：

```text
BF16:
  是 Megatron forward/backward 里的 runtime/model parameter dtype。

FP32 main_param:
  是 Megatron optimizer 维护和更新的高精度主副本/分片。
```

Miles 的 Megatron 参数默认逻辑里有：

```python
args.bf16 = not args.fp16
```

也就是说不显式开 `fp16` 时，Megatron 后端默认走 BF16。Miles 代码里还说明 Megatron 的 `Float16Module` 会把浮点参数 cast 到 BF16/FP16。

但 optimizer 侧又能看到 `main_param`。Miles 的 checksum / witness 相关代码直接把它称为：

```text
fp32_main_param
fp32 main weights
```

所以本节之后统一用这个更准确的表达：

```text
BF16 model/runtime weight + FP32 optimizer main_param
```

而不是简单说“BF16 master weight”。

### 21.2 训练侧 fake quant 到底做了什么

Miles 的 INT4 QAT 开关来自 runtime env：

```bash
OPEN_TRAINING_INT4_FAKE_QAT_FLAG=1
OPEN_TRAINING_INT4_GROUP_SIZE=32 或 128
```

Kimi-K2.5 recipe 里 group size 是 32；其他一些模型常见配置是 128。

核心逻辑被挂在 TransformerEngine 的 `TEGroupedLinear._get_weight_tensors()` 里。打开 env 后，它不是把参数真的改成 packed INT4，而是在每次取 weight tensor 做 grouped GEMM 前，替换成 fake quant 后的 tensor：

```python
weight_tensors = [
    fake_int4_quantization_ste(w, group_size)
    for w in weight_tensors
]
```

fake quant 的数学过程是：

```text
对每个 block/group：
  1. 取 max(abs(w)) 得到 block_max
  2. symmetric INT4 下 scale = max(block_max / 7, 1e-5)
  3. q = round(w / scale)
  4. q = clamp(q, -7, 7)
  5. w_qdq = q * scale
```

这里的 `q` 模拟 INT4 整数格点，但训练 forward 返回的是 `w_qdq`，它仍然是浮点 tensor，通常是 BF16 dtype。

也就是说：

```text
训练侧 fake quant:
  不把参数物理存成 INT4；
  不 pack nibble；
  只让 forward 看到“被 INT4 量化误差污染过”的浮点权重。
```

### 21.3 STE 梯度怎么走

INT4 量化里的 `round` / `clamp` 对训练很麻烦：

```text
round:
  阶梯函数，几乎处处导数是 0。

clamp:
  超出范围后导数也会被截断。
```

如果严格按数学导数反传，梯度会很难穿过量化层，底层权重基本学不动。

Miles 的做法是 STE。源码里的 backward 非常直接：

```python
def backward(ctx, grad_output):
    return grad_output, None
```

含义是：

```text
forward:
  用 w_qdq 算，所以 loss 来自量化后的模型行为。

backward:
  不给 scale/group_size 求梯度；
  把 dLoss/d(w_qdq) 近似当成 dLoss/dw；
  直接传回原始训练权重。
```

这不是“忽略量化”。量化已经影响了 forward 的输出，所以 loss 和梯度方向都来自量化后的网络。STE 只是说在过 `round/clamp` 这道不可导关口时，把它近似成恒等函数。

可以写成这个直觉代码：

```python
w_qdq = dequantize_int4(quantize_int4(w))
w_train = w + (w_qdq - w).detach()
```

forward 看到 `w_train == w_qdq`，backward 看到 `d w_train / d w == 1`。

### 21.4 权重更新阶段：fake quant 变成 real quant

训练侧 fake quant 不产生推理需要的 packed 权重。真正的 INT4 packing 发生在权重导出 / Megatron-to-HF / weight update 阶段。

Miles 的 INT4 quantizer 会对需要量化的 `.weight` 做：

```text
1. fake_int4_quant_cuda 得到 q、scale、zero_point
2. symmetric 情况下 w = q * scale，重建一次量化后权重
3. 再按 scale 重新得到 INT4 code
4. pack_to_int32，把 8 个 4-bit 值 pack 到 1 个 int32
5. 生成：
     xxx.weight_packed
     xxx.weight_scale
     xxx.weight_shape
     可选 xxx.weight_zero_point
```

这里和训练 fake quant 最大区别是：

```text
训练 fake quant:
  产物仍是浮点 tensor，服务于 backward。

权重更新 real quant:
  产物是推理侧真实要读的 packed INT4 storage，服务于 rollout/inference。
```

### 21.5 SGLang rollout 阶段：Marlin W4A16

rollout 侧加载的是 INT4 actor checkpoint。Kimi-K2.5 文档里明确是：

```text
actor:
  Kimi-K2.5-int4

reference:
  Kimi-K2.5-bf16
```

推理侧是 W4A16：

```text
W4:
  weight 是 4-bit INT4 packed storage。

A16:
  activation 是 BF16。
```

在 H200 / H100 这类 Hopper 卡上，INT4 的主要收益不是“原生 INT4 Tensor Core 算得更快”，而是：

```text
1. 权重显存更小；
2. MoE expert 权重加载带宽更低；
3. 大模型 rollout 更容易压到单机/更少节点；
4. 减少跨机通信和权重同步压力。
```

Marlin W4A16 kernel 的常见理解是：

```text
packed INT4 weight
  -> kernel 内部 unpack/dequant
  -> BF16 activation 相乘
  -> BF16 Tensor Core 路径完成 GEMM
```

所以 INT4 QAT 在 H200 上的价值重点是：

```text
让模型适应 INT4 权重量化误差；
让 rollout 能用更小权重；
让训练侧和 rollout 侧看到的权重误差尽量接近。
```

而不是把训练和推理都变成真正的 INT4 算术。

### 21.6 Kimi-K2-Thinking / Kimi-K2.5 的 INT4 checkpoint 例子

Kimi-K2-Thinking 和 Kimi-K2.5 看到的 MoE expert 权重可以是这种格式：

```text
model.layers.8.mlp.experts.8.gate_proj.weight_packed  [2048, 896]  I32
model.layers.8.mlp.experts.8.gate_proj.weight_scale   [2048, 224]  BF16
```

这不是普通 `int32` 权重，而是 packed INT4。

如果按 `gate_proj` 的逻辑 weight shape 理解，它原本大概率是：

```text
[2048, 7168]
```

因为：

```text
896 * 8 = 7168
```

每个 `I32` 里面 pack 了 8 个 4-bit code：

```text
8 * 4 bit = 32 bit
```

再看 scale：

```text
224 * 32 = 7168
```

说明它按 K 维每 32 个元素一组 scale：

```text
group size = 32
scale dtype = BF16
```

所以这个 checkpoint 格式可以概括成：

```text
Kimi-K2-Thinking / Kimi-K2.5:
  weight_packed: I32 container，每个 int32 pack 8 个 INT4
  weight_scale : BF16 scale，每 32 个 K 维元素一个 scale
  compute path : rollout 侧通常是 Marlin W4A16
```

和 Kimi-K3 的 MXFP4 不同：

```text
Kimi-K3:
  weight_packed 通常是 U8 / I8 container，每个 byte pack 2 个 FP4 E2M1 code
  weight_scale  通常是 U8 / E8M0 语义，每 32 个 K 维元素一个 scale
  语义是 MXFP4，不是 INT4
```

因此一个简单记法是：

```text
Kimi-K2-Thinking / Kimi-K2.5:
  INT4 + BF16 scale，常见 I32 packed。

Kimi-K3:
  MXFP4，FP4 E2M1 code + E8M0 scale，常见 U8/I8 packed。
```

### 21.7 为什么 QAT 和 rollout 必须配套

中文文档里有两个很关键的对比：

```text
开启 QAT INT4 训练，但 rollout 用 BF16：
  也会出现明显 mismatch。

关闭 QAT，直接 INT4 rollout：
  属于 PTQ 风格，模型没适应量化噪声，也会 mismatch。
```

这说明 INT4 QAT 的核心不是“低比特一定更准”，而是要让训练和 rollout 形成同一套数值假设：

```text
训练 forward:
  模拟 INT4 权重误差。

rollout:
  真实使用 INT4 权重误差。
```

如果训练时模型已经学会补偿 INT4 误差，rollout 却突然换回 BF16，这个补偿本身也会变成扰动。反过来，如果训练一直 BF16，rollout 突然 INT4，模型又没学会适应量化误差。

所以最核心的一句话是：

```text
QAT 的目标不是让 INT4 tensor 参与反向传播；
而是让训练 forward 中的数值误差和部署/rollout 中的数值误差对齐。
```

### 21.8 和 Kimi-K3 / DSKV4 的 FP4 QAT 有什么可迁移之处

Miles INT4 QAT 对理解 Kimi-K3 / DSKV4 的 FP4 QAT 很有帮助，但只能迁移流程，不要迁移数值格式。

可迁移的是：

```text
1. 训练保留高精度可更新权重；
2. forward 插入低比特 QDQ；
3. backward 用 STE 或等价近似；
4. rollout 使用真实 packed 低比特权重；
5. 每次权重同步都重新 quantize/pack；
6. 目标是减少 train/rollout mismatch。
```

不能直接迁移的是：

```text
INT4:
  q 是整数 code；
  scale 通常是普通浮点 scale；
  可有 zero_point；
  group size 常见 32/128；
  推理是 Marlin W4A16。

MXFP4 / NVFP4:
  q 是 FP4 E2M1 code；
  scale 可能是 E8M0 或 FP8 scale；
  通常没有普通 INT zero_point 语义；
  block size / scale layout 不一样；
  B 卡上可能走 native FP4 Tensor Core 路径。
```

所以对后续讨论最有用的抽象是：

```text
Miles INT4 QAT 是“低比特 QAT 闭环”的清晰样板；
但 Kimi-K3 / DSKV4 的 MXFP4 需要替换 quantizer、scale 语义、packing layout 和 kernel。
```

## 参考资料

- PyTorch Tensor Attributes / dtype 官方文档：[https://docs.pytorch.org/docs/2.13/tensor_attributes.html](https://docs.pytorch.org/docs/2.13/tensor_attributes.html)
- ONNX Float8 技术说明：[https://onnx.ai/onnx/technical/float8.html](https://onnx.ai/onnx/technical/float8.html)
- NVIDIA Transformer Engine FP8/FP4 primer：[https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- NVIDIA Transformer Engine common recipe API：[https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html)
- NVIDIA cuDNN frontend：MXFP8/NVFP4 scale layout：[https://nvidia.github.io/cudnn-frontend/mxfp8-scale-factor-128x4-layout/](https://nvidia.github.io/cudnn-frontend/mxfp8-scale-factor-128x4-layout/)
- NVIDIA Technical Blog：NVFP4：[https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
- Kimi-K3 技术报告摘要：[https://www.alphaxiv.org/abs/2607.24653](https://www.alphaxiv.org/abs/2607.24653)
- Kimi-K3 HuggingFace README：[https://huggingface.co/moonshotai/Kimi-K3/blob/main/README.md](https://huggingface.co/moonshotai/Kimi-K3/blob/main/README.md)
- DeepSeek-V4 HuggingFace paper page：[https://huggingface.co/papers/2606.19348](https://huggingface.co/papers/2606.19348)
- DeepSeek-V4-Pro HuggingFace README：[https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/README.md](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/README.md)
- LMSYS / Miles INT4 QAT blog：[https://www.lmsys.org/blog/2026-01-26-int4-qat/](https://www.lmsys.org/blog/2026-01-26-int4-qat/)
- 中文 INT4 QAT 说明：[https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/int4/readme.md](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/slime/int4/readme.md)
- Miles INT4 QAT 文档：`/mnt/shared-storage-user/huanghaian/code/slime_package/miles/docs/advanced/int4-qat.md`
- Miles Kimi-K2.5 recipe：`/mnt/shared-storage-user/huanghaian/code/slime_package/miles/scripts/run-kimi-k25.sh`
- Miles INT4 fake quant patch：`/mnt/shared-storage-user/huanghaian/code/slime_package/miles/docker/npu_patch/megatron_common.patch`
- Miles INT4 real quant / packing：`/mnt/shared-storage-user/huanghaian/code/slime_package/miles/miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_compressed_tensors.py`

