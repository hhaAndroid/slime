# Rollout 拓扑：start_rollout_servers

这篇只分析非 PD 分离场景下的 rollout 拓扑。先不考虑 prefill/decode 分离、encoder、external rollout、多模型 reward 等扩展分支，避免把主线搞混。

核心代码：

- `slime/ray/rollout.py`
- `slime/backends/sglang_utils/sglang_engine.py`
- `slime/backends/sglang_utils/sglang_config.py`

核心入口：

```python
start_rollout_servers(args, pg)
```

它负责把 slime 的 rollout GPU 配置翻译成 SGLang serving 拓扑。重点不是“启动几个进程”这么简单，而是决定：

- 有几个 router；
- 有几个 `RolloutServer`；
- 每个 `RolloutServer` 下面有几个 `ServerGroup`；
- 每个 `ServerGroup` 里有多少个 Ray `SGLangEngine` actor；
- 哪些 actor 组成同一个 SGLang distributed engine；
- 哪些 node-rank 0 actor 对外注册到 router。

## 三层结构

目前先按三层理解：

```text
RolloutServer
  -> ServerGroup
    -> SGLangEngine Ray actor
```

如果不用 `--sglang-config`，默认只有一个 model：

```text
servers = {
  "default": RolloutServer(...)
}
```

默认的 `RolloutServer` 里也只有一个 `regular` 类型的 `ServerGroup`：

```text
RolloutServer(default)
  -> ServerGroup(regular)
    -> 多个 SGLangEngine Ray actor
```

### 第一层：RolloutServer

`RolloutServer` 可以理解为“一个模型的 serving 单元”。

它有自己的 router：

```python
router_ip, router_port = _start_router(...)
```

默认情况下只有一个模型，所以只有一个：

```text
RolloutServer(default)
```

它保存：

```python
server_groups
router_ip
router_port
model_name
update_weights
```

在默认单模型场景下，`model_name` 就是 `"default"`，这个 server 会接收 actor 的权重更新。

### 第二层：ServerGroup

`ServerGroup` 是同构的一组 SGLang engine 分片。这里的同构指：

- 同一个 `worker_type`，默认是 `regular`；
- 同一个 `num_gpus_per_engine`；
- 同一组 SGLang overrides；
- 同一个 router。

非 PD 默认场景下，只有一个 regular group：

```python
ServerGroup(
    worker_type="regular",
    num_gpus_per_engine=args.rollout_num_gpus_per_engine,
    gpu_offset=0 或 actor_num_gpus,
)
```

它最关键的两个列表是：

```python
all_engines
engines
```

区别是：

```text
all_engines:
  所有 Ray SGLangEngine actors，包括多节点 engine 的非 node-rank 0 分片。

engines:
  每个 SGLang distributed engine 的 node-rank 0 actor。
  只有这些会对外注册到 router。
```

代码里：

```python
@property
def nodes_per_engine(self):
    return max(1, self.num_gpus_per_engine // self.args.num_gpus_per_node)

@property
def engines(self):
    return self.all_engines[:: self.nodes_per_engine]
```

所以 `engines` 不是所有 Ray actor，而是 node-rank 0 actor 的抽样结果。

### 第三层：SGLangEngine Ray actor

`SGLangEngine` 是 Ray actor，但它不等价于“一张 GPU”。

更准确地说：

```text
一个 SGLangEngine Ray actor = 一个 SGLang distributed engine 在一个节点上的分片
```

如果一个 SGLang engine 只用本节点 GPU，那么它只需要 1 个 Ray actor。

如果一个 SGLang engine 跨 4 个节点，那么它需要 4 个 Ray actor。每个 actor 代表这个 engine 在一个节点上的分片；在 8 卡节点上，这个分片内部会使用本节点的 8 张卡。

```text
node_rank 0
node_rank 1
node_rank 2
node_rank 3
```

这 4 个 actor 启动出来的 SGLang server process 通过同一组 distributed 参数组成一个完整 engine。

## actor 数量怎么算

`start_rollout_servers()` 里创建 group 时有这段：

```python
gpus_per_engine = group_cfg.num_gpus_per_engine
num_gpu_per_engine_local = min(gpus_per_engine, args.num_gpus_per_node)
num_engines = group_cfg.num_gpus // num_gpu_per_engine_local
```

这里的 `num_engines` 名字容易误导。它实际表示：

```text
这个 ServerGroup 要创建多少个 Ray SGLangEngine actors
```

不是 HTTP engine 数。

公式可以记成：

```text
Ray actor 数 = group 总 GPU 数 / 每个 actor 在本节点负责的 GPU 数

每个 actor 在本节点负责的 GPU 数 = min(num_gpus_per_engine, num_gpus_per_node)
```

如果 `num_gpus_per_engine <= num_gpus_per_node`，一个 SGLang engine 不跨节点，1 个 Ray actor 就是 1 个 HTTP engine。

如果 `num_gpus_per_engine > num_gpus_per_node`，一个 SGLang engine 跨多个节点，多个 Ray actor 才组成 1 个 HTTP engine。

## 多个 Ray actor 如何组成一个 SGLang engine

靠 SGLang 的 distributed 启动参数：

```text
tp_size
nnodes
node_rank
dist_init_addr
base_gpu_id
```

slime 会给同一个 SGLang distributed engine 的多个 Ray actor 分配相同的：

```text
tp_size
nnodes
dist_init_addr
```

同时给它们不同的：

```text
node_rank
host
port
nccl_port
base_gpu_id
```

在 `_compute_server_args()` 里：

```python
_gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
nnodes = max(1, _gpus_per_engine // args.num_gpus_per_node)
node_rank = rank % nnodes

kwargs = {
    "tp_size": _gpus_per_engine // args.sglang_pp_size,
    "nnodes": nnodes,
    "node_rank": node_rank,
    "dist_init_addr": dist_init_addr,
    "base_gpu_id": base,
}
```

`dist_init_addr` 在 `_allocate_rollout_engine_addr_and_ports_normal()` 里分配。对于跨节点 engine，它会每 `num_node_per_engine` 个 actor 分成一组，并给这一组同一个地址：

```python
num_node_per_engine = _gpus_per_engine // args.num_gpus_per_node

if local_rank % num_node_per_engine == 0:
    dist_init_addr = f"{get_addr()}:{get_port(...)}"
    for i in range(num_node_per_engine):
        addr_and_ports[rank + i]["dist_init_addr"] = dist_init_addr
```

因此，同一个 32-GPU engine 的 4 个 Ray actor 会类似这样。这里的 `actor_global_rank` 是 slime 给 `SGLangEngine` Ray actor 的编号，不是 GPU rank；真正传给 SGLang 分布式初始化、用于区分节点的是 `node_rank`。

```text
actor_global_rank 0:
  tp_size=32, nnodes=4, node_rank=0, dist_init_addr=A:PORT

actor_global_rank 1:
  tp_size=32, nnodes=4, node_rank=1, dist_init_addr=A:PORT

actor_global_rank 2:
  tp_size=32, nnodes=4, node_rank=2, dist_init_addr=A:PORT

actor_global_rank 3:
  tp_size=32, nnodes=4, node_rank=3, dist_init_addr=A:PORT
```

SGLang 内部看到同一个 `dist_init_addr`、同一个 `tp_size`、同一个 `nnodes`，再根据不同 `node_rank` 建立分布式实例。

所以不是 Ray 把它们合成一个实例，而是 slime 给这些 Ray actor 启动的 SGLang 进程传了同一组 distributed rendezvous 参数。

## 谁会注册到 router

`SGLangEngine._register_to_router()` 里有这个判断：

```python
if self.node_rank == 0 and self.router_ip and self.router_port:
    ...
```

也就是说，只有 `node_rank=0` 的 SGLangEngine actor 会注册到 router。

所以：

```text
Ray actor 数量 != HTTP worker 数量
```

HTTP worker 数量应该看：

```python
ServerGroup.engines = all_engines[::nodes_per_engine]
```

## 例子 1：Qwen3-30B-A3B

脚本里核心参数：

```text
rollout_num_gpus = 8
rollout_num_gpus_per_engine = 8
num_gpus_per_node = 8
```

计算：

```text
num_gpu_per_engine_local = min(8, 8) = 8
Ray actor 数 = 8 // 8 = 1
nodes_per_engine = 8 // 8 = 1
HTTP worker 数 = 1
```

拓扑：

```text
RolloutServer(default)
  -> ServerGroup(regular, num_gpus_per_engine=8)
    -> all_engines:
       [actor_global_rank 0 / node_rank 0 / 8 GPUs on this node]

    -> engines:
       [actor_global_rank 0 / node_rank 0 / registered to router]
```

这个场景里：

```text
1 个 Ray actor = 1 个 SGLang engine = 1 个 HTTP worker
```

SGLang 参数大致是：

```text
tp_size = 8
nnodes = 1
node_rank = 0
```

## 例子 2：Qwen3-235B-A22B

脚本里核心参数：

```text
rollout_num_gpus = 64
rollout_num_gpus_per_engine = 32
num_gpus_per_node = 8
```

计算：

```text
num_gpu_per_engine_local = min(32, 8) = 8
Ray actor 数 = 64 // 8 = 8
nodes_per_engine = 32 // 8 = 4
HTTP worker 数 = 8 // 4 = 2
```

拓扑：

```text
RolloutServer(default)
  -> ServerGroup(regular, num_gpus_per_engine=32)
    -> all_engines:
       [actor_global_rank 0 / node_rank 0 / 8 GPUs on this node]
       [actor_global_rank 1 / node_rank 1 / 8 GPUs on this node]
       [actor_global_rank 2 / node_rank 2 / 8 GPUs on this node]
       [actor_global_rank 3 / node_rank 3 / 8 GPUs on this node]

       [actor_global_rank 4 / node_rank 0 / 8 GPUs on this node]
       [actor_global_rank 5 / node_rank 1 / 8 GPUs on this node]
       [actor_global_rank 6 / node_rank 2 / 8 GPUs on this node]
       [actor_global_rank 7 / node_rank 3 / 8 GPUs on this node]

    -> engines:
       [actor_global_rank 0 / node_rank 0 / registered to router]
       [actor_global_rank 4 / node_rank 0 / registered to router]
```

也就是：

```text
8 个 Ray SGLangEngine actors
  -> 2 个 SGLang distributed engines
  -> 2 个 HTTP workers 注册到 router
```

分组可以这样看：

```text
SGLang engine 0:
  actor_global_rank 0, node_rank 0, dist_init_addr=A:PORT
  actor_global_rank 1, node_rank 1, dist_init_addr=A:PORT
  actor_global_rank 2, node_rank 2, dist_init_addr=A:PORT
  actor_global_rank 3, node_rank 3, dist_init_addr=A:PORT

SGLang engine 1:
  actor_global_rank 4, node_rank 0, dist_init_addr=B:PORT
  actor_global_rank 5, node_rank 1, dist_init_addr=B:PORT
  actor_global_rank 6, node_rank 2, dist_init_addr=B:PORT
  actor_global_rank 7, node_rank 3, dist_init_addr=B:PORT
```

只有 `actor_global_rank=0` 和 `actor_global_rank=4` 这两个 `node_rank=0` 的 actor 会注册到 router。

## 为什么 Ray actor 只申请 0.2 GPU

`ServerGroup.start_engines()` 创建 Ray actor 时：

```python
num_gpus = 0.2
num_cpus = 0.2
```

这看起来和真实 GPU 使用量不一致，但这是刻意设计。

Ray actor 在这里主要负责 placement：把 `SGLangEngine` actor 放到目标 bundle 上。真正的多 GPU 使用由 SGLang 子进程根据 `tp_size`、`nnodes`、`node_rank`、`base_gpu_id` 自己完成。

所以：

```text
Ray 负责调度位置
SGLang 负责多卡分布式
```

这也是为什么 Ray actor 数量不是 GPU 数量。

## 从 PG 到 SGLang 子进程

这里最容易误解的是：PG、Ray actor 和 SGLang 子进程不是同一层资源模型。

以单节点 8 卡 engine 为例：

```text
rollout_num_gpus = 8
rollout_num_gpus_per_engine = 8
num_gpus_per_node = 8
```

PG 创建时仍然是 8 个 bundle：

```text
bundle 0: {"GPU": 1, "CPU": 1}
bundle 1: {"GPU": 1, "CPU": 1}
...
bundle 7: {"GPU": 1, "CPU": 1}
```

这些 bundle 会先把 8 张卡从 Ray 集群资源里预留出来。后面 `ServerGroup.start_engines()` 不是给每张 GPU 启一个 Ray actor，而是先算：

```python
num_gpu_per_engine = min(num_gpus_per_engine, args.num_gpus_per_node)
```

对于 8 卡单节点 engine，`num_gpu_per_engine = 8`，所以 `len(all_engines) = 1`，只创建 1 个 `SGLangEngine` Ray actor。

这个 actor 会锚定到 engine 的第一张卡对应的 bundle：

```python
gpu_index = gpu_offset + i * num_gpu_per_engine
base_gpu_id = int(reordered_gpu_ids[gpu_index])
placement_group_bundle_index = reordered_bundle_indices[gpu_index]
```

所以 Ray 层面看起来只用了一个 bundle，但它不是说这个 engine 只拥有一张卡。更准确地说：

```text
PG 已经预留了一段连续的 GPU bundle；
Ray actor 只需要绑定到这段 GPU 的起点 bundle；
SGLang 通过 base_gpu_id 知道自己应该从哪张 GPU 开始启动本地 TP 进程。
```

`SGLangEngine` actor 的 Ray 资源申请也能看出这个意图：

```python
num_gpus = 0.2
num_cpus = 0.2
```

这里的 `0.2 GPU` 不是 SGLang 真实使用的 GPU 数，只是为了让 Ray actor 被调度到有 GPU 的 bundle 上，并获得一个合理的 placement anchor。

`placement_group_capture_child_tasks=True` 也不要过度理解。它的含义是：如果这个 Ray actor 内部再创建 Ray child task / actor，默认捕获到同一个 PG 里。SGLang 这里启动的是 `multiprocessing.Process`，不是 Ray child actor，所以 SGLang 子进程占卡的关键不是这个参数。

真正把 8 张卡用起来的是后面的 SGLang 参数。

`SGLangEngine.init()` 里会调用 `_compute_server_args()`：

```python
base_gpu_id = base
tp_size = rollout_num_gpus_per_engine // sglang_pp_size
nnodes = rollout_num_gpus_per_engine // num_gpus_per_node
node_rank = rank % nnodes
```

30B 例子里等价于：

```text
base_gpu_id = 0
tp_size = 8
nnodes = 1
node_rank = 0
```

然后 slime 的 `launch_server_process()` 只启动 1 个 SGLang server 主进程：

```text
Ray SGLangEngine actor
  -> multiprocessing.Process(target=launch_server)
```

但 SGLang server 主进程内部还会启动 scheduler/model-runner 子进程。对于 `tp_size=8, nnodes=1, node_rank=0`，SGLang 会算出本节点的：

```text
tp_rank_range = 0..7
tp_size_per_node = 8
```

然后对每个 `tp_rank` 启动一个 scheduler 进程，并计算：

```python
gpu_id = base_gpu_id + (tp_rank % tp_size_per_node) * gpu_id_step
```

也就是：

```text
tp_rank 0 -> gpu 0
tp_rank 1 -> gpu 1
...
tp_rank 7 -> gpu 7
```

所以完整路径是：

```text
PG:
  预留 8 个 GPU bundle

ServerGroup.start_engines:
  只创建 1 个 Ray SGLangEngine actor
  actor 锚定到 bundle 0
  base_gpu_id = 0

SGLangEngine.init:
  生成 ServerArgs(tp_size=8, nnodes=1, node_rank=0, base_gpu_id=0)

launch_server_process:
  启动 1 个 SGLang server 主进程

SGLang 内部:
  启动 8 个 scheduler/model-runner 子进程
  每个子进程绑定一张 GPU
```

多节点 32 卡 engine 也是同一个逻辑，只是每个节点一个 Ray actor：

```text
rollout_num_gpus_per_engine = 32
num_gpus_per_node = 8
nnodes = 4
tp_size = 32
```

每个节点 actor 都锚定到自己 8 卡分片的第一张 GPU：

```text
actor_global_rank 0, node_rank 0, base_gpu_id = 0, tp_rank 0..7
actor_global_rank 1, node_rank 1, base_gpu_id = 0, tp_rank 8..15
actor_global_rank 2, node_rank 2, base_gpu_id = 0, tp_rank 16..23
actor_global_rank 3, node_rank 3, base_gpu_id = 0, tp_rank 24..31
```

如果每个节点都是 8 卡机器，那么每个节点本地的 `base_gpu_id` 通常都是 0；不同节点通过 `node_rank` 和同一个 `dist_init_addr` 组成同一个 32 卡分布式 SGLang engine。

## start_rollout_servers 的主线流程

非 PD、非 external、默认单模型时，可以简化成：

```text
1. _resolve_sglang_config(args)
   得到一个 default model，一个 regular server group。

2. _start_router(args)
   为 default model 启动一个 router。

3. _make_group(...)
   计算 group 的 Ray actor 数、gpu_offset、rank_offset。

4. ServerGroup.start_engines()
   创建 Ray SGLangEngine actors。

5. _allocate_rollout_engine_addr_and_ports_normal()
   给每个 actor 分配 host/port/nccl_port/dist_init_addr。

6. SGLangEngine.init()
   根据分布式参数启动 SGLang server process。

7. node_rank=0 的 SGLangEngine 注册到 router。

8. 返回 servers 和 init_handles。
```

`RolloutManager.__init__()` 会在之后：

```python
if rollout_init_handles:
    ray.get(rollout_init_handles)
```

等待所有 SGLang engine 初始化完成。

## 当前心智模型

先记住这几个关系：

```text
RolloutServer:
  一个模型，一个 router。

ServerGroup:
  一组同构 SGLang engine 分片。

SGLangEngine Ray actor:
  一个 SGLang engine 在一个节点上的分片。

all_engines:
  所有 Ray actor。

engines:
  node_rank=0 actors，也就是对 router 和权重同步暴露的 engine handles。

dist_init_addr + nnodes + node_rank:
  把多个 Ray actor 组成一个 SGLang distributed engine。
```

最关键的一句话：

```text
Ray actor 数量按“节点分片”算，HTTP worker 数量按“SGLang distributed engine”算。
```

## 补充：tp32 实例和 HTTP URL 数量

一个跨节点的 SGLang distributed engine 通常只有一个对外 HTTP 入口，也就是 `node_rank=0` 对应的 server。

例如一个 32 卡实例：

```text
rollout_num_gpus_per_engine = 32
num_gpus_per_node = 8
nnodes = 4
tp_size = 32
```

这个实例内部有 4 个 Ray `SGLangEngine` actor：

```text
node_rank 0: 启动本节点 scheduler，并对外提供 HTTP URL
node_rank 1: 启动本节点 scheduler，不对外提供 HTTP URL
node_rank 2: 启动本节点 scheduler，不对外提供 HTTP URL
node_rank 3: 启动本节点 scheduler，不对外提供 HTTP URL
```

SGLang 源码里 `node_rank >= 1` 的进程会在 scheduler ready 后阻塞等待，不跑 tokenizer / detokenizer / HTTP serving 这条主入口链路。slime 里 `_register_to_router()` 也只在 `node_rank == 0` 时把 worker URL 注册到 router。

所以：

```text
一个 tp32 distributed engine
  -> 只有 1 个 HTTP URL
```

如果看到 2 个 HTTP URL，通常不是一个 tp32 实例暴露了两个入口，而是启动了两个 tp32 副本。

例如：

```text
rollout_num_gpus = 64
rollout_num_gpus_per_engine = 32
```

等价于：

```text
64 张 rollout GPU / 每个 engine 32 张 GPU = 2 个 SGLang distributed engine
```

拓扑是：

```text
engine 0:
  node_rank 0 -> HTTP URL 0
  node_rank 1 -> scheduler only
  node_rank 2 -> scheduler only
  node_rank 3 -> scheduler only

engine 1:
  node_rank 0 -> HTTP URL 1
  node_rank 1 -> scheduler only
  node_rank 2 -> scheduler only
  node_rank 3 -> scheduler only
```

router 看到的是两个 `node_rank=0` worker，所以可以在两个 tp32 副本之间做请求分发。单个 tp32 实例内部仍然只有 `node_rank=0` 接请求，然后通过 TP 通信驱动所有 rank 参与计算。

## 补充：tp32 内部 scheduler 在做什么

在一个 32 卡实例里，`node_rank=0` 收到 HTTP 请求后，大致流程是：

```text
HTTP request
  -> node_rank=0 的 TokenizerManager 做 tokenizer / 请求整理
  -> 发给 scheduler
  -> 各节点 scheduler 驱动本节点 TP ranks 做模型前向
  -> node_rank=0 汇总输出并走 DetokenizerManager
  -> HTTP response
```

每个节点的 scheduler 可以理解成“本节点 GPU worker 的调度器”：

```text
node_rank 0 scheduler:
  管本节点 tp_rank 0..7
  同时连接 tokenizer / detokenizer 这条入口链路

node_rank 1 scheduler:
  管本节点 tp_rank 8..15
  不接 HTTP 请求，只参与 distributed forward

node_rank 2 scheduler:
  管本节点 tp_rank 16..23
  不接 HTTP 请求，只参与 distributed forward

node_rank 3 scheduler:
  管本节点 tp_rank 24..31
  不接 HTTP 请求，只参与 distributed forward
```

这些 scheduler 的核心职责不是“对外服务”，而是：

```text
1. 维护本节点上的请求执行状态和 KV/cache 状态；
2. 把请求组成 batch，决定 prefill/decode 的执行节奏；
3. 调用本节点对应 TP rank 的 model runner 做前向；
4. 通过 TP/NCCL 等通信和其他节点的 scheduler/model runner 协同；
5. 把生成结果沿内部 IPC 链路送回入口侧。
```

所以对外看，tp32 只有一个 HTTP 入口；对内看，32 张卡上的 scheduler/model-runner 都在参与一次请求的前向计算。

## 补充：TP、EP、DP 对外形态的区别

这一块先只记对外服务形态，不深入 MoE forward。

### TP

`tp_size` 描述一个模型实例被多少 rank 共同切分执行。

例如：

```text
tp_size = 32
nnodes = 4
num_gpus_per_node = 8
```

它表示一个 SGLang distributed engine 跨 4 个节点、32 张卡共同完成一次 forward。

对外形态仍然是：

```text
一个 tp32 engine -> 一个 node_rank=0 HTTP URL
```

内部其他 `node_rank` 不接 HTTP，但会启动 scheduler/model-runner 参与 TP 计算。

### EP

`ep_size` 描述 MoE expert 在多少 rank 上切分。

它不是请求入口数量，也不是 request-level worker 数量。`ep_size=32` 不表示可以独立接 32 路请求；它表示 MoE 层里 token/expert 的计算会分布到 32 个 expert-parallel rank 上。

所以：

```text
tp32 + ep32 + dp_size=1
  -> 仍然是一个 HTTP URL
  -> 仍然是一条 engine 内部请求调度链路
  -> EP 只改变 MoE expert 的内部分布方式
```

EP 也不能完全脱离 TP/world ranks 单独存在。原生 SGLang 里只设置 `ep_size=32`，但 `tp_size` 还是默认 1，通常是不成立的；EP ranks 需要从已有的 TP/world rank 空间里切出来。

在 slime 里容易感觉像“只开了 ep32”，是因为 slime 会根据：

```text
rollout_num_gpus_per_engine = 32
```

自动传给 SGLang：

```text
tp_size = 32
```

然后再把配置里的：

```text
sglang_expert_parallel_size = 32
```

传成：

```text
ep_size = 32
```

所以实际是：

```text
tp_size = 32
ep_size = 32
```

不是只有 `ep_size=32`。

另外，某些 SGLang MoE backend，例如 DeepEP，会把 `ep_size` 调整成 `tp_size`。这也是基于已有的 `tp_size` 设置 EP，而不是用 EP 反推 TP。

### DP

`dp_size` 才更接近 request lane 的概念。

当：

```text
dp_size > 1
enable_dp_attention = true
```

SGLang 内部会出现 `DataParallelController`：

```text
HTTP
  -> TokenizerManager
  -> DataParallelController
  -> 某个 dp_rank 的 scheduler group
```

这个 controller 负责在多个 DP lane 之间分发请求。

所以：

```text
tp32:
  没有 DataParallelController，除非 dp_size > 1。

tp32 + ep32 + dp_size=1:
  仍然没有 DataParallelController。

tp32 + ep32 + dp_size=4:
  有 DataParallelController。
```

slime 里还要求：

```python
if args.sglang_dp_size > 1:
    assert args.sglang_enable_dp_attention
```

### 235B 的实际形态

以 235B 脚本为例，核心配置可以理解成：

```text
rollout_num_gpus = 64
rollout_num_gpus_per_engine = 32
sglang_expert_parallel_size = 32
sglang_data_parallel_size = 4
sglang_enable_dp_attention = true
```

slime 传给 SGLang 的单个 engine 大致是：

```text
tp_size = 32
ep_size = 32
dp_size = 4
nnodes = 4
```

对外：

```text
64 张 rollout GPU / 每个 engine 32 张 GPU = 2 个 SGLang engine 副本
router 看到 2 个 node_rank=0 HTTP URL
```

每个 HTTP URL 背后：

```text
node_rank=0:
  HTTP + TokenizerManager + DataParallelController

node_rank=0/1/2/3:
  scheduler/model-runner 共同组成 tp32 / ep32 / dp-attention engine
```

当前阶段先记住这句话：

```text
TP/EP 改的是单个 engine 内部怎么并行；
DP 改的是单个 engine 内部是否有多条 request lane；
HTTP URL 数量仍然主要由 SGLang engine 副本数决定。
```

## 补充：多模型架构

前面主线一直按默认单模型理解：

```text
servers = {
  "default": RolloutServer(...)
}
```

默认情况下，slime 只启动一个模型、一个 router、一个 `regular` server group：

```text
RolloutServer(default)
  -> router(default)
  -> ServerGroup(regular)
    -> SGLangEngine actors
```

多模型不是默认 rollout 主线，而是通过 `--sglang-config` 显式开启。

开启后，`start_rollout_servers()` 返回的 `servers` 可能包含多个 model name：

```text
servers = {
  "actor": RolloutServer(...),
  "ref": RolloutServer(...),
  "reward": RolloutServer(...),
}
```

每个 `RolloutServer` 都有自己的 router：

```text
actor model:
  router(actor)
  ServerGroup(...）

ref model:
  router(ref)
  ServerGroup(...）

reward model:
  router(reward)
  ServerGroup(...）
```

所以多模型架构的层级变成：

```text
SglangConfig
  -> ModelConfig(name="actor")
    -> RolloutServer(actor)
      -> router(actor)
      -> ServerGroup(...)
        -> SGLangEngine actors

  -> ModelConfig(name="ref")
    -> RolloutServer(ref)
      -> router(ref)
      -> ServerGroup(...)
        -> SGLangEngine actors
```

这里的关键变化不是单个 engine 内部怎么并行，而是 **router 隔离到了模型维度**。

### 配置结构

`--sglang-config` 是 YAML，顶层是 `sglang`，下面是模型列表：

```yaml
sglang:
  - name: actor
    model_path: /path/to/actor_model
    update_weights: true
    num_gpus_per_engine: 4
    server_groups:
      - worker_type: regular
        num_gpus: 8
        num_gpus_per_engine: 4

  - name: ref
    model_path: /path/to/ref_model
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: 4
        num_gpus_per_engine: 2
```

模型级字段：

```text
name:
  模型名，也是 args.sglang_model_routers 的 key。

model_path:
  这个模型加载的 HF checkpoint。
  不写时默认使用 args.hf_checkpoint。

update_weights:
  是否接收训练侧权重更新。
  actor 通常是 true。
  ref/reward 这种冻结模型通常是 false。

num_gpus_per_engine:
  该模型内 server group 的默认每 engine GPU 数。
```

server group 字段：

```text
worker_type:
  regular / prefill / decode / placeholder / encoder。
  当前只看 regular 主线。

num_gpus:
  这个 group 总共占多少 GPU。

num_gpus_per_engine:
  每个 SGLang distributed engine 占多少 GPU。

overrides:
  覆盖 SGLang ServerArgs。
```

### GPU 总数校验

使用 `--sglang-config` 时，配置里所有模型、所有 server group 的 `num_gpus` 之和必须等于：

```text
args.rollout_num_gpus
```

代码里会校验：

```python
expected = args.rollout_num_gpus
actual = config.total_num_gpus
assert actual == expected
```

例如：

```yaml
sglang:
  - name: actor
    server_groups:
      - worker_type: regular
        num_gpus: 8
  - name: ref
    server_groups:
      - worker_type: regular
        num_gpus: 4
```

那么命令行必须对应：

```text
--rollout-num-gpus 12
```

### update_weights 的含义

多模型里最重要的字段是 `update_weights`。

```text
update_weights: true
  这个模型会接收训练权重更新。
  通常是 actor / rollout model。

update_weights: false
  这个模型是 frozen serving model。
  通常是 ref / reward / judge / tool-side model。
```

如果不显式写，slime 会根据 `model_path` 推断：

```text
model_path == args.hf_checkpoint:
  update_weights 默认 true

model_path != args.hf_checkpoint:
  update_weights 默认 false
```

这意味着：

```text
actor:
  一般用 args.hf_checkpoint，所以会更新。

ref/reward:
  一般是另一个 checkpoint，所以默认不更新。
```

训练侧权重同步只会选一个 `update_weights=True` 的 server：

```text
RolloutManager._get_updatable_server()
```

所以多模型配置里，不要随便让多个模型都 `update_weights: true`。当前主线可以理解成：只有 actor model 接训练权重。

### 多模型访问方式

`start_rollout_servers()` 最后会写：

```python
args.sglang_model_routers = {
    name: (srv.router_ip, srv.router_port)
    for name, srv in servers.items()
}
```

默认 `generate()` 仍然使用：

```python
url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
```

也就是第一个模型的 router。

如果自定义 rollout 需要访问其他模型，需要显式用：

```python
from slime.rollout.sglang_rollout import get_model_url

actor_url = get_model_url(args, "actor", "/generate")
ref_url = get_model_url(args, "ref", "/generate")
reward_url = get_model_url(args, "reward", "/v1/chat/completions")
```

`get_model_url()` 会从 `args.sglang_model_routers` 里查对应模型的 router。

所以多模型不等于默认 rollout 自动调用多个模型。默认 rollout 还是只请求默认 router；多模型通常需要自定义 rollout 函数主动使用不同 model name。

### 配置案例：actor + ref

这是最容易理解的多模型配置：

```yaml
sglang:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: 8
        num_gpus_per_engine: 4

  - name: ref
    model_path: /path/to/ref_model
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: 4
        num_gpus_per_engine: 2
```

拓扑是：

```text
actor:
  8 GPU total / 4 GPU per engine = 2 actor engine 副本
  1 个 actor router
  接收训练权重更新

ref:
  4 GPU total / 2 GPU per engine = 2 ref engine 副本
  1 个 ref router
  frozen，不接收训练权重更新
```

命令行需要：

```text
--sglang-config sglang_actor_ref.yaml
--rollout-num-gpus 12
--hf-checkpoint /path/to/actor_model
--rollout-num-gpus-per-engine 4
```

注意：`ref` 的 `num_gpus_per_engine=2` 覆盖了命令行的默认值。

### 配置案例：actor + ref + reward

如果 reward 也希望作为 SGLang model 服务，可以写成：

```yaml
sglang:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: 8
        num_gpus_per_engine: 4

  - name: ref
    model_path: /path/to/ref_model
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: 4
        num_gpus_per_engine: 2

  - name: reward
    model_path: /path/to/reward_model
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: 4
        num_gpus_per_engine: 2
```

拓扑是：

```text
actor router:
  服务 rollout generation。

ref router:
  服务 reference logprob / 对比类请求。

reward router:
  服务 reward model 请求。
```

命令行：

```text
--rollout-num-gpus 16
--sglang-config sglang_actor_ref_reward.yaml
```

这种配置通常必须配合自定义 rollout：

```text
--rollout-function-path my_pkg.rollout.generate_rollout
```

或者至少自定义单样本生成逻辑：

```text
--custom-generate-function-path my_pkg.rollout.custom_generate
```

否则默认 `generate()` 不会自动调用 `ref` 和 `reward`。

### 和 ServerGroup 的关系

多模型只是多了一层 `ModelConfig` / `RolloutServer`。

每个模型内部仍然沿用前面讲过的规则：

```text
num_gpus_per_engine <= num_gpus_per_node:
  一个 engine 通常对应一个 Ray actor。

num_gpus_per_engine > num_gpus_per_node:
  一个 engine 跨多个节点，由多个 Ray actor 组成。

engines:
  每个 distributed engine 的 node_rank=0 actor。

all_engines:
  该模型该 group 的所有 Ray actors。
```

所以多模型的心智模型是：

```text
先按 model 拆 router；
再在每个 model 内按 server group 拆 engine；
每个 engine 内部再按 node_rank 拆 Ray actor。
```

### 当前阶段的结论

多模型架构先记住三点：

```text
1. 一个 model 一个 RolloutServer，一个 router。

2. update_weights 决定哪个 model 接训练权重。
   actor 通常 true，ref/reward 通常 false。

3. 默认 rollout 不会自动使用多个模型。
   真正调用 ref/reward，需要自定义 rollout 或 custom generate。
```

## 补充：External Rollout Engines

`start_rollout_servers()` 最前面有一个分支：

```python
if args.rollout_external:
    return start_external_rollout_servers(args, start_router=_start_router)
```

这个分支对应的是：

```text
SGLang engine 不由 slime 启动；
SGLang engine 已经由外部系统预启动和管理；
slime 只负责连接这些 engine、注册 router、发 rollout 请求、同步权重。
```

触发参数是：

```text
--rollout-external-engine-addrs host1:port host2:port ...
```

这条路线和前面默认的 PG/Ray actor 启动方式完全不同。

默认模式是：

```text
slime
  -> 创建 PG
  -> 创建 Ray SGLangEngine actor
  -> actor 内 launch_server_process()
  -> 启动 SGLang server
  -> 注册 router
```

external 模式是：

```text
外部系统
  -> 已经启动 SGLang server

slime
  -> 请求 external server 的 /server_info
  -> 推断 num_gpus / tp_size / pp_size / worker_type
  -> 启动或复用 router
  -> 创建轻量 SGLangEngine wrapper actor
  -> wrapper 不启动 SGLang，只校验并注册 external worker
```

所以 external 模式下，slime 里的 `SGLangEngine` actor 只是 control handle：

```text
它不拥有本地 SGLang process；
它不负责 kill/restart external engine；
它主要负责把 external engine 接入 slime 的控制面。
```

### 使用场景

这个功能主要面向训推解耦，或者更进一步的全服务化 rollout serving。

典型形态：

```text
训练侧:
  slime + Megatron + Ray

推理侧:
  外部 SGLang serving 集群
  独立环境 / 独立容器 / 独立 GPU 池 / 独立编排系统
```

适合的场景：

```text
1. rollout serving 不想由 slime 训练任务启动；
2. SGLang 环境和训练环境不同；
3. 训练和推理在不同 Ray 集群；
4. 训练 GPU 和 rollout GPU 不是同型号，甚至不是同厂家；
5. 训练侧和推理侧不能方便地建立 NCCL group；
6. 权重同步需要走共享文件系统上的 full checkpoint 或 delta。
```

这套基建的方向很接近 Cursor 公开技术路线里描述的高度服务化 RL inference：训练循环和 rollout generation 解耦，训练侧产出权重更新，推理侧作为独立服务消费新权重并持续提供 generation。

slime 官方文档里也明确把 external engine、update from disk、delta disk transport 放在同一类基础设施问题下讨论：

[External Rollout Engines 配置路线图](/mnt/shared-storage-user/huanghaian/code/slime_package/slime/docs/zh/advanced/external-rollout-engines.md)

### 权重同步

external engine 仍然可以接收训练后的 actor 权重，但同步方式要看训练侧和推理侧的部署关系。

如果训练器和 external engine 可以建立 NCCL group，可以继续使用 NCCL transport。

如果不能建立 NCCL group，但能看到同一个共享文件系统路径，可以用：

```text
--update-weight-mode full
--update-weight-transport disk
--update-weight-disk-dir /shared/fs/full-updates
```

训练侧会写完整 HF checkpoint，然后调用 external SGLang worker 的：

```text
update_weights_from_disk
```

大模型场景下，完整 checkpoint 太重，可以走 delta：

```text
--update-weight-mode delta
--update-weight-transport disk
--update-weight-encoding deltas_zstd
--update-weight-disk-dir /shared/fs/delta-updates
```

这个模式下，训练侧只写变化部分，SGLang 侧通过 `update_weights_from_disk(load_format="delta")` 应用 delta。

### 和共卡 colocate 的关系

external 模式基本不是给共卡场景用的。

因为 external 模式下：

```text
slime 不为 rollout engine 分配 PG GPU；
slime 不启动 SGLang process；
slime 不负责 rollout offload/onload；
slime 不支持 external engine 的 fault tolerance recover；
external engine 生命周期由外部系统负责。
```

真正共卡时，Megatron 和 SGLang 要共享同一批 GPU，必须有人协调：

```text
什么时候 offload train；
什么时候 offload rollout；
什么时候 onload；
显存如何错峰；
PG 如何布局。
```

这应该走 slime 自己管理的 colocate 模式，而不是 external engine：

```text
--colocate
```

external 模式可以在同一批机器上使用，但前提通常是：

```text
训练 GPU 和 rollout GPU 是不同 GPU 池；
不是同一张卡的显存复用。
```

### 和 --sglang-config 的关系

`--rollout-external-engine-addrs` 和 `--sglang-config` 互斥。

原因是两者的边界不同：

```text
--sglang-config:
  slime 负责 engine 生命周期。
  你用 YAML 描述 topology，slime 启动 server group、router、多模型等。

--rollout-external-engine-addrs:
  外部系统负责 engine 生命周期。
  slime 只发现已启动 engine，接入 router，并把它们当作默认 rollout model。
```

所以如果需求是：

```text
多模型 serving
reference/reward 冻结模型
PD 分离
异构 server group
per-group overrides
```

优先看 `--sglang-config`。

如果需求是：

```text
SGLang engine 已经在训练任务外部部署好；
希望 slime 只接入这些服务；
训练和推理基础设施解耦；
```

再使用 external engine。

### 配套例子

仓库里有正式文档：

[External Rollout Engines 配置路线图](/mnt/shared-storage-user/huanghaian/code/slime_package/slime/docs/zh/advanced/external-rollout-engines.md)

也有 E2E 测试覆盖 external PD fleet：

```text
tests/test_qwen3_4B_external_pd.py
```

这个测试会预启动外部 SGLang PD engine，然后训练侧通过：

```text
--rollout-external-engine-addrs ...
--update-weight-mode delta
--update-weight-transport disk
```

接入这些 external engine。

## 补充：placeholder group

官方文档：

- [SGLang Config：worker_type 字段](/mnt/shared-storage-user/huanghaian/code/slime_package/slime/docs/zh/advanced/sglang-config.md:58)
- [SGLang Config：占位组用于 GPU 预留](/mnt/shared-storage-user/huanghaian/code/slime_package/slime/docs/zh/advanced/sglang-config.md:220)

`placeholder` 的含义是：

```text
占用 rollout GPU slot，但不启动 SGLang engine。
```

官方例子：

```yaml
sglang:
  - name: actor
    server_groups:
      - worker_type: regular
        num_gpus: 6
        num_gpus_per_engine: 2
      - worker_type: placeholder
        num_gpus: 2
```

这里的意思是：

```text
regular group:
  使用 6 张 GPU，启动 SGLang engine。

placeholder group:
  预留 2 张 GPU slot，不创建 engine。
```

代码里 `_make_group()` 仍然会给 placeholder 累加 `gpu_offset`：

```python
gpu_offset += group_cfg.num_gpus
```

但创建 `ServerGroup` 时：

```python
all_engines=[None] * num_engines if group_cfg.worker_type != "placeholder" else []
```

`start_engines()` 里也会直接返回：

```python
if self.args.debug_train_only or self.worker_type == "placeholder":
    self.num_new_engines = 0
    return [], port_cursors
```

所以 placeholder 的本质是：

```text
参与 GPU offset / group 顺序计算；
不创建 Ray actor；
不注册 router；
不参与 health；
不参与 offload/onload；
不参与 weight update。
```

它解决的问题不是“少启动几个 engine”。如果只是想少用 GPU，可以直接调小 `rollout_num_gpus`。

placeholder 真正解决的是：

```text
保留 rollout 总 slot 布局；
某一段 slot 不启动 SGLang；
后面的 group 仍然从指定 offset 往后排。
```

### 一个 colocate 例子

假设一台 8 卡机器：

```text
GPU 0-3: actor 训练用
GPU 4-7: 想给 rollout-only SGLang 用
```

如果希望 SGLang regular group 不落到前 4 张训练卡上，可以这样配：

```yaml
sglang:
  - name: default
    server_groups:
      - worker_type: placeholder
        num_gpus: 4
      - worker_type: regular
        num_gpus: 4
        num_gpus_per_engine: 4
```

布局变成：

```text
rollout slot 0-3:
  placeholder，不启动 SGLang。

rollout slot 4-7:
  regular group，启动 SGLang。
```

这样 regular group 的 `gpu_offset` 从 4 开始。

如果训练侧 `megatron_num_gpus=4`，那么：

```python
needs_offload = group_abs_start < megatron_num_gpus
              = 4 < 4
              = False
```

也就是说这个 regular group 会被视为 rollout-only，不需要 offload/onload。

如果不用 placeholder，直接写：

```yaml
server_groups:
  - worker_type: regular
    num_gpus: 4
```

这个 regular group 会从 `gpu_offset=0` 开始，更容易落在训练共卡区间里。

### 中间留洞例子

也可以在中间留一段 slot：

```yaml
server_groups:
  - worker_type: prefill
    num_gpus: 2
    num_gpus_per_engine: 2
  - worker_type: placeholder
    num_gpus: 2
  - worker_type: decode
    num_gpus: 4
    num_gpus_per_engine: 4
```

布局是：

```text
slot 0-1: prefill
slot 2-3: placeholder，留空 / 给训练 / 对齐拓扑
slot 4-7: decode
```

没有 placeholder 的话，decode 会从 slot 2 开始，而不是 slot 4。

## 补充：port_cursors

`port_cursors` 用来避免同一个 `start_rollout_servers` 流程里多个 group 之间端口冲突。

一个 SGLang engine 不只需要一个端口，至少有：

```text
HTTP server port
NCCL port
dist_init_addr port
```

如果是 prefill worker，还会多一个：

```text
disaggregation_bootstrap_port
```

因此 PD/EPD/多 group 场景下，同一批节点上可能连续启动：

```text
encoder group
prefill group
decode group
```

如果每个 group 都从固定端口开始找，很容易抢到同一批“刚才看起来可用、但马上会被另一个 SGLang 子进程占用”的端口。

`start_rollout_servers` 里每个 model 有一个：

```python
port_cursors: dict[int, int] = {}
```

然后 group 之间传递：

```python
handles, port_cursors = group.start_engines(port_cursors)
```

`port_cursors` 的含义是：

```text
node_index -> 这个节点下一次应该从哪个端口继续找
```

例如第一个 group 在 node 0 上用了：

```text
15000
15001
15002
15033
```

那么 `port_cursors[0]` 会推进到后面的某个值。第二个 group 再在 node 0 上启动时，就不会从 15000 开始，而是从 cursor 之后继续找。

代码里还有一个保守处理：

```python
base_port = max(port_cursors.values()) if port_cursors else 15000
```

也就是说，后续 group 会整体从更靠后的端口段开始找。它不是每个节点最紧凑地复用端口段，而是更偏向减少 group 间端口冲突概率。

所以：

```text
placeholder:
  控制 GPU slot 布局，不启动 engine。

port_cursors:
  控制同一轮多 group 启动时的端口布局，避免端口冲突。
```
