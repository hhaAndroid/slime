# Rollout Control Plane：engines、router、health 与 recover

这篇只记录 rollout 控制面，不继续展开 SGLang 内部 TP/EP/DP forward。

## all_engines 和 engines

在 slime 里有两个很容易混淆的列表：

```python
ServerGroup.all_engines
ServerGroup.engines
```

`all_engines` 是 Ray 层真实创建出来的所有 `SGLangEngine` actor。

跨节点 32 卡实例时：

```text
rollout_num_gpus_per_engine = 32
num_gpus_per_node = 8
nodes_per_engine = 4
```

一个 SGLang distributed engine 会对应 4 个 Ray actor：

```text
all_engines[0] = engine 0, node_rank 0
all_engines[1] = engine 0, node_rank 1
all_engines[2] = engine 0, node_rank 2
all_engines[3] = engine 0, node_rank 3
```

`engines` 是 `all_engines` 的切片：

```python
return self.all_engines[:: self.nodes_per_engine]
```

也就是只取每个 distributed engine 的 `node_rank=0` actor。

所以：

```text
all_engines:
  表示 Ray/SGLang 实际进程拓扑，包含 node_rank=0/1/2/3。

engines:
  表示对外可控入口，只有 node_rank=0。
```

## 哪些操作走 engines

大部分 HTTP 控制接口只需要打到 `node_rank=0`。

因此这些操作走 `engines (`self._server_group.engines，和 router 没有关心`)`：

```text
health_generate
flush_cache
release_memory_occupation
resume_memory_occupation
update_weights_from_tensor
update_weights_from_disk
init_weights_update_group
destroy_weights_update_group
weights_checker
get_weight_version
update_weight_version
get_url
router register/remove
```

原因是 `node_rank=0` 是 SGLang distributed engine 的 HTTP/control 入口。控制请求进入 `node_rank=0` 后，SGLang 内部再协调其他 rank。

所以 slime 不会对 `node_rank=1/2/3` 分别调用这些 HTTP endpoint。

这和 lmdeploy 有很大区别，在 ep8 情况下，sglang 只有 1 个 ray actor 也只有 1 个 http url，健康检测自然也就只要管这个就行，但是在 lmdeploy 中，会存在 8 个 url，因此健康检测就要发给 8 个，一旦某个挂了需要把其余 7 个都标记为失败，会比较复杂，sglang  引擎则不需要考虑这个问题。这不只是是否引入 router 的问题了，除非 router 自己做了这个健康检测。

## 哪些操作必须看 all_engines

进程生命周期必须看 `all_engines`。

典型场景是失败恢复：

```text
一个 32 卡实例坏了
  -> 不能只杀 node_rank=0
  -> 必须杀 node_rank=0/1/2/3
  -> 然后整组重建
```

因此 health monitor 在 kill 时会根据 `nodes_per_engine` 找到整个实例对应的 actor 范围：

```text
rollout_engine_id = 0
nodes_per_engine = 4

kill all_engines[0]
kill all_engines[1]
kill all_engines[2]
kill all_engines[3]
set all_engines[0..3] = None
```

这也是为什么 `all_engines` 不能被 `engines` 替代。

## router 负责什么

router 只知道 worker URL。

对一个跨节点 tp32 实例来说，router 只看到：

```text
http://node_rank_0_host:port
```

它不知道：

```text
node_rank=1
node_rank=2
node_rank=3
Ray actor 列表
PG bundle
dist_init_addr
```

router 主要负责：

```text
1. 保存 worker URL 列表；
2. 接收正常 rollout 生成请求；
3. 在多个 node_rank=0 worker URL 之间路由；
4. 提供 worker 注册、删除、查询接口。
```

正常生成请求经过 router：

```text
slime rollout
  -> http://router_ip:router_port/generate
  -> router 选择某个 node_rank=0 worker URL
  -> SGLang engine
```

但生命周期管理接口基本不经过 router。

## 哪些请求经过 router

默认 rollout 生成经过 router：

```python
url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
```

也就是：

```text
/generate:
  经过 router
```

abort 时会先向 router 查询 worker：

```text
GET router /workers
```

这里 router 只是 worker discovery，不负责转发 abort。真正的 abort 请求会直接发给每个 worker。

## 哪些请求绕过 router

`SGLangEngine._make_request()` 直接打 worker URL：

```python
url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
```

这些请求不经过 router：

```text
release_memory_occupation
resume_memory_occupation
update_weights_from_tensor
update_weights_from_disk
init_weights_update_group
destroy_weights_update_group
weights_checker
update_weight_version
```

这些也直接打 worker：

```text
health_generate
flush_cache
get_weight_version
abort_request
v1/loads
```

所以心智模型是：

```text
正常推理流量:
  经过 router。

控制面请求:
  基本直接打 node_rank=0 worker。

router 自身:
  只维护 worker 列表和转发生成请求。
```

## 从最外层 generate 到 SGLang 请求

理解 `abort()` 前，先把默认 rollout 生成链路串起来。

最外层入口在 `RolloutManager.generate()`：

```text
RolloutManager.generate(rollout_id)
  -> health_monitoring_resume()
  -> _get_rollout_data(rollout_id)
  -> _convert_samples_to_train_data(data)
  -> _split_train_data_by_dp(data)
```

其中真正取 rollout 数据的是 `_get_rollout_data()`：

```text
RolloutManager._get_rollout_data()
  -> call_rollout_fn(self.generate_rollout, args, rollout_id, data_source, evaluation=False)
```

`self.generate_rollout` 来自：

```python
self.generate_rollout = load_function(self.args.rollout_function_path)
```

默认配置下：

```text
--rollout-function-path = slime.rollout.sglang_rollout.generate_rollout
```

所以默认主链路是：

```text
RolloutManager.generate()
  -> RolloutManager._get_rollout_data()
  -> slime.rollout.sglang_rollout.generate_rollout()
  -> generate_rollout_async()
  -> GenerateState(args)
  -> submit_generate_tasks()
  -> generate_and_rm_group()
  -> generate_and_rm()
  -> generate()
  -> POST router /generate
```

几个关键函数的职责：

```text
RolloutManager.generate:
  Ray actor 方法，训练侧调用它拿一轮 rollout 数据。

generate_rollout:
  默认 rollout function，是 --rollout-function-path 的默认值。
  它负责启动 async rollout，并在最后把 partial aborted samples 放回 data_source。

generate_rollout_async:
  默认 rollout 调度核心。
  它循环提交 group、等待完成、动态过滤、凑够 rollout_batch_size。
  凑够后会调用 abort() 清理多余 pending 请求。

GenerateState:
  当前进程内的 rollout 共享状态。
  管 semaphore、pendings、aborted、tokenizer、processor、dp_counts。

submit_generate_tasks:
  把 data_source 返回的每个 group 包成 asyncio task，放进 state.pendings。

generate_and_rm_group:
  处理一个 prompt group，通常包含 n_samples_per_prompt 个 sample。

generate_and_rm:
  处理单个 sample 的生成和 reward。
  如果设置了 --custom-generate-function-path，会在这里替换默认 generate()。

generate:
  默认单 sample SGLang 调用。
  负责组装 input_ids / sampling_params，然后 POST router /generate。
```

默认情况下，生成请求经过 router：

```text
generate()
  -> http://{args.sglang_router_ip}:{args.sglang_router_port}/generate
  -> router 选择 worker
  -> worker node_rank=0
  -> SGLang distributed engine
```

而每条样本自己的 prompt、response、reward 都保存在 `Sample` 里：

```text
GenerateState:
  共享控制状态。

Sample:
  单条请求的数据状态。
```

这个结构解释了为什么后面 `abort()` 能工作：

```text
state.pendings:
  知道当前 rollout 还有哪些 asyncio task 没收尾。

state.aborted:
  能阻止后续 task 继续发新的 generate 请求。

router /workers:
  能发现已经发出去的请求可能落在哪些 worker 上。
```

## abort 的完整逻辑

slime 里的 `abort()` 不是健康恢复，也不是重启 engine。它主要用于 rollout 已经凑够训练 batch 后，把多余还在跑的 SGLang 请求停掉。

主流程在 `generate_rollout_async()`：

```text
1. 持续提交 generate tasks；
2. 谁先完成就先拿回来；
3. 动态过滤后，凑够 rollout_batch_size；
4. 此时可能还有一些 pending requests 已经发给 SGLang；
5. 调 abort(args, rollout_id) 停掉这些多余请求；
6. partial rollout 场景下，把已经生成了一部分的样本放回 data_source。
```

为什么会有 pending？

因为 rollout 为了保持吞吐，会提前提交任务：

```python
while state.remaining_batch_size < target_data_size:
    samples = data_source(args.over_sampling_batch_size)
    state.submit_generate_tasks(samples)
```

`submit_generate_tasks()` 会把每个 group 包成一个 asyncio task：

```python
state.pendings.add(asyncio.create_task(generate_and_rm_group(...)))
state.remaining_batch_size += len(samples)
```

当 `len(data) == rollout_batch_size` 后，训练需要的数据已经够了，但剩下的 pending task 可能处于几种状态：

```text
1. 还没真正发出 /generate；
2. 已经发到 router /generate；
3. 正在某个 SGLang worker 上生成；
4. 已经生成了一部分 token。
```

`abort()` 第一件事是停止本地继续发新请求：

```python
state.aborted = True
```

后续还没进入 generate 的 task 会看到：

```python
if state.aborted:
    sample.status = Sample.Status.ABORTED
    return sample
```

第二件事是停止 SGLang 里已经在跑的请求。

它会先通过 router 拿当前 worker URL 列表：

```text
GET router /workers
```

为啥不通过 engine 来 abort 而要多此一举？原因是这一步只是 discovery，因为 rollout 函数层只有 `args.sglang_router_ip` / `args.sglang_router_port`，没有 `RolloutServer`、`ServerGroup` 或 Ray actor handle。

拿到 worker URL 后，真正 abort 是直接打 worker：

```text
POST worker /abort_request {"abort_all": true}
```

然后检查这个 worker 是否已经 idle：

```text
GET worker /v1/loads?include=core
```

如果还有 running/waiting requests，就隔 3 秒继续 abort：

```text
POST worker /abort_request
GET worker /v1/loads?include=core
...
```

所以 abort 的 HTTP 路径是：

```text
router:
  GET /workers
  只用于发现 worker URL。

worker:
  POST /abort_request
  GET /v1/loads?include=core
  真正执行 abort 和 idle 检查。
```

为什么不直接从 `engines` 拿 URL？

因为 `abort()` 在 `slime/rollout/sglang_rollout.py` 的 rollout 函数层，它只通过 HTTP 和 SGLang 交互。`engines` 存在于 `RolloutManager -> servers -> server_groups` 这层控制面对象里，没有传进 rollout function。

因此当前实现选择：

```text
rollout function
  -> 问 router 当前有哪些 worker
  -> 逐个直接 abort worker
```

这样还能兼容 external rollout engine 和动态变化后的 worker 列表。

第三件事是等待本地 pending tasks 全部收尾：

```python
while state.pendings:
    done, state.pendings = await asyncio.wait(
        state.pendings,
        return_when=asyncio.FIRST_COMPLETED,
    )
```

如果不是 partial rollout，这些 pending task 结束后就丢掉。

如果是 partial rollout，会把已经生成了部分 response 的样本收集起来：

```python
if sample.response:
    sample.metadata["start_rollout_id"] = rollout_id
aborted_samples.append(group)
```

最后在 `generate_rollout()` 里放回 data source：

```python
if aborted_samples:
    data_source.add_samples(aborted_samples)
```

所以 partial rollout 下，这些半成品样本不会浪费，后面可以继续生成。

总结：

```text
abort =
  停止本地继续发新请求
  + 查询 router 获取 worker URL
  + 直接通知所有 worker 清空正在跑的请求
  + 等 pending asyncio task 收尾
  + partial rollout 时回收半成品样本
```

### abort 相关源码摘录

先看 `generate_rollout_async()` 里为什么最后一定会调用 `abort()`。

```python
async def generate_rollout_async(args, rollout_id, data_source):
    state = GenerateState(args)
    target_data_size = args.rollout_batch_size

    data = []
    all_data = []

    while len(data) < target_data_size:
        # 为了保持 rollout 吞吐，这里会提前提交一批 group。
        #
        # 注意：remaining_batch_size 统计的是“已提交但还没被最终处理掉”的 group 数。
        # 它不是已经成功进入训练 batch 的数量。
        while state.remaining_batch_size < target_data_size:
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # 每次只等至少一个 group 完成。
        # 这意味着当 data 凑够 rollout_batch_size 时，state.pendings 里通常还会有
        # 之前为了吞吐提前发出去、但现在已经不再需要的请求。
        done, state.pendings = await asyncio.wait(
            state.pendings,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in done:
            group = task.result()
            all_data.append(group)

            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                # 被过滤掉的 group 不能进入训练 batch。
                # 所以 remaining_batch_size 要减掉，让外层继续补新请求。
                state.remaining_batch_size -= 1
                continue

            if len(data) < target_data_size:
                data.append(group)

    # 走到这里时，训练需要的 batch 已经够了。
    # 但是 state.pendings 里可能还有多余请求：
    #   - 有些还在本地 asyncio 队列里；
    #   - 有些已经发到 router；
    #   - 有些已经在 worker 上生成了一部分 token。
    #
    # 所以这里需要 abort，把多余请求停掉并收尾。
    aborted_samples = await abort(args, rollout_id)

    # reset 很关键：GenerateState 是 Singleton，
    # 不 reset 会影响下一轮 rollout/eval。
    state.reset()
    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples
```

`submit_generate_tasks()` 本身只负责把 group 包成 task 并放进 `state.pendings`：

```python
def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
    for group in samples:
        self.pendings.add(
            asyncio.create_task(
                generate_and_rm_group(
                    self.args,
                    group,
                    sampling_params=self.sampling_params.copy(),
                    evaluation=False,
                )
            )
        )

    # 这里加的是 group 数，不是 sample 数。
    self.remaining_batch_size += len(samples)
```

再看 `abort()` 本体：

```python
async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)

    # 一个 rollout 周期内只允许 abort 一次。
    assert not state.aborted

    # 本地开关：阻止还没真正发出 generate 的 task 继续请求 SGLang。
    state.aborted = True

    # rollout 函数层没有 RolloutServer / engines / Ray actor handle。
    # 它只有 router 地址，所以先通过 router 查询当前 worker URL。
    if parse(sglang_router.__version__) <= parse("0.2.1"):
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    # 注意：这里不是把 abort 请求发给 router。
    # router 只提供 worker discovery。
    # 真正的 abort 会在 abort_servers_until_idle(urls) 里逐个直接打 worker。
    await abort_servers_until_idle(urls)

    # SGLang 侧 abort 完后，本地 asyncio task 也要收尾。
    # 有些 task 会因为 SGLang abort 返回，有些 task 可能在 state.aborted 后直接返回 ABORTED。
    while state.pendings:
        done, state.pendings = await asyncio.wait(
            state.pendings,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # 非 partial rollout 下，多余请求直接丢掉。
        if not args.partial_rollout:
            continue

        # partial rollout 下，已经生成了部分 response 的样本要回收。
        # 后面会通过 data_source.add_samples(aborted_samples) 放回数据源。
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)

    return aborted_samples
```

`abort_servers_until_idle()` 做的是逐 worker abort，并等待 worker 进入 idle：

```python
async def _abort_server_once(url: str) -> None:
    # url 是 worker URL，不是 router URL。
    # 请求的是 SGLang worker 的 /abort_request。
    await post(f"{url}/abort_request", {"abort_all": True})


async def _get_server_num_requests(url: str) -> int:
    # 直接查 worker 当前 load。
    # include=core 会返回 scheduler/core 侧的 running/waiting request 信息。
    return num_requests_from_load(await get(f"{url}/v1/loads?include=core"))


async def abort_server_until_idle(url: str, retry_interval: int = 3) -> None:
    attempt = 1
    while True:
        # 反复 abort 是为了处理第一次 abort 后仍有请求残留的情况。
        await _abort_server_once(url)

        try:
            num_requests = await _get_server_num_requests(url)
        except Exception:
            # 如果 load 查询失败，当前实现选择返回。
            # 这里不是健康恢复逻辑，不负责 kill/restart worker。
            return

        if num_requests <= 0:
            return

        await asyncio.sleep(retry_interval)
        attempt += 1


async def abort_servers_until_idle(urls: list[str]) -> None:
    # 对所有 worker 并发执行 abort。
    # 这很重要：如果有多个 engine 副本，不能只 abort router 选中的一个 worker。
    await asyncio.gather(*(abort_server_until_idle(url) for url in urls))
```

因此源码层面可以总结成：

```text
generate_rollout_async:
  决定什么时候训练 batch 已经够了，并触发 abort。

abort:
  负责本地状态切换、通过 router 发现 worker、等待 pending task 收尾。

abort_servers_until_idle:
  负责逐个 worker 发送 /abort_request，并用 /v1/loads 判断是否清空。
```

## health monitor 怎么判断失败

slime 里启动 router 时明确关掉了 router 自带 health check：

```python
router_args.disable_health_check = True
```

因此健康检查主要由 `RolloutHealthMonitor` 做。

health monitor 只检查 `group.engines`，也就是每个 distributed engine 的 `node_rank=0`：

```text
for engine in server_group.engines:
  engine.health_generate()
```

`health_generate()` 会请求：

```text
http://node_rank_0_host:port/health_generate
```

这个不是普通 ping，而是 SGLang 的 generate health check。对于跨节点 tp32，如果某个非 0 rank 的 scheduler/model-runner 失效，`node_rank=0` 的 health generate 理论上也会失败或 timeout。

## 失败后怎么处理

当 health check 失败：

```text
RolloutHealthMonitor._check_engine_health()
  -> _kill_engine(rollout_engine_id)
```

`_kill_engine()` 会杀掉整个 distributed engine：

```text
1. 找到这个 engine 对应的 all_engines 范围；
2. 对每个 actor 调 shutdown；
3. ray.kill(actor)；
4. all_engines[i] = None。
```

对于 `node_rank=0`，`shutdown()` 会先尝试从 router 删除 worker URL：

```text
DELETE router /workers/{id}
```

对于 `node_rank=1/2/3`，`shutdown()` 不碰 router，只杀 SGLang process。

所以失败处理的主线是：

```text
health_generate 失败
  -> node_rank=0 shutdown: 从 router 删除 worker URL
  -> kill node_rank=0/1/2/3
  -> all_engines[对应范围] = None
```

## recover 怎么重建

recover 不是 health monitor 线程一检测到失败就立刻执行。

当前逻辑是两段式：

```text
health monitor:
  发现失败、kill 整个 distributed engine、把 all_engines 标记成 None。

recover:
  由训练侧在 update_weights 前统一触发。
```

也就是说，health monitor 的职责到这里为止：

```text
health_generate 失败
  -> _kill_engine()
  -> shutdown + ray.kill
  -> all_engines[i] = None
```

它不会自己调用 `start_engines()`。

真正的 recover 调用点在 Megatron actor 的 `update_weights()` 里：

```python
if self.args.use_fault_tolerance:
    if dist.get_rank() == 0:
        ray.get(self.rollout_manager.recover_updatable_engines.remote())
    dist.barrier(group=get_gloo_group())
```

所以恢复时机是：

```text
generate/eval 期间:
  health monitor 发现坏 engine，先摘掉并标记 None。

下一次 update_weights 前:
  rank 0 统一调用 recover_updatable_engines()。

recover 后:
  新 engine 会进入本轮 update_weights 流程，拿到最新权重。
```

这个时机很重要。新 engine 即使能启动，也需要重新接入权重同步；因此把 recover 放在 `update_weights()` 前面，比在 health monitor 线程里立即重启更自然。

流程可以理解成：

```text
先重启 dead engine
  -> 再 connect_rollout_engines
  -> 再 update_weights
  -> 最后恢复成可服务状态
```

注意：`recover_updatable_engines()` 只恢复 `update_weights=True` 的 server，也就是训练权重会同步过去的 rollout model。它不是对所有 server 做无差别即时恢复。

`RolloutServer.recover()` 会先记录哪些 `all_engines` 是 `None`：

```text
dead_per_group = indices where g.all_engines[i] is None
```

然后重新调用：

```text
ServerGroup.start_engines()
```

因为 `start_engines()` 只创建 `all_engines[i] is None` 的 actor，所以它会只重建死掉的那些节点分片。

对于一个被 health monitor 整组 kill 的 tp32 实例，4 个 actor 都是 `None`，因此会整组重建：

```text
new node_rank=0
new node_rank=1
new node_rank=2
new node_rank=3
same distributed engine structure
new dist_init_addr / ports # 可以和之前不一样
```

初始化完成后，新的 `node_rank=0` 会重新注册到 router。

## offload/onload 和 health 的关系

offload/onload 不通过 health monitor 通知。

它们的协同方式是 pause/resume：

```text
RolloutManager.offload()
  -> health_monitoring_pause()
  -> srv.offload()
  -> release_memory_occupation

RolloutManager.generate()/eval()
  -> health_monitoring_resume()
  -> 开始请求 rollout
```

这样做是因为 offload 后 engine 可能不能正常响应 generate health check。如果不暂停，health monitor 可能把 offloaded engine 误判成失败。

## 是否可能出现不一致

有可能存在短窗口。

正常路径下：

```text
health check 失败
  -> node_rank=0 shutdown 删除 router worker
  -> kill 整组 actor
  -> all_engines = None
  -> recover 重建
  -> 新 node_rank=0 注册 router
```

这个流程最终会收敛。

但如果 actor 或 SGLang process 已经硬死，可能发生：

```text
shutdown.remote() 失败
router worker 没删掉
all_engines 已经被置 None
```

这时 router 里可能短时间残留旧 worker URL。

因为 router 自身 health check 被关掉了，它不会主动判断这个 stale worker。slime 主要依赖 `shutdown()` 成功删除 router worker，以及后续 recover 注册新 worker。

所以准确说：

```text
正常可控失败:
  一致性能收敛。

硬失败:
  可能存在 router worker 列表和 all_engines 状态短暂不一致。
```

这个风险的根源是：

```text
router 只知道 node_rank=0 URL；
slime 才知道完整的 distributed engine actor 组。
```

## 当前心智模型

总结成几句话：

```text
engines:
  控制入口列表，只有 node_rank=0。

all_engines:
  真实进程拓扑列表，包含所有 node_rank。

router:
  负责正常生成请求路由，不负责 slime 的完整生命周期管理。

health monitor:
  检查 node_rank=0 的 /health_generate。
  一旦失败，按 nodes_per_engine 杀整个 distributed engine。

recover:
  根据 all_engines 里的 None 重建 actor。
  新 node_rank=0 初始化后重新注册 router。
```
