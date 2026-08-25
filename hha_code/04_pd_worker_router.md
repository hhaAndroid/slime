# PD Worker 与 Router 协同

这篇记录 PD 分离场景下 prefill worker、decode worker、router 以及 slime rollout 侧的协同关系。重点回答几个问题：

```text
PD worker 的协同是不是在 router 中做的？
prefill/decode 都返回时，上游到底拿谁的 response？
logprob、R3 routed experts、abort 这些边界在 PD 下怎么处理？
如果不用 router，RL 系统自己要补哪些逻辑？
```

先给当前结论：

```text
1. router 负责请求级配对和请求改写。
   它选择 prefill/decode pair，生成 bootstrap_room，注入 bootstrap 信息，并把请求并发发给两侧。

2. SGLang worker 内部负责真正的 prefill/decode 执行、KV 传输和等待同步。
   router 不是 KV 传输执行者。

3. 上游主 response 通常来自 decode worker。
   prefill response 主要给 router 使用，典型用途是提供 input_token_logprobs。

4. logprob 是 router 明确支持的 PD 特判字段。
   input_token_logprobs 来自 prefill，output_token_logprobs 来自 decode，router 负责合并。

5. R3 routed experts 在 slime Docker pin 的 patched SGLang/router 组合下是目标支持路径。
   这个支持依赖 slime 版 sglang_router、SGLang patch，以及 slime 对 routed_experts_start_len 的解析。
   换成未打 patch 的 SGLang 或非 slime 版 router 时，不能默认可用。

6. abort 在 slime 里是粗粒度 cleanup 语义。
   当前是对 router 注册的所有 prefill/decode worker 发 abort_all 并等 idle，不是 per-request pair 精确 abort。

7. 不用 router 不会减少 PD 协同复杂度。
   只是把 pair 选择、bootstrap 注入、并发请求、logprob/R3 拼接、abort 和健康管理搬到 RL client 里。
```

也就是说，router 是 PD 的入口协调器；worker 才是真正执行 prefill/decode 和 KV transfer 的地方。slime 当前默认把这层协调交给 SGLang router，而不是在 rollout client 里手写一套。

这篇里的“支持 R3”特指 slime 当前 Docker/runtime 口径：`slime` pin 的 patched SGLang image、patched `sglang_router` wheel，以及当前 slime 代码三者一起工作。直接看一个未打 patch 的 SGLang checkout，结论会不完整。

## 相关代码

slime 侧：

- `slime/ray/rollout.py`
- `slime/backends/sglang_utils/sglang_engine.py`
- `slime/rollout/sglang_rollout.py`
- `slime/rollout/sglang_streaming_rollout.py`
- `slime/utils/types.py`
- `slime/utils/health_monitor.py`
- `slime/backends/sglang_utils/server_control.py`
- `docker/Dockerfile`
- `docker/patch/latest/sglang.patch`

SGLang 侧：

- `../sglang/sgl-model-gateway/bindings/python/src/sglang_router/mini_lb.py`
- `../sglang/experimental/sgl-router/src/discovery/types.rs`
- `../sglang/python/sglang/srt/managers/tokenizer_manager.py`
- `../sglang/python/sglang/srt/managers/data_parallel_controller.py`
- `../sglang/python/sglang/srt/disaggregation/`

官方文档：

- slime PD 分离文档：[docs/zh/advanced/pd-disaggregation.md](../docs/zh/advanced/pd-disaggregation.md)
- slime SGLang Config 文档：[docs/zh/advanced/sglang-config.md](../docs/zh/advanced/sglang-config.md)
- SGLang EPD 文档：[../sglang/docs_new/docs/advanced_features/epd_disaggregation.mdx](../../sglang/docs_new/docs/advanced_features/epd_disaggregation.mdx)

## PD  ServerGroup 拓扑

在 slime 里，PD 分离不是“启动一个特殊 engine”，而是把同一个 model 拆成多个 `ServerGroup`。

`sglang-config` 里的结构是：

```yaml
sglang:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: prefill
        ...
      - worker_type: decode
        ...
```

`ServerGroup` 是 slime rollout 拓扑里很重要的一层：

```text
RolloutServer(model)
  -> ServerGroup(prefill)
    -> 多个 SGLangEngine Ray actor
  -> ServerGroup(decode)
    -> 多个 SGLangEngine Ray actor
```

也就是说，PD 模式下至少会有两类 group：

```text
prefill group:
  专门处理 prompt prefill。
  会注册成 prefill worker pool。

decode group:
  专门处理 token generation。
  会注册成 decode worker pool。
```

更完整地说，`worker_type` 支持这些类型：

```text
regular:
  普通 SGLang engine，同时处理 prefill 和 decode。

prefill:
  PD 分离里的 prefill worker。

decode:
  PD 分离里的 decode worker。

placeholder:
  占位 group，不启动 engine，常用于复杂资源布局。

encoder:
  EPD / encoder disaggregation 里的 encoder worker。
```

这篇主要只看 `prefill` 和 `decode`。`regular` 是非 PD 主线，`encoder` 是 EPD，不要混进来理解。

一个 model entry 里不要混用 `regular` 和 `prefill/decode`。PD 模式下，这个 model 应该由 prefill/decode group 组成。

## 一个 16 GPU 例子

官方文档里有一个简洁例子：

```yaml
sglang:
  - name: actor
    server_groups:
      - worker_type: prefill
        num_gpus: 4
        num_gpus_per_engine: 2
      - worker_type: decode
        num_gpus: 12
        num_gpus_per_engine: 4
```

这表示：

```text
prefill group:
  总共 4 张 GPU
  每个 prefill engine 2 张 GPU
  => 2 个 prefill distributed engines
  => 2 个 prefill worker URL 注册到 router

decode group:
  总共 12 张 GPU
  每个 decode engine 4 张 GPU
  => 3 个 decode distributed engines
  => 3 个 decode worker URL 注册到 router
```

最终 router 看到的是两个池子：

```text
prefill pool:
  prefill-0
  prefill-1

decode pool:
  decode-0
  decode-1
  decode-2
```

每个请求进来时，router 从两个池子里各选一个，组成一对：

```text
request A -> prefill-0 + decode-2
request B -> prefill-1 + decode-0
request C -> prefill-0 + decode-1
```

所以 PD 不是一个 prefill group 固定绑定一个 decode group，也不是某个 prefill engine 永久绑定某个 decode engine。绑定关系是在每个请求到达 router 时动态产生的。

## 一个大模型例子

`scripts/run-glm5.2-744B-A40B.sh` 里有一个更接近生产的大模型布局：

```yaml
sglang:
  - name: default
    server_groups:
      - worker_type: prefill
        num_gpus: 64
        num_gpus_per_engine: 64
        overrides:
          dp_size: 64
          ep_size: 64
          enable_dp_attention: true
          enable_dp_lm_head: true
          load_balance_method: follow_bootstrap_room
      - worker_type: decode
        num_gpus: 192
        num_gpus_per_engine: 64
        overrides:
          dp_size: 64
          ep_size: 64
          enable_dp_attention: true
          enable_dp_lm_head: true
          load_balance_method: round_robin
```

这里是：

```text
prefill:
  64 GPUs / 64 GPUs per engine = 1 个 prefill engine

decode:
  192 GPUs / 64 GPUs per engine = 3 个 decode engines
```

如果每个节点 8 卡，那么一个 64 GPU engine 会跨 8 个节点，对应 8 个 Ray `SGLangEngine` actor，但只有 `node_rank=0` 的 actor 会注册到 router。

因此 router 最终看到：

```text
prefill pool:
  1 个 prefill worker URL

decode pool:
  3 个 decode worker URL
```

这个例子也说明，prefill 和 decode 可以有完全不同的 GPU 总量、engine 数量和 SGLang overrides。PD 分离的意义之一就是让这两类 workload 独立扩缩和独立调参。

## group 到 worker URL 的计算

对任意一个 `ServerGroup`，可以先按下面理解：

```text
distributed engine 数量 = group.num_gpus / group.num_gpus_per_engine
```

这些 distributed engines 里，每个 engine 只有一个 `node_rank=0` HTTP 入口会注册到 router，所以：

```text
注册到 router 的 worker URL 数量 = distributed engine 数量
```

如果 engine 跨多节点，Ray actor 数会更多：

```text
每个 engine 的 Ray actor 数 = group.num_gpus_per_engine / num_gpus_per_node
```

例如：

```text
num_gpus_per_node = 8
group.num_gpus = 64
group.num_gpus_per_engine = 32

distributed engines = 64 / 32 = 2
每个 engine 跨 32 / 8 = 4 个节点
Ray actors = 2 * 4 = 8
router worker URLs = 2
```

这点和非 PD 场景一致：`all_engines` 记录所有 Ray actors，`engines` 只取每个 distributed engine 的 `node_rank=0`。

## PD 会不会天然多一倍卡

不是严格天然至少 2 倍，但很多场景会接近或达到 2 倍。

原因是 PD 分离后，prefill group 和 decode group 都要能独立跑这个模型，通常两边都会各自加载一份模型权重。

非 PD 的资源可以粗略写成：

```text
regular 总 GPU = regular_engine_num * gpus_per_regular_engine
```

PD 的资源变成：

```text
PD 总 GPU =
  prefill_engine_num * gpus_per_prefill_engine
  + decode_engine_num * gpus_per_decode_engine
```

如果原来是：

```text
1 个 8-GPU regular engine
```

改成最直观的 PD：

```text
1 个 8-GPU prefill engine
1 个 8-GPU decode engine
```

那就是 16 张卡，确实是 2 倍。

但 PD 不要求 prefill/decode 必须同规格。例如前面的 16 GPU 例子：

```text
prefill: 2 个 engine，每个 2 GPU => 4 GPU
decode:  3 个 engine，每个 4 GPU => 12 GPU
总共 16 GPU
```

这里不是简单把某个 8 卡实例复制一份，而是按 prefill 和 decode 的不同瓶颈重新配资源。

更准确的说法是：

```text
PD 会复制 serving 角色和模型权重，因此资源开销一定变大；
但是否严格至少 2 倍，取决于模型最小可运行 GPU 数、prefill/decode 的 TP/EP/DP 配置，以及目标吞吐比例。
```

如果一个大模型最少就要 8 张卡才能放下，那么 PD 至少要：

```text
8 卡 prefill + 8 卡 decode = 16 卡
```

这种情况下就是至少 2 倍。PD 的价值不是省卡，而是让 prefill 和 decode 两类 workload 独立扩缩、独立调参，用更多资源换更适合长 prompt、多轮、agentic rollout 的吞吐结构。

## slime 如何开启 PD router

在 `start_rollout_servers` 里，每个 model config 会判断：

```python
has_pd = model_cfg.has_pd_disaggregation
router_ip, router_port = _start_router(args, has_pd_disaggregation=has_pd, force_new=(model_idx > 0))
```

`has_pd_disaggregation` 的含义是：这个模型下面存在 `worker_type == "prefill"` 或 `worker_type == "decode"` 的 server group。

如果是 PD 场景，`_start_router` 会设置：

```python
router_args.pd_disaggregation = True
router_args.disable_circuit_breaker = True
router_args.disable_health_check = True
```

这里有几个点要注意：

- `pd_disaggregation=True`：router 进入 prefill/decode 双池模式。
- `disable_circuit_breaker=True`：避免 RDMA/KV transfer 短暂 timeout 被 router 误判成 worker 死亡。
- `disable_health_check=True`：slime 不使用 router 自带健康检查，健康检查由 `RolloutHealthMonitor` 做。

所以 slime 里 router 主要用于请求路由，不是健康检测的主控组件。

## prefill/decode 如何注册到 router

每个 `ServerGroup` 会启动一批 `SGLangEngine` Ray actor。

对于多节点 engine，只有 `node_rank == 0` 的 actor 会注册到 router：

```python
if self.node_rank == 0 and self.router_ip and self.router_port:
    worker_url = f"http://{self.server_host}:{self.server_port}"
```

新版本 router 走：

```python
payload = {
    "url": worker_url,
    "worker_type": self.worker_type,
}
```

如果是 prefill worker，还会额外注册：

```python
payload["bootstrap_port"] = bootstrap_port
```

因此 router 看到的是两类 worker：

```text
prefill worker:
  url
  worker_type = "prefill"
  bootstrap_port

decode worker:
  url
  worker_type = "decode"
```

decode worker 没有 `bootstrap_port`。这个 port 是 prefill worker 暴露给 decode worker 做 disaggregation bootstrap/KV 交接用的。

## worker 自身启动参数

slime 在构造 SGLang server args 时，根据 `worker_type` 写入不同参数。

prefill worker：

```python
kwargs["disaggregation_mode"] = "prefill"
kwargs["load_balance_method"] = "follow_bootstrap_room"
kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
```

decode worker：

```python
kwargs["disaggregation_mode"] = "decode"
kwargs["prefill_round_robin_balance"] = True
```

这里能看出来：

- prefill worker 会启动 bootstrap 服务，等待 decode 侧来完成 KV 交接。
- prefill worker 的 DP 路由方法是 `follow_bootstrap_room`。
- decode worker 是真正返回 decode 结果的入口。

## router 请求路径

以 `/generate` 为例，PD router 收到请求后会做几件事。

### 1. 选择一对 worker

router 从 prefill pool 和 decode pool 中各选一个：

```python
prefill_server, bootstrap_port, decode_server = lb.select_pair()
```

debug 版 `MiniLB` 里是随机选：

```python
pidx = random.randint(0, len(self.prefill_urls) - 1)
didx = random.randint(0, len(self.decode_urls) - 1)
return (
    self.prefill_urls[pidx],
    self.prefill_bootstrap_ports[pidx],
    self.decode_urls[didx],
)
```

正式 router 也遵循同一个抽象：每个请求需要一组 prefill/decode pair。

### 2. 注入 bootstrap 信息

router 会把被选中的 prefill worker 地址写进请求：

```python
modified_request.update(
    {
        "bootstrap_host": hostname,
        "bootstrap_port": bootstrap_port,
        "bootstrap_room": _generate_bootstrap_room(),
    }
)
```

这三个字段是 PD 协同的关键：

```text
bootstrap_host:
  prefill worker 所在 host。

bootstrap_port:
  prefill worker 的 disaggregation bootstrap port。

bootstrap_room:
  本次请求的唯一房间号，用来让 prefill/decode 对上同一次请求。
```

`bootstrap_room` 可以理解成请求级 rendezvous id。prefill 和 decode 都拿到同一个 room，后面 worker 内部就靠这个 room 完成 KV 交接。

### 3. 同时请求 prefill 和 decode

router 会把改写后的请求同时发给两个 worker：

```python
tasks = [
    session.post(f"{prefill_server}/{endpoint}", json=prefill_req),
    session.post(f"{decode_server}/{endpoint}", json=decode_req),
]

prefill_response, decode_response = await asyncio.gather(*tasks)
```

这里容易误解：普通路径下 `prefill_req` 和 `decode_req` 并不是两个完全不同的业务请求。它们通常就是同一个 `modified_request`，只是分别发给 prefill worker 和 decode worker。

slime 默认 rollout 发给 router 的原始 payload 大概是：

```json
{
  "input_ids": [151644, 872, 198, 9906, 11, 1148, 374, 697, 836, 30, 151645, 198],
  "sampling_params": {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_new_tokens": 1024,
    "stop": null,
    "stop_token_ids": null,
    "skip_special_tokens": false,
    "no_stop_trim": true,
    "spaces_between_special_tokens": false
  },
  "return_logprob": true
}
```

如果是多模态 sample，可能不是 `input_ids`，而是：

```json
{
  "text": "<image>\\nDescribe the image.",
  "image_data": ["data:image/jpeg;base64,..."],
  "sampling_params": {...},
  "return_logprob": true
}
```

router 选中：

```text
prefill_server = http://prefill-0:30000
bootstrap_port = 31000
decode_server = http://decode-2:30000
bootstrap_room = 88442211
```

然后生成 `modified_request`：

```json
{
  "input_ids": [151644, 872, 198, 9906, 11, 1148, 374, 697, 836, 30, 151645, 198],
  "sampling_params": {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "max_new_tokens": 1024,
    "stop": null,
    "stop_token_ids": null,
    "skip_special_tokens": false,
    "no_stop_trim": true,
    "spaces_between_special_tokens": false
  },
  "return_logprob": true,
  "bootstrap_host": "prefill-0",
  "bootstrap_port": 31000,
  "bootstrap_room": 88442211
}
```

普通情况下：

```python
prefill_req = modified_request
decode_req = modified_request
```

实际发送是：

```text
POST http://prefill-0:30000/generate  body = prefill_req
POST http://decode-2:30000/generate   body = decode_req
```

两边都拿到相同的 `bootstrap_host/bootstrap_port/bootstrap_room`，才能对上同一次请求。

区别在 worker 内部：

```text
prefill worker:
  自己是 disaggregation_mode=prefill。
  看到 bootstrap_room 后，执行 prompt prefill，并把本次请求的 KV 放到/传到这个 room 对应的位置。

decode worker:
  自己是 disaggregation_mode=decode。
  看到同一个 bootstrap_host/bootstrap_port/bootstrap_room 后，去对应 prefill worker 拿 KV，然后继续 decode。
```

如果开启了 SGLang router 的 `test_external_dp_routing` 测试路径，router 才会额外把请求 fork 成带 DP rank 的两个 payload：

```python
prefill_req["routed_dp_rank"] = p_rank
decode_req["routed_dp_rank"] = d_rank
decode_req["disagg_prefill_dp_rank"] = p_rank
```

这不是 slime 默认主线，可以先忽略。

最后通常返回 decode worker 的结果：

```python
ret_json = await decode_response.json()
return ORJSONResponse(content=ret_json, status_code=decode_response.status)
```

如果请求里需要 prompt logprob，router 还会把 prefill response 里的 input token logprobs 合并到 decode response 里。

这里要特别区分几类返回信息。

prefill worker 的 `/generate` 请求确实也会返回 HTTP response，但这个 response 的直接消费者通常是 router，不是 slime rollout 侧的用户逻辑。router 会检查 prefill response，并在需要时取其中的 `meta_info.input_token_logprobs`，再合并到 decode response。最终对上游返回的主体仍然是 decode response。

所以可以按下面理解：

```text
prefill response:
  给 router 用。
  主要承载 prefill 侧状态，以及 prompt/input token logprob 这类只能在 prefill 阶段拿到的信息。

decode response:
  是 router 返回给上游的主 response。
  承载 generated text、output token logprobs、finish_reason 等 decode 侧结果。

router:
  默认返回 decode response。
  对 input_token_logprobs 做特殊合并。
```

这也是为什么“prefill 和 decode 都返回内容，router 直接拼起来”这个说法只对一部分字段成立。当前明确看到的特殊拼接是 logprob，不是所有字段的通用拼接。

### 返回字段：logprob 与 R3

slime 在 rollout 请求里会设置 `return_logprob=true`，因此 PD 下需要处理一个自然问题：

```text
input token logprob 在 prefill 侧产生；
output token logprob 在 decode 侧产生；
上游希望拿到一个完整 response。
```

SGLang router 对这个问题有专门逻辑：

```text
prefill response 里取 meta_info.input_token_logprobs
decode response 作为主 response
把 input_token_logprobs merge 到 decode response 的 meta_info 里
```

非流式和流式路径都有对应处理。流式时，decode worker 不会自己带 input logprob，router 会把 prefill logprob merge 到 decode stream chunk 的 `meta_info` 里。

R3 / routed expert replay 不能简单套用 logprob 的结论，因为它需要的不是一个标量数组合并，而是按 token 维度对齐的 routed experts 序列。

slime 侧开启 `--use-rollout-routing-replay` 时，会做两件事：

```text
server args:
  enable_return_routed_experts = true

request payload:
  return_routed_experts = true
```

训练侧期望拿到的是和 token 序列对齐的完整 routed experts：

```text
len(routed_experts) == len(tokens) - 1
shape ~= [tokens - 1, num_layers, moe_router_topk]
```

这里有一个容易误判的地方：不能只看未打 patch 的 SGLang checkout，也不能只看开源 router 的普通 logprob merge 逻辑。slime 的 Docker runtime 里 pin 了一个带 slime patch 的 router wheel：

```text
sglang_router-0.3.2, build v0.3.2-9daabcd
```

而 slime 历史里也有明确 commit：

```text
[docker] Update SGLang patch for PD R3 routed experts (#2173)
```

这个 commit 做了几类事情：

```text
router:
  更新到支持 PD R3 的 slime 版 sglang_router wheel。
  这部分是预编译 wheel，具体 merge/response 代码不在当前 repo 源码里。

SGLang patch:
  在 PD prefill transfer success 后收集 routed_experts。
  decode 侧设置 routed_experts_start_len = prompt_len，只返回 prompt 之后的 routed_experts。

slime:
  后续又补了 routed_experts_start_len 解析，允许分段拼接 routed_experts。
```

### R3 在 PD 下的具体流程

#### 1. slime 打开 R3 请求开关

当 `--use-rollout-routing-replay` 打开时，slime 做两件事。

启动 SGLang worker 时：

```text
enable_return_routed_experts = true
```

每个 rollout 请求里：

```text
return_routed_experts = true
```

这表示 prefill worker 和 decode worker 都会尝试返回 routed experts。

#### 2. router 仍然先做普通 PD fanout

router 收到请求后，仍然按 PD 主流程选一对 worker：

```text
prefill worker = P
decode worker = D
bootstrap_room = R
```

然后把同一个请求加上 bootstrap 信息，并发发给两边：

```text
POST P/generate:
  return_routed_experts = true
  bootstrap_host/port/room = P/R

POST D/generate:
  return_routed_experts = true
  bootstrap_host/port/room = P/R
```

R3 不是另起一条请求链路，而是挂在这条 PD fanout 链路上。

#### 3. prefill 侧负责 prompt/prefix 段

prefill worker 执行 prompt prefill 时，模型 forward 会捕获 routed experts。

普通未打 patch 的 PD prefill 路径里，request 在 KV transfer 完成后很快结束；如果不额外处理，prefill 侧捕获到的 routed experts 可能不会进入最终 response。

#2173 的关键补丁是在 prefill transfer 成功时补了一步：

```text
if poll == KVPoll.Success:
  if req.return_routed_experts:
    _maybe_collect_routed_experts(req)
  release_kv_cache(req)
  finish prefill request
```

这一步的意义是：

```text
在 prefill request 结束前，把 prompt/prefix 侧 routed_experts 收到 req 上；
随后 tokenizer/output 路径才有机会把它放进 prefill response 的 meta_info。
```

也就是说，prefill response 里应该能提供 prompt/prefix 段的 `meta_info["routed_experts"]`。

#### 4. decode 侧负责 prompt 之后的尾段

decode worker 也会收到 `return_routed_experts=true`。但它不应该再返回 prompt/prefix 那段，否则会和 prefill 侧重复。

#2173 的 decode 侧补丁在 scheduler 创建请求后加了这个逻辑：

```text
if disaggregation_mode == DECODE:
  req.routed_experts_start_len = len(req.origin_input_ids)
```

注释也说明了语义：

```text
Prefill returns prompt-side routing;
decode only owns rows after the prompt.
```

结合 SGLang routed experts API：

```text
routed_experts_start_len = N
=> 返回 full_rows[N:]
```

decode 侧设置 `N = prompt_len`，意思就是：

```text
prefill 负责 prompt/prefix 段；
decode 只返回 prompt 后继续生成产生的 routed experts 尾段。
```

#### 5. router / 上游 response 必须恢复成 slime 能理解的形态

到这里，worker 侧天然会形成两段数据：

```text
prefill response:
  meta_info.routed_experts = prefix rows

decode response:
  meta_info.routed_experts = suffix rows
  内部语义是 routed_experts_start_len = prompt_len
```

但 slime rollout client 最终不会直接消费两个 worker response。它只会消费 router 返回给它的 response，然后调用：

```text
sample.append_response_tokens(..., meta_info=output["meta_info"])
```

所以 patched router 必须把两段数据整理成 slime 能理解的形态。合理的 wire shape 有两种：

```text
形态 A：router 先合并
  router 把 prefill prefix rows + decode suffix rows 拼成完整 rows。
  上游只看到一个 meta_info["routed_experts"]。
  routed_experts_start_len 可以省略或为 0。

形态 B：router/streaming 暴露分段
  先让 slime 收到 prefix rows，start_len = 0。
  后续再让 slime 收到 suffix rows，routed_experts_start_len = prompt_len。
  slime Sample 根据 routed_experts_start_len 继续拼接。
```

#2173 当时的 slime `Sample` 还不支持 `routed_experts_start_len`，只接受完整 `meta_info["routed_experts"]`。所以 #2173 那条非流式目标路径更像是形态 A：router 最终要给 slime 一个标准的完整 `routed_experts`。

#2185 之后，slime `Sample` 增加了 `routed_experts_start_len` 支持，所以形态 B 也有了 sample 侧承接能力。

#### 6. slime Sample 做最终 shape 校验和拼接

当前 slime 侧解析的是：

```text
meta_info["routed_experts"]
meta_info.get("routed_experts_start_len", 0)
```

它不解析自定义拆分字段，例如：

```text
pd_prefill_routed_experts
pd_decode_routed_experts
```

解析逻辑是：

```text
start_len = meta_info.get("routed_experts_start_len", 0)
expected_rows = len(sample.tokens) - 1 - start_len

如果 start_len == 0:
  当前 routed_experts 被视为从序列起点开始。
  sample.rollout_routed_experts = current_rows

如果 start_len > 0:
  sample 必须已经有 existing routed_experts。
  sample.rollout_routed_experts =
    existing[:start_len] + current_rows
```

最后训练侧期望的是：

```text
sample.rollout_routed_experts.shape[0] == len(sample.tokens) - 1
```

如果 router 没拼好，或者分段 response 的 `start_len` / row count 对不上，slime 会在 shape 校验或训练侧 routing replay 处报错。

### R3 版本和排查口径

后续如果升级 SGLang base image、替换 router wheel，或者直接使用本地 SGLang checkout，需要重新确认这些逻辑还在运行时里。只看 commit 标题不够，实际要看：

```text
prefill 侧是否会在 PD transfer success 后收集 routed_experts；
decode 侧是否设置 routed_experts_start_len = prompt_len；
router response 到上游时，是已经合成完整 routed_experts，还是按 routed_experts_start_len 分段交给 slime；
slime Sample 是否能按 routed_experts_start_len 正确拼接。
```

因此 R3 的准确结论是：

```text
logprob:
  PD router 有明确 merge 逻辑。

routed_experts / R3:
  slime Docker 里 pin 的 patched SGLang/router 方向上是支持 PD R3 的。
  这个支持依赖特定 SGLang patch + slime 版 router wheel + slime 的 routed_experts_start_len 解析。
  不能拿一个未打 patch 的 SGLang checkout 直接判断“已经支持”。
  排查时要同时确认 prefill collect、decode start_len、router response 形态、Sample 拼接四件事。
```

所以 router 确实参与了 PD 请求流程，但它做的是：

```text
选 prefill/decode pair
生成 bootstrap_room
注入 bootstrap_host/bootstrap_port/bootstrap_room
同时转发请求
返回 decode 结果
必要时合并 input_token_logprobs
在 slime patched runtime 中按 R3 协议处理 routed_experts
```

它没有做：

```text
执行 prefill forward
执行 decode forward
搬运 KV cache
等待 KV transfer 完成
管理每个 TP rank 的细节
无协议地通用拼接所有 prefill/decode 元信息
```

这里的“按 R3 协议处理”不是指 router 可以自动理解所有字段，而是指 slime pin 的 patched SGLang/router 组合对 `routed_experts` 和 `routed_experts_start_len` 有约定。其他自定义字段不能默认复用这个能力。

## worker 内部如何接上

请求进 SGLang worker 后，`bootstrap_host/bootstrap_port/bootstrap_room` 会进入 SGLang 的 request object。

SGLang 里相关字段包括：

```python
bootstrap_host
bootstrap_port
bootstrap_room
routed_dp_rank
disagg_prefill_dp_rank
```

对于 prefill worker，`load_balance_method="follow_bootstrap_room"` 时，DP controller 会根据 `bootstrap_room` 选择 DP rank：

```python
target_rank = req.bootstrap_room % len(self.workers)
```

这保证同一个请求的 prefill 子任务能稳定落到某个 DP lane 上。decode 侧则使用同一组 bootstrap 信息去找对应 prefill worker 产生的 KV。

真正的 KV 传输和状态等待在 SGLang 的 disaggregation 模块里完成，例如：

```text
../sglang/python/sglang/srt/disaggregation/
```

这里才是 RDMA/Mooncake/Mori 等 KV transfer 后端真正工作的地方。

## PD 下 offload/onload 的粒度

PD 下，一个 `RolloutServer` 里会有多个 `ServerGroup`：

```text
RolloutServer(default)
  -> ServerGroup(prefill)
  -> ServerGroup(decode)
```

`RolloutServer.offload()` 会遍历所有 group：

```python
def offload(self):
    handles = []
    for g in self.server_groups:
        handles.extend(g.offload())
    return ray.get(handles) if handles else []
```

所以从一个 model 的 serving 单元看，prefill group 和 decode group 会一起进入 offload/onload 调度。

但每个 group 内部还会判断：

```python
if not self.needs_offload:
    return []
```

因此准确说是：

```text
同一个 RolloutServer 下，所有 needs_offload=True 的 group 会一起 offload/onload。
```

如果 prefill 和 decode 都与训练 GPU colocate：

```text
prefill group offload
decode group offload
```

如果某个 group 的 GPU 不和 Megatron 重叠，`needs_offload=False`，这个 group 会跳过。

这也说明 PD colocate 的成本更高：不是只有一套 regular serving engine 进入显存释放/恢复生命周期，而是 prefill/decode 两套 serving 角色都可能要管理权重、KV cache、CUDA graph 等显存占用。

## 为什么 prefill 全挂后服务不可用

PD 模式下，一个完整请求需要：

```text
prefill worker + decode worker
```

如果只挂掉一个 prefill engine，但 prefill pool 里还有其他 prefill worker：

```text
服务还能继续，只是 prefill 容量下降。
```

如果 prefill pool 全挂：

```text
router 无法选出 prefill/decode pair
新请求无法完成 prompt prefill
服务对新请求基本不可用
```

decode worker 还活着也没用，因为 decode 不能凭空得到 prompt prefill 后的 KV。

反过来，如果 decode pool 全挂，也是不可用。prefill 能算 prompt，但没有 decode worker 继续生成和返回最终结果。

## 故障摘除边界

slime 关闭了 router 自带健康检查：

```python
router_args.disable_health_check = True
```

所以健康检查主逻辑在 `RolloutHealthMonitor`。

每个 `ServerGroup` 有自己的 monitor。PD 场景下通常是：

```text
prefill ServerGroup -> 一个 RolloutHealthMonitor
decode  ServerGroup -> 一个 RolloutHealthMonitor
```

monitor 只检查该 group 的 `engines`，也就是每个 distributed engine 的 `node_rank=0` actor。

失败时：

```text
发现某个 engine 健康检查失败
  -> kill 这个 distributed engine 的所有 node ranks
  -> all_engines 对应位置置 None
  -> engine.shutdown() 尝试从 router 删除该 worker
```

`shutdown()` 里会删除 router 中的 worker：

```python
all_workers = requests.get(f"http://{self.router_ip}:{self.router_port}/workers").json()["workers"]
for worker in all_workers:
    if worker["url"] == worker_url:
        worker_id = worker["id"]
        requests.delete(f"http://{self.router_ip}:{self.router_port}/workers/{worker_id}")
```

因此正常情况下，失败 worker 会被从 router 的 worker pool 里摘掉，后续请求不会再选到它。

如果删除 router worker 失败，代码只打 warning，然后继续 kill 本地进程。这种情况下理论上 router 可能短时间保留一个已经坏掉的 worker URL，后续请求可能失败。后续 recover 成功后，新 worker 会重新注册到 router。

## PD 下 abort 的当前语义

slime 现在有 abort 支持，但在 PD 场景下它是粗粒度的 worker cleanup 语义，不是精确的 per-request prefill/decode 拼接语义。

rollout 侧 abort 入口大致是：

```text
slime/rollout/sglang_rollout.py::abort
  -> 从 router 查询 worker 列表
  -> 对每个 worker 调 /abort_request
  -> 轮询 /v1/loads?include=core
  -> 等所有 worker load 归零
```

真正发到 worker 的请求是：

```json
{
  "abort_all": true
}
```

也就是说，slime 并不是记录每个 rollout request 被 router 分配到了哪一对 prefill/decode worker，然后对这一对 worker 发精确 abort。它是从 router 拿当前注册的 worker URL 集合，对所有 worker 做 abort-all。

PD 下这个方式之所以能覆盖 prefill 和 decode，是因为 slime 启动 worker 时会分别注册：

```text
prefill worker:
  worker_type = "prefill"

decode worker:
  worker_type = "decode"
```

router 的 `/workers` 会包含这两类 worker。abort 逻辑拿到这个列表后，会同时打到 prefill workers 和 decode workers。

因此可以把当前能力描述成：

```text
支持:
  PD 下把所有注册的 prefill/decode workers abort 到 idle。
  用于 rollout 结束、超时、partial rollout 截断后的资源清理。

不支持或不应依赖:
  按单个请求精确找到 router 当时选中的 prefill/decode pair。
  让 prefill abort response 和 decode abort response 返回两段 partial 内容。
  在 router 里把 abort 产生的 partial logprob/R3/文本通用拼接起来。
```

SGLang worker 内部本身有 PD abort 处理。prefill 侧会清理 bootstrap / inflight queue，decode 侧会清理 prealloc / transfer queue，KV transfer 相关对象也会收到 abort。这保证的是 worker 内部资源释放和请求终止，不等价于 slime/router 能稳定拿到拼好的 partial rollout。

### 非流式 partial rollout

非流式 `/generate` 的自然形态是：

```text
请求发出
  -> worker 完整生成
  -> 最后一次性返回 JSON
```

abort 时，slime 额外调用的是 worker 的 `/abort_request`，这个控制接口本身只返回 abort 请求是否被接收，不返回生成到一半的文本或 logprob。原来的 `/generate` 请求在被 abort 后，能不能返回一个带 partial 内容的 JSON，取决于当时请求处于什么状态、tokenizer manager 是否已经有 request state、以及 worker 如何结束该请求。

所以非流式在 PD 下不能假设：

```text
prefill 返回 partial A
decode 返回 partial B
router 把 A/B 拼好
slime 拿到完整 partial sample
```

当前没有这样的强语义。

更实际的判断是：

```text
非流式 abort:
  更适合作为资源清理和停止继续生成。
  partial rollout 能收多少取决于原 rollout task 是否已经返回可用 sample。
  在 PD 下尤其不要假设 router 会合并 abort partial。
```

### 流式 partial rollout

流式 rollout 更适合做 abort 后的数据保留，因为数据不是等最终 response 才落到 slime。

slime 的 streaming rollout 会持续消费 SSE chunk。SGLang 默认流式输出是 cumulative 形态，slime 每收到一个 chunk，就用当前累计的 `output_token_logprobs` / token 信息更新 sample。这样 abort 发生时，已经收到的 chunk 已经在 sample 上了。

PD 下流式 logprob 还要经过 router：

```text
prefill response:
  提供 input_token_logprobs

decode stream:
  提供生成 token、output_token_logprobs、finish_reason

router:
  把 prefill input_token_logprobs merge 到 decode stream chunk
```

因此流式路径在 abort 场景更稳，是因为：

```text
partial token/logprob 已经提前到达 slime；
abort 只需要停止后续生成并清理 worker；
不依赖 abort response 返回完整 partial 内容。
```

但这同样只适用于当前已经明确处理的字段。对于 R3 / routed experts，要确认运行时确实是 slime pin 的 patched SGLang/router 组合；如果换成未打 patch 的 SGLang 或非 slime 版 router，不能因为流式路径可用就默认 R3 也可用。

## 一句话模型

PD 请求可以按下面这条线记：

```text
client/slime
  -> router
    -> 选择 prefill worker
    -> 选择 decode worker
    -> 生成 bootstrap_room
    -> 给请求注入 prefill bootstrap 地址
    -> 并发发给 prefill/decode
      -> prefill 做 prompt prefill 并准备 KV
      -> decode 通过 bootstrap 信息拿到 KV 后继续 decode
    -> router 返回 decode response
```

所以，“PD worker 的协同是在 router 中做的吗？”更精确的回答是：

```text
请求级协同是在 router 中发起的；
执行级协同在 SGLang worker 内部完成。
```

## 如果不用 router

有些 RL 系统在非 PD 场景下不使用 router，而是维护一组 SGLang URL，例如一共 8 个 URL，然后对每个 rollout 请求随机选择一个 URL 发送。

这种方式在普通 serving 下可以成立，因为每个 URL 都是完整的 regular worker：

```text
request -> random regular worker -> prefill + decode -> response
```

但 PD 分离后不能再把所有 URL 当成一个无角色的 worker pool。URL 变成了两类：

```text
prefill URL:
  只负责 prompt prefill，并准备/发送 KV。

decode URL:
  负责等待 prefill KV，然后继续 decode 生成最终结果。
```

所以 PD 下如果不用 SGLang router，RL 系统自己就必须实现一个 client-side router。最小职责包括：

```text
维护两个 pool:
  prefill_urls = [...]
  decode_urls = [...]

每个请求选择一对 worker:
  prefill = pick(prefill_urls)
  decode = pick(decode_urls)

生成 bootstrap 信息:
  bootstrap_host = prefill host
  bootstrap_port = prefill disaggregation bootstrap port
  bootstrap_room = unique request id / random room

改写请求:
  注入 bootstrap_host/bootstrap_port/bootstrap_room

并发发送:
  POST prefill/generate
  POST decode/generate

返回:
  以 decode response 作为主 response
```

这还只是基础请求路径。如果 rollout 需要训练可用的数据，还要继续处理一些边界：

```text
logprob:
  input_token_logprobs 来自 prefill response。
  output_token_logprobs 来自 decode response。
  client-side router 需要自己 merge。

streaming:
  prefill 侧 input logprob 和 decode stream chunk 需要在客户端合并。
  每个 chunk 的 meta_info 都要保持和 slime 的 sample 解析逻辑兼容。

abort:
  需要同时覆盖 prefill/decode 两侧。
  如果想做 per-request abort，还要记录每个 request 当时选中的 pair。
  如果只是 abort_all，也要知道所有 prefill/decode worker 的 URL。

health/retry:
  需要维护 worker 健康状态。
  失败 worker 要摘除。
  prefill/decode 任一侧失败时，要定义整请求如何失败或重试。

负载均衡:
  不能把 prefill 和 decode 混成一个随机池。
  要分别考虑 prompt 长度、decode 长度、worker load、DP lane 等因素。

R3 / routed experts:
  需要复刻 slime patched runtime 的 routed_experts_start_len 协议。
  prefill 段、decode 段如何返回，以及 sample 侧如何拼接，都要自己定义清楚。
  不能把普通 logprob merge 逻辑直接套到 routed_experts 上。
```

因此，不用 router 并不会让 PD 协同消失，只是把这些职责从 `sg-router` 搬到了 RL client 里。

可以总结成：

```text
非 PD:
  随机打多个 regular URL 可以是一个简单可行的方案。

PD:
  随机打多个 URL 不成立。
  必须有一个 coordinator 负责 prefill/decode 配对、bootstrap 注入、并发请求和结果合并。

coordinator 可以是:
  SGLang router
  或 RL 系统自己实现的 client-side router
```

对 slime 当前实现来说，继续依赖 SGLang router 更稳。否则需要在 slime rollout client 里显式引入一层 `PDClient` / `PDRolloutRouter`，并重新定义 logprob、streaming partial、abort、R3 等行为。

## PD 分离和共卡训推

PD 分离可以和共卡训推一起用。

它们不是互斥概念：

```text
PD 分离:
  rollout serving 拓扑如何拆分，是否分成 prefill/decode group。

共卡训推 colocate:
  训练和 rollout 是否共享同一批 GPU，并通过 offload/onload 交替使用显存。
```

所以可以有：

```text
非 PD + 非共卡
非 PD + 共卡
PD + 非共卡
PD + 共卡
```

slime 脚本里也有 PD + colocate 的例子，例如 `scripts/run-glm5.2-744B-A40B.sh`：rollout 侧是 `1 prefill engine + 3 decode engines`，同时使用 `--colocate`。

### 和非 PD 共卡的区别

非 PD 共卡的 rollout 拓扑是：

```text
RolloutServer
  -> ServerGroup(regular)
```

PD 共卡的 rollout 拓扑是：

```text
RolloutServer
  -> ServerGroup(prefill)
  -> ServerGroup(decode)
```

所以共卡对象从“一组 regular engines”变成了“多组 server groups”。

### offload/onload 更复杂

非 PD 共卡时，基本是一组 regular engines 和训练侧交替占卡。

PD 共卡时，要分别看：

```text
prefill group needs_offload?
decode group needs_offload?
```

`RolloutServer.offload()` 会遍历所有 group：

```python
for g in self.server_groups:
    handles.extend(g.offload())
```

但每个 group 内部会先判断：

```python
if not self.needs_offload:
    return []
```

因此 PD + colocate 下准确说是：

```text
同一个 RolloutServer 里，所有 needs_offload=True 的 prefill/decode group 都会一起 offload/onload。
```

如果 prefill 和 decode 都与训练 GPU 重叠：

```text
prefill group offload
decode group offload
```

如果 rollout GPU 多于训练 GPU，后面的某些 group 可能是 rollout-only，`needs_offload=False`，就不会 offload。

### group 顺序会影响共卡边界

slime 按 `sglang-config` 里的 group 顺序分配 GPU offset：

```text
group 0 -> 先占一段 rollout GPU slot
group 1 -> 再占下一段 rollout GPU slot
```

然后判断：

```python
needs_offload = args.offload_rollout and group_abs_start < megatron_num_gpus
```

所以在 `--colocate` 且 `rollout_num_gpus > actor/训练 GPU 数` 的情况下，哪些 group 被认为与训练侧重叠，会受 `server_groups` 顺序影响。

通常配置是：

```yaml
server_groups:
  - worker_type: prefill
  - worker_type: decode
```

这意味着 prefill 先占前面的 rollout GPU slot，decode 再占后面的 slot。如果只有前一段 GPU 与训练重叠，可能出现 prefill 需要 offload，而一部分 decode group 是 rollout-only 的情况。

### router 形态不同

非 PD 共卡：

```text
router -> regular worker pool
```

PD 共卡：

```text
router -> prefill worker pool + decode worker pool
```

因此 PD 共卡时，请求进入 router 后仍然要做：

```text
选择 prefill worker
选择 decode worker
注入 bootstrap_host/bootstrap_port/bootstrap_room
同时转发给 prefill/decode
```

共卡只改变 GPU 生命周期管理，不改变 PD 请求配对逻辑。

### 权重和显存成本更高

非 PD 共卡只有 regular serving 角色。

PD 共卡下，prefill 和 decode 通常各自加载一份模型权重，并且各自有 serving 侧的显存结构：

```text
weights
KV cache
CUDA graph
runtime buffers
```

所以 PD + colocate 的成本不只是“多一个 router 配对步骤”，而是 prefill/decode 两套 serving 角色都可能进入 offload/onload 生命周期。

可以把区别总结成：

```text
非 PD colocate:
  一套 regular rollout engine 和训练侧交替占卡。

PD colocate:
  prefill/decode 多套 rollout group 和训练侧交替占卡；
  router 还要做 prefill/decode 请求配对；
  资源规划、group 顺序和 offload 边界都更敏感。
```

## 开启 EPD 后的变化

这里的 EPD 是：

```text
Encoder-Prefill-Decode Disaggregation
```

不是 `ep_size` 的 Expert Parallel。

PD 只有两段：

```text
prefill group
decode group
```

EPD + PD 变成三段：

```text
encoder group
prefill group
decode group
```

配置上会出现：

```yaml
server_groups:
  - worker_type: encoder
  - worker_type: prefill
  - worker_type: decode
```

只要某个 model 下存在 `worker_type == "encoder"`，slime 就会认为开启了 encoder disaggregation：

```python
has_epd = model_cfg.has_encoder_disaggregation
```

### 启动顺序变成两阶段

普通 PD 可以一轮启动所有 group。

EPD 不行，因为 prefill / regular language worker 需要知道 encoder worker 的 URL。

slime 的启动顺序是：

```text
Phase 1:
  先启动 encoder group
  等 encoder ready
  收集 encoder_urls

Phase 2:
  再启动 prefill / regular / decode group
  给 prefill / regular 注入 encoder_urls
```

代码上，只有这些 group 会被注入：

```text
worker_type in ("prefill", "regular")
```

注入内容是：

```python
language_only = True
encoder_urls = [...]
```

decode group 不注入 `encoder_urls`。

### encoder group 不注册到 router

`SGLangEngine._register_to_router()` 里有这个判断：

```python
if self.worker_type == "encoder":
    return
```

所以 EPD 下 router 仍然只直接管理：

```text
prefill worker pool
decode worker pool
```

encoder worker 不在 router 的 worker pool 里。

因此 EPD 的请求路径不是：

```text
router -> encoder -> prefill -> decode
```

更准确是：

```text
client/slime
  -> router
    -> 选择 prefill worker
    -> 选择 decode worker
    -> 发请求给 prefill/decode
      -> prefill/language-only worker 根据 encoder_urls 调用 encoder
      -> encoder 做视觉/音频等 encode
      -> prefill 用 encoder embedding 做 language prefill
      -> decode 继续生成
```

也就是说，router 仍然只做 PD 的 prefill/decode 配对；encoder 调用是 SGLang worker 内部的 EPD 逻辑。

### EPD 改变的是 prefill 前面的工作

普通 PD：

```text
prefill:
  处理 prompt prefill

decode:
  逐 token 生成
```

EPD + PD：

```text
encoder:
  处理 multimodal encoder 部分，例如视觉/音频 encode

prefill:
  处理 language prefill，并接收 encoder 输出

decode:
  逐 token 生成
```

所以 EPD 主要把原来 prefill worker 里的一部分 encoder 工作拆出去，适合 VLM、音频、多模态等 encoder 很重的场景。

### EPD + colocate 的影响

如果 EPD 和共卡一起用，一个 model 下会变成：

```text
RolloutServer
  -> ServerGroup(encoder)
  -> ServerGroup(prefill)
  -> ServerGroup(decode)
```

`RolloutServer.offload()` 仍然遍历所有 group：

```python
for g in self.server_groups:
    handles.extend(g.offload())
```

所以 EPD + colocate 下，可能参与 offload/onload 的 group 从两个变成三个：

```text
encoder group needs_offload?
prefill group needs_offload?
decode group needs_offload?
```

哪些 group 真正 offload，仍然看 `needs_offload`。如果 encoder/prefill/decode 都落在与训练侧重叠的 GPU slot 上，它们都会进入 offload/onload 生命周期。

这比 PD + colocate 更敏感：

```text
PD colocate:
  prefill/decode 两类 serving group 和训练侧交替占卡。

EPD + PD colocate:
  encoder/prefill/decode 三类 serving group 和训练侧交替占卡。
```

注意 encoder group 虽然不注册到 router，但它仍然是一个真实的 rollout server group：要占 GPU、要加载对应权重/encoder 组件，也要参与健康、recover、offload/onload 等生命周期管理。
