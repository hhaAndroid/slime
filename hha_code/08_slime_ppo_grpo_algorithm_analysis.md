# slime PPO / GRPO 算法实现分析

本文只抽算法核心，不做逐行源码解读。结论基于当前仓库中的这些核心位置：

- 训练总循环：`train.py`
- actor / critic 训练调度：`slime/backends/megatron_utils/actor.py`
- loss 与 logprob/value 抽取：`slime/backends/megatron_utils/loss.py`
- advantage / returns / KL / PPO clip 工具：`slime/utils/ppo_utils.py`
- micro-batch、sequence packing、CP loss reducer：`slime/backends/megatron_utils/data.py`、`slime/backends/megatron_utils/cp_utils.py`
- 参数与约束：`slime/utils/arguments.py`

## 1. 总体定位

slime 里 PPO 和 GRPO 共享一套“rollout -> 训练数据 -> Megatron forward/backward -> 更新 actor -> 同步 rollout engine 权重”的训练框架。区别主要集中在：

1. 是否需要 critic。
2. advantage / returns 怎么算。
3. policy loss 使用同一个 PPO-style clipped objective，但 advantage 的来源不同。
4. PPO 额外有 value loss 和 critic 更新链路。

参数层面：

- `--advantage-estimator ppo` 会在参数校验阶段自动令 `args.use_critic = True`。
- 非 PPO 的 `grpo/gspo/cispo/reinforce...` 默认 critic-free。
- PPO 下 critic GPU 规模被设置为和 actor 相同：`critic_num_gpus_per_node = actor_num_gpus_per_node`，`critic_num_nodes = actor_num_nodes`。
- PPO 会强制 `offload_train=True`，因为 actor / critic / rollout engine 之间存在更强的显存轮换压力。

## 2. 先从算法层面理解 PPO

在看 slime 的进程和数据传输之前，可以先把 PPO 理解成下面一句话：

> actor 根据 advantage 判断哪些动作应该增加或降低概率，但每次更新不能离生成数据时的旧策略太远；critic 学习预测未来回报，为 actor 提供低方差的 advantage。

PPO 中最重要的不是 actor 和 critic 谁先运行，而是下面三份量在一次 PPO 更新期间的语义必须固定：

- rollout 数据：旧策略实际生成的 token、reward 和 mask。
- `old_log_prob`：生成这些 token 的旧策略概率，作为 policy ratio 的分母。
- `old_value`：critic 更新前对这些 token 的 value 预测，既用于 GAE，也作为 value clipping 的中心。

actor 和 critic 都会更新自己的模型参数，但不会把已经生成的 token 或环境 reward 改掉。它们只是在原始轨迹之上计算 `log_probs`、`values`、`advantages`、`returns` 等派生量。

### 2.1 PPO 中各个模型的职责

一次 PPO 训练涉及以下逻辑角色：


| 角色                        | 作用                                                                | 是否训练               |
| ------------------------- | ----------------------------------------------------------------- | ------------------ |
| rollout / behavior policy | 生成本轮 response；同步训练中通常就是更新前的 actor                                 | 本轮 rollout 完成后视为固定 |
| actor / policy            | 输出每个 token 的概率，目标是提高高 advantage token 的概率、降低低 advantage token 的概率 | 是                  |
| critic / value function   | 预测从当前位置开始的期望未来回报 `V(s_t)`                                         | 是                  |
| reference policy          | 可选的固定参考模型，用于 KL 约束，避免 actor 偏离基础模型过远                              | 通常否                |


这里的 old actor 和 new actor 通常不是必须常驻的两个物理模型：

- `old` 表示生成当前 batch 时的策略语义，通常保存成 detached `old_log_probs` 就够了。
- `new` 表示当前正在反向传播和更新的 actor。
- 同步在线训练开始更新时两者参数可能完全相同，但 actor optimizer step 之后 new policy 会逐渐偏离 old policy。
- 异步或 replay buffer 场景下，必须使用 rollout 时记录的 logprob，或者维护正确版本的 old actor，才能保证 ratio 的分母正确。

### 2.2 一轮 PPO 的完整闭环

忽略工程并行后，一轮 PPO 可以画成：

```text
actor_old 生成轨迹
  -> 固定 tokens / rewards / masks / old_log_probs
  -> critic_old 计算 old_values
  -> 构造 token rewards
  -> GAE 计算 advantages 和 returns
  -> critic 用 returns 更新 value function
  -> actor 用 advantages 和 old_log_probs 更新 policy
  -> actor_new 同步给 rollout engine
  -> 下一轮 rollout
```

展开后是以下步骤。

#### 第 1 步：旧策略生成 rollout

对 prompt `x`，rollout policy 依次采样 response token：

```text
y_t ~ pi_old(. | x, y_<t)
```

环境或 reward model 对完整 response 给出 sequence reward，例如数学答案正确为 `1`、错误为 `0`。这一阶段得到的 token 和 reward 是事实数据，后续 actor/critic 训练不会改写它们。

#### 第 2 步：记录 old policy 概率

对已经采样出来的每个 token，保存或重算：

```text
old_logp_t = log pi_old(y_t | x, y_<t)
```

它不是训练目标，而是后面限制策略更新幅度的锚点。PPO 更新过程中必须把它当常量，不能让梯度流入 `old_logp_t`。

#### 第 3 步：critic 更新前先计算 old value

critic 对 response 中每个 token 位置预测：

```text
old_value_t = V_old(s_t)
```

其中 `s_t` 可以理解为 prompt 加上当前 token 之前的上下文。value 不是“当前 token 的即时 reward”，而是从该状态继续执行后，预期还能获得多少累计回报。

必须先保存 `old_value` 再更新 critic，因为：

- GAE 需要一套固定 baseline。
- clipped value loss 需要用 `old_value` 限制本轮 value function 的变化。
- actor 应使用 critic 更新前的这套 value 计算本轮 advantage，而不是混用更新前后的 value。

#### 第 4 步：把 sequence reward 变成 token reward

LLM 任务经常只在 response 结束时得到一个 scalar reward。最简单的 token reward 是：

```text
r_t = 0                         # 非最后 token
r_T = sequence_reward           # 最后 token
```

如果启用 reference KL reward shaping，则每个 token 还会有偏离 reference policy 的惩罚：

```text
r_t = -beta * KL_t
r_T += sequence_reward
```

因此即使任务 reward 只在末尾出现，前面 token 也可以通过 GAE 获得 credit，并且可在每个位置受到 KL 约束。

#### 第 5 步：用 GAE 计算 advantage 和 return

先计算 TD residual：

```text
delta_t = r_t + gamma * old_value_{t+1} - old_value_t
```

再从后向前累计：

```text
advantage_t = delta_t + gamma * lambda * advantage_{t+1}
return_t = advantage_t + old_value_t
```

两个输出的用途不同：

- `advantage_t` 给 actor 使用，表示这个实际 token 比 critic 的预期好多少或差多少。
- `return_t` 给 critic 使用，是本轮 critic 要拟合的 value target。

直觉上：

- `advantage_t > 0`：这个 token 所在轨迹比 baseline 好，actor 应提高它的概率。
- `advantage_t < 0`：比 baseline 差，actor 应降低它的概率。
- critic 越准确，advantage 的方差通常越小，actor 梯度越稳定。

`gamma` 控制远期 reward 的折扣，`lambda` 控制 bias/variance trade-off。slime 默认两者都是 `1.0`，此时没有 KL 且只有 terminal reward 时，return 接近把最终 reward 传播给整段 response。

#### 第 6 步：更新 critic

critic 重新 forward 得到可训练的 `value_new`，拟合上一步的 `return`。PPO 不只使用普通 MSE，还把 value 更新限制在 old value 附近：

```text
value_clipped_t = old_value_t
                  + clip(value_new_t - old_value_t,
                         -value_clip, value_clip)

value_loss_t = max(
    (value_new_t     - return_t)^2,
    (value_clipped_t - return_t)^2
)
```

随后只更新 critic 参数及其 optimizer state。actor 参数不会因为 critic backward 而变化。

#### 第 7 步：更新 actor

actor 当前 forward 得到 `logp_new_t`，并与固定的 `old_logp_t` 构造 importance ratio：

```text
ratio_t = exp(logp_new_t - old_logp_t)
```

如果只最大化 `ratio_t * advantage_t`，一次梯度更新可能让策略变化过大。PPO 用 clip 截断收益：

```text
objective_t = min(
    ratio_t * advantage_t,
    clip(ratio_t, 1 - eps, 1 + eps) * advantage_t
)

policy_loss = -mean(objective_t)
```

因此：

- 正 advantage token 的概率可以提高，但超过上界后不再获得额外收益。
- 负 advantage token 的概率可以降低，但超过下界后不再获得额外收益。
- clip 不是保证模型参数或 KL 一定落在某个范围内，而是让超出 ratio 区间的样本不再推动同方向的激进更新。

actor loss 还可以叠加 entropy bonus、reference KL loss、TIS 等项，但 clipped policy objective 是 PPO 的主体。

#### 第 8 步：进入下一轮

actor 更新完成后，把 `actor_new` 权重同步到 rollout engine。下一轮生成时，它就成为新的 `pi_old`：

```text
本轮 actor_new == 下一轮 rollout actor_old
```

这形成 on-policy 闭环。若 rollout 和训练异步执行，数据可能来自更旧版本的 actor，就需要 rollout logprob、TIS 等机制处理 policy mismatch。

### 2.3 一个三 token 的 GAE 小例子

假设 response 有 3 个 token，只有答对时最后得到 reward `1`：

```text
rewards    = [0.0, 0.0, 1.0]
old_values = [0.2, 0.4, 0.3]
gamma = 1.0
lambda = 1.0
```

从后向前算：

```text
delta_3 = 1.0 - 0.3       =  0.7  -> advantage_3 = 0.7
delta_2 = 0.3 - 0.4       = -0.1  -> advantage_2 = -0.1 + 0.7 = 0.6
delta_1 = 0.4 - 0.2       =  0.2  -> advantage_1 =  0.2 + 0.6 = 0.8

returns = advantages + old_values = [1.0, 1.0, 1.0]
```

这说明：

- 最终 reward 通过 GAE 传给了前面的 token。
- critic 对第一个位置只预期 `0.2`，实际最终得到 `1.0`，所以它的 advantage 最大。
- actor 会倾向于提高这三个实际 token 的概率。
- critic 则学习让三个位置的 value 更接近 return `1.0`。

若答案错误、最终 reward 很低，advantage 可能为负，actor 就会降低这条轨迹中 token 的概率。

### 2.4 算法数据与模型状态要分开看

理解 slime 的多进程实现时，可以把状态分成四类：


| 内容                                 | 本轮是否变化               | 如何流动                                        |
| ---------------------------------- | -------------------- | ------------------------------------------- |
| rollout tokens、原始 rewards、masks    | 不应被训练改写              | rollout manager 同时提供给 actor/critic          |
| values、logprobs、advantages、returns | 会在各 worker 本地计算或加入字典 | 大多本地派生；critic 的 old values 需要传给 actor       |
| actor 参数和 optimizer state          | actor step 时变化       | actor DP/TP/PP 组内部同步；更新后再同步到 rollout engine |
| critic 参数和 optimizer state         | critic step 时变化      | critic DP/TP/PP 组内部同步，不与 actor 共享           |


所以“actor 和 critic 都读取同一批数据”不等于它们共享一个可变训练字典：

- 两边分别从同一个 rollout data reference 取得自己的本地数据。
- critic 本地加入 `values/advantages/returns` 并更新 critic。
- critic 只把更新前算出的 `values` 返回给 actor。
- actor 把这些 values 加入自己的本地数据，再自行计算 advantages/returns 并更新 actor。
- critic 本地对字典的修改不会自动写回 rollout manager，也不会被 actor 自动看见。

## 3. 为什么 actor 和 critic 可以复用同一套代码

你的观察是对的：slime 没有为 actor 和 critic 分别实现两套完全独立的 trainer。二者都使用 `MegatronTrainRayActor`，只是运行在两套独立的 Ray/Megatron worker group 中，并通过 `role="actor"` 或 `role="critic"` 选择不同分支。

更准确地说，不是“只有两个进程”，而是：

```text
actor RayTrainGroup
  ├─ actor worker rank 0: MegatronTrainRayActor(role="actor")
  ├─ actor worker rank 1: MegatronTrainRayActor(role="actor")
  └─ ...

critic RayTrainGroup
  ├─ critic worker rank 0: MegatronTrainRayActor(role="critic")
  ├─ critic worker rank 1: MegatronTrainRayActor(role="critic")
  └─ ...
```

每组内部可能还包含 DP、TP、PP、CP 等多个进程。两组使用同一个 Python 类和大部分函数，但模型参数、optimizer state、distributed process group 和进程内存彼此独立。

### 3.1 为什么可以共用训练骨架

actor 和 critic 虽然优化目标不同，但从大模型训练系统的角度看，它们做的事情高度相似：

1. 输入都是同一批 `prompt + response` tokens。
2. 都要经过相同的 Transformer backbone。
3. 都需要 sequence packing、micro-batch schedule 和 loss mask。
4. 都要使用 Megatron 的 TP/PP/DP/CP forward/backward。
5. 都需要独立的 optimizer、scheduler、checkpoint 和 offload 生命周期。
6. 都是在 response token 位置产生输出并计算 loss。

最大的区别只发生在 Transformer 最后一层之后：

```text
                         ┌─ actor output head: hidden -> vocab logits
tokens -> Transformer ---┤
                         └─ critic output head: hidden -> scalar value
```

因此没有必要复制数据加载、分布式初始化、模型并行、训练循环、显存管理等大量代码。slime 复用公共训练骨架，只把“输出是什么”和“如何算 loss”做成角色分支。

### 3.2 `role` 如何把同一个类变成两个角色

actor 和 critic 的构建链路可以简化为：

```text
create_training_models
  ├─ allocate_train_group(role="actor")
  │    └─ MegatronTrainRayActor.init(args_actor, role="actor")
  │
  └─ allocate_train_group(role="critic")
       └─ MegatronTrainRayActor.init(args_critic, role="critic")
```

两边都执行：

```text
initialize_model_and_optimizer(args, role)
  -> build model
  -> build independent optimizer
  -> build independent scheduler
  -> load role-specific checkpoint
```

`role` 主要控制三个分叉点。

#### 分叉点 1：模型输出头

两边主体都是同一种 `GPTModel`。actor 保留语言模型 vocabulary output head，输出形状近似为：

```text
[tokens, vocab_size]
```

critic 在 model provider 中把最后的 output layer 替换为：

```text
LinearForLastLayer(hidden_size, 1)
```

所以 critic 输出形状近似为：

```text
[tokens, 1]
```

这也是 actor checkpoint 可以作为 critic backbone 初始化来源的原因：Transformer 主体结构相同，只有最后的 value head 与 vocabulary head 不同。当前实现还会处理 critic output head 与 checkpoint shape 不匹配时的重新初始化。

#### 分叉点 2：训练入口

同一个 `MegatronTrainRayActor.train()` 先读取 rollout 数据，然后只做一次 role dispatch：

```python
if self.role == "critic":
    result = self.train_critic(...)
else:
    self.train_actor(...)
```

之后二者又会汇合到公共的 Megatron `train(...)`，只是 `loss_type` 不同：

```text
actor  -> loss_type="policy_loss" -> policy_loss_function
critic -> loss_type="value_loss"  -> value_loss_function
```

也就是说，外层的训练机械过程是一样的：

```text
DataIterator
  -> Megatron forward
  -> role-specific loss function
  -> backward
  -> optimizer.step()
  -> scheduler.step()
```

#### 分叉点 3：角色专属能力

actor 侧还负责策略训练特有的逻辑：

- 计算 old actor logprob。
- 可选计算 reference logprob。
- 可选 teacher / OPD forward。
- 计算 policy ratio、entropy、policy KL、TIS/OPSM 等。
- 训练后把 actor 权重同步给 rollout engine。
- 可选使用 routing replay。

critic 不需要这些能力。critic 不会加载 actor 侧的 ref/teacher 权重；在 role-specific YAML 解析路径中，critic args 还会显式关闭 KL reward shaping、OPD 和自定义 actor advantage hook，并把 output head 设置成与 embedding 解耦。critic 的专属工作是：

- forward 得到 token-level `values`。
- 基于 old values 计算 GAE returns。
- 计算 clipped value loss。
- 把更新前的 old values 返回给 actor。

### 3.3 actor 和 critic 的相同点与不同点

先看相同点：


| 相同点       | 说明                                                            |
| --------- | ------------------------------------------------------------- |
| Worker 实现 | 都是 `MegatronTrainRayActor`                                    |
| 输入轨迹      | 都读取同一轮 rollout 的 tokens、rewards、masks                         |
| Backbone  | 都使用 GPT/Transformer 主体                                        |
| 数据管线      | 共用 DP partition、`DataIterator`、packing 和 micro-batch schedule |
| 并行框架      | 共用 Megatron TP/PP/DP/CP 训练代码                                  |
| 训练机械过程    | 都执行 forward、loss、backward、optimizer step、scheduler step       |
| 生命周期      | 共用 checkpoint、wake/sleep、offload、日志和显存管理框架                    |
| 配置来源      | 都先继承公共 CLI，再通过 YAML 做 role-specific override                  |


再看不同点：


| 维度         | Actor                                           | Critic                                |
| ---------- | ----------------------------------------------- | ------------------------------------- |
| 模型输出       | 每个 token 的 vocabulary logits                    | 每个 token 的 scalar value               |
| 输出头        | `hidden_size -> vocab_size`                     | `hidden_size -> 1`                    |
| 训练目标       | 提高高 advantage token 的概率，降低低 advantage token 的概率 | 拟合 GAE return                         |
| Loss       | clipped policy loss，可叠加 entropy/KL/TIS 等        | clipped value loss                    |
| Loss 输入    | `old_log_probs + advantages`                    | `old_values + returns`                |
| 本地派生量      | logprobs、ref logprobs、advantages、returns        | values、advantages、returns             |
| 附加模型       | 可临时持有 ref、teacher、old actor 权重                  | 不需要                                   |
| 对外输出       | 新 actor 权重同步给 rollout engine                    | old values 返回给 actor                  |
| 参数更新       | 只更新 actor 参数和 actor optimizer state             | 只更新 critic 参数和 critic optimizer state |
| Checkpoint | actor 自己的 load/save                             | critic 可有独立 load/save                 |


虽然两者可以使用相同的 optimizer 类型和公共超参数，但 optimizer 实例绝不共享：

```text
actor optimizer  -> 只引用 actor parameters
critic optimizer -> 只引用 critic parameters
```

因此 actor 的 `optimizer.step()` 不会修改 critic，critic 的 `optimizer.step()` 也不会修改 actor。YAML 中常见 actor 使用较小的 `lr`、critic 使用较大的 `lr`，只是因为两者学习目标和收敛速度不同，并不是因为它们共享 optimizer。

### 3.4 两套进程分别拿到什么数据

RolloutManager 先按 DP rank 打包基础数据，并把每个分片放入 Ray Object Store 或 NIXL transport。基础数据大致包括：

```text
rollout_data_ref[dp_rank]
  ├─ tokens
  ├─ response_lengths / total_lengths
  ├─ rewards / raw_reward
  ├─ loss_masks
  ├─ rollout_ids / partition
  ├─ micro_batch_indices
  ├─ num_microbatches / global_batch_sizes
  └─ optional rollout_log_probs / teacher_log_probs / replay metadata
```

同一个 `rollout_data_ref` 会分别传给 critic group 和 actor group。每个 worker 根据自己的 DP rank 取得相同语义的数据分片：

```text
base rollout object
  ├─ ray.get(...) -> critic 进程内的 local rollout_data
  └─ ray.get(...) -> actor  进程内的 local rollout_data
```

这两个 local dict 不是同一个跨进程可变 Python 字典。critic 在自己的 dict 中 `update()` 新字段，不会让 actor 自动看到；actor 的本地修改也不会写回 critic 或 RolloutManager。

### 3.5 critic 到 actor 只需要传 old values

当前 PPO 主循环中的跨角色依赖是：

```text
value_refs = critic_model.async_train(rollout_id, rollout_data_ref)

actor_model.async_train(
    rollout_id,
    rollout_data_ref,
    external_data=value_refs,
)
```

critic 的本地过程是：

```text
critic_local_data = fetch(base_rollout_data)
critic_local_data["values"] = critic_old.forward(tokens)
critic_local_data["advantages"], critic_local_data["returns"] = GAE(...)
train critic with values + returns
return {"values": old_values_on_cpu}
```

actor 的本地过程是：

```text
actor_local_data = fetch(base_rollout_data)
old_log_probs = rollout 时记录的 logprob 或训练前 actor forward 的结果
actor_local_data["values"] = external_data["values"]
actor_local_data["advantages"], actor_local_data["returns"] = GAE(...)
train actor with old_log_probs + advantages
```

所以 critic 到 actor 当前只传：

```text
external_data = {"values": list[CPU Tensor]}
```

这里传的是 critic 更新前 forward 得到的 old values。critic 的 `advantages` 和 `returns` 不传给 actor，actor 收到 values 后会重新计算它们。

这样做有几个原因：

1. actor 侧拥有自己的 old policy logprob、reference logprob 和 KL 配置。
2. advantage normalization 可能需要在 actor 自己的 DP/CP group 中执行 collective。
3. 只传 values 的接口更小，避免把 critic 本地修改过的整份训练字典重新序列化。
4. actor 和 critic 分别保留完整、自洽的本地 batch，减少跨进程共享可变状态。

实际传递时是每个 critic worker 返回一个 Ray ref，并与同 rank 的 actor worker 一一对应。只有 PP last stage 产生并消费 `values`；其他 pipeline stage 返回空字典，因为 value loss 和 GAE 都在 last stage 使用这些输出。

### 3.6 哪些东西不会通过 `external_data` 传递

`external_data` 不是整个系统的通用同步通道，它只是当前 critic -> actor 的旁路数据接口。以下内容不经过它：

- 基础 rollout 数据：通过 `rollout_data_ref` 进入两套 worker。
- actor/critic 各自的梯度：通过各自 Megatron process group 的 collective 同步。
- actor/critic 参数和 optimizer state：始终各自独立，不在两者之间同步。
- actor 新权重：通过 weight update 模块同步到 SGLang rollout engine，而不是同步给 critic。
- critic 本地算出的 returns/advantages：不会传给 actor。
- actor 本地算出的 logprobs/advantages：不会传给 critic。

可以把完整数据通道总结为：

```text
RolloutManager --rollout_data_ref--> Actor group
       |
       +-------rollout_data_ref----> Critic group

Critic group ----external_data: values----> Actor group

Actor group -----updated actor weights----> Rollout engine

Actor group 内部  <---- Megatron collectives ----> actor ranks
Critic group 内部 <---- Megatron collectives ----> critic ranks
```

### 3.7 为什么当前要求两者并行拓扑一致

当前 PPO 中 critic 训练资源跟随 actor，二者使用同一组 placement group，并通过 offload 轮流占用同一批 GPU。actor 和 critic 的 Megatron 并行拓扑也要求一致。

除了便于复用 GPU，这还能让跨角色 values 按 worker/rank 直接对齐：

```text
critic worker rank i 的 values -> actor worker rank i
```

如果 actor 和 critic 使用不同的 DP/PP/CP 拓扑，就必须额外实现 values 的全局 gather、重新 partition 和 redistribution。当前实现通过相同拓扑避免了这层复杂的数据重分布。

但需要特别注意：**“资源数量相同”是代码强制不变量，“并行拓扑相同”目前主要是配置约束，并没有被完整地硬校验。**

- `critic_num_nodes` 和 `critic_num_gpus_per_node` 会被强制设置成 actor 的值。
- 不使用 `--megatron-config-path` 时，两者继承同一套 CLI 并行参数，通常自然一致。
- 使用 role-specific YAML 时，parser 只忽略 `num_nodes/num_gpus_per_node`，仍允许分别覆盖 `tensor_model_parallel_size`、`pipeline_model_parallel_size`、`context_parallel_size` 等字段。
- 当前 role config 单测也覆盖了 actor/critic 使用不同 TP 值的“解析成功”行为，说明 parser 层没有拒绝拓扑不一致。

因此用户可以构造出“总 GPU 数相同，但 TP/PP/CP 分解不同”的配置。此时可能发生：

1. actor 初始化后先把自己的 `train_parallel_config` 写入 RolloutManager。
2. critic 随后初始化，再把同一个字段覆盖为 critic layout。
3. RolloutManager 最终按 critic layout 生成唯一一套 DP/micro-batch schedule。
4. actor 再读取这套 ref；如果 DP size 不同，会直接触发 ref 数量断言或索引问题。
5. 即使 DP size 恰好相同，PP last-stage、CP shard 或 worker-rank 语义也可能不一致，导致 critic values 无法与 actor worker 一一正确对齐。

所以目前应该把以下规则当作必须由配置作者维护的 invariant：

```text
actor topology == critic topology
```

最稳妥的配置方式是把所有拓扑参数只写在公共 CLI 中，role YAML 只覆盖：

```text
lr / optimizer / scheduler / load / save / warmup 等非拓扑参数
```

更健壮的实现应该增加两层校验：

1. 创建 actor/critic 前比较 role args 中全部拓扑字段，不一致就给出明确错误。
2. RolloutManager 第二次收到 train layout 时与已保存 layout 比较，而不是静默覆盖；如果不一致立即失败。

在这些校验加入前，这个限制能否满足取决于配置，而不是框架能够无条件保证。

## 4. 通用训练数据调度（已拆分为独立文档）

RolloutManager 的 micro-batch packing、DP 分配、梯度累计数量关系，以及 static/dynamic、FLOPs balancing 和 EP-aware 调度限制，属于 slime 通用训练基础设施，已移至：

- [slime RolloutManager 训练数据调度与负载均衡分析](21_slime_rolloutmanager_train_scheduling_analysis.md)

## 5. slime 中的 PPO 工程流程

### 5.1 rollout 阶段

`RolloutManager` 调用 rollout function / SGLang engine 对 prompt 采样，产出 `Sample`。随后 `_convert_samples_to_train_data` 将样本转换为训练字段：

- `tokens`: prompt + response token。
- `response_lengths`: response 长度。
- `rewards` / `raw_reward`: reward 值。
- `truncated`: 是否截断。
- `loss_masks`: response token 级 mask；如果样本被移除，mask 全 0。
- `rollout_ids`: 用于把一个 rollout 拆出的多个训练样本重新聚合。
- `rollout_mask_sums`: 同一个 rollout 的所有 sibling 样本 mask 总和，供 loss reducer 做 per-rollout mean。
- 可选 `rollout_log_probs`、top-p replay、routing replay、teacher logprob 等字段。

这里的关键不是简单按 sample 平均，而是提前构造 `rollout_mask_sums`。当一个 rollout 拆成多个训练 sample，或者这些 sample 被分到不同 micro-batch 时，loss 仍然能按“一个 rollout”而不是“多个样本”计权。

### 5.2 数据切分与 micro-batch

`_split_train_data_by_dp` 根据 DP 配置、总长度、`global_batch_size` 和 rollout id 构造每个 DP rank 的 partition 与 micro-batch schedule。训练侧 `DataIterator` 按这个 schedule 取数据。

`get_batch` 会做三件和算法语义相关的事：

1. 保留原始 list 形式 token 到 `unconcat_tokens`，后续从 packed logits 中恢复每个 response 的 logprob/value。
2. 把多条序列 concat / pad 成 Megatron 需要的 packed sequence。
3. 将 response-only 的 `loss_masks` 对齐到完整 token stream，并处理 Context Parallelism 的切片。

所以 loss 实际只作用在 response token 上，prompt token 只是模型条件上下文。

### 5.3 PPO 的 actor / critic 调度

训练主循环中，每个 `rollout_id` 的大致顺序是：

1. `rollout_manager.generate()` 生成一批训练数据。
2. 如果启用 PPO，即 `use_critic=True`：
  - 先调用 `critic_model.async_train(...)`。
  - critic 先 forward 当前 value，计算 PPO GAE returns，然后用 value loss 更新 critic。
  - critic 返回这批 rollout 的 old values 给 actor。
3. 如果当前步不是 critic-only：
  - actor 接收同一批 rollout 和 critic 返回的 `values`。
  - actor 按需计算 ref logprob、teacher logprob、old actor logprob。
  - actor 计算 advantages / returns。
  - actor 用 policy loss 更新。
4. 保存 checkpoint。
5. actor 将新权重同步到 rollout engine。
6. 周期性 eval。

`--num-critic-only-steps` 会让前 N 个 rollout 只训练 critic，不训练 actor。这在 PPO 初期很实用，因为随机或未充分训练的 value head 会直接污染 GAE。

### 5.4 critic 训练链路

critic 的 `train_critic` 逻辑：

1. `forward_only(get_values)` 从 value head 输出中抽取每个 response token 的 value。
2. 调用 `compute_advantages_and_returns`。
3. 设置 `loss_type = "value_loss"`。
4. 走 Megatron `train(...)` 做 backward / optimizer step。
5. 在 PP last stage 返回本轮旧 value 给 actor。

注意：critic 用于 value loss 的 `old_values` 和 actor 用于 GAE 的 `values` 是同一次 critic forward 的结果。这符合 PPO 的“old value baseline”语义。

### 5.5 actor 训练链路

actor 的 `train_actor` 逻辑：

1. 可选切换到 reference model，计算 `ref_log_probs`。
2. 可选切换到 teacher model，计算 `teacher_log_probs`，用于 OPD。
3. 切换到 `old_actor` 或当前 actor，计算 old policy logprob：
  - 默认用训练 actor forward 得到的 logprob。
  - 如果 `keep_old_actor=True`，可以显式维护 old actor。
  - 如果 `use_rollout_logprobs=True`，policy ratio 直接用 rollout 时记录的 logprob。
4. 注入 critic 返回的 `values`。
5. 计算 PPO advantages / returns。
6. 记录 rollout 指标。
7. 用 `policy_loss` 更新 actor。
8. 备份新 actor 权重，并按需更新 ref。

## 6. PPO advantage / returns 计算

PPO 走的是 `compute_advantages_and_returns(..., advantage_estimator="ppo")`。

### 6.1 KL 先变成 token-level reward shaping

如果 `args.kl_coef == 0`，KL 为零张量。如果不为 0，会先计算当前策略和 reference 策略的近似 KL：

```text
kl_t = approx_kl(logp_old_or_rollout_t, ref_logp_t)
```

然后 PPO 分支把 scalar reward 转成 token-level reward：

```text
r_t = -kl_coef * kl_t
r_last += sequence_reward
```

也就是除最后一个有效 token 外，reward 主要是 KL 惩罚；最终任务 reward 加在 response 的最后一个 token 上。Context Parallelism 下只有 `cp_rank == 0` 给最后 token 加 sequence reward，避免重复加。

这里要区分两个 KL 入口：

- `kl_coef`: reward shaping，在 advantage / return 计算前进入 reward。
- `kl_loss_coef`: 额外加到最终 actor loss 上的 KL loss。

参数校验禁止两者同时非零：`kl_coef != 0` 和 `kl_loss_coef != 0` 只能选一个，避免重复约束。

### 6.2 GAE

PPO 使用 GAE：

```text
delta_t = r_t + gamma * V_{t+1} - V_t
A_t = delta_t + gamma * lambda * A_{t+1}
R_t = A_t + V_t
```

实现上先把 CP rank 上的 value / reward gather 成完整 response，再按 batch padding 到同一 `max_response_len`，最后调用 `chunked_gae`。`chunked_gae` 的数学结果等价于标准 backward GAE，但用 chunk 内并行 scan + chunk 间 recurrent state 传播，减少长序列上纯 Python 反向循环的串行依赖。

默认 `gamma=1.0`、`lambd=1.0`，这会退化成 undiscounted Monte Carlo 风格的 return-minus-value；实际 PPO 可通过命令行调小。

### 6.3 advantage normalization

如果 `--normalize-advantages` 开启，slime 会：

1. 拼接当前训练 step 的所有 token-level advantages。
2. 用 loss mask 只统计有效 response token。
3. 跨 data parallel group 做 masked whitening。
4. 再按 sample 切回 list。

这一步对 PPO 非强制，但测试配置里通常打开。它降低 reward scale / value scale 对 policy loss 的影响。

## 7. policy loss 计算

PPO actor 使用 `policy_loss_function`。核心输入：

- 当前 actor logprob：`log_probs`
- old logprob：`old_log_probs`
- advantages：`advantages`
- loss mask：`loss_masks`

### 7.1 response token logprob 抽取

`get_log_probs_and_entropy` 做的是：

1. 从模型 logits `[1, T, V]` 中构造 shifted target token。
2. 用 tensor parallel aware 的 `_VocabParallelLogProbEntropy` 计算目标 token logprob。
3. 可选计算 entropy。
4. 从 packed sequence 中抽取每个 sample 的 response 部分。
5. 如果是 allgather CP，再重分布回训练使用的 CP layout。

这块有几个优化点：

- 一次性在完整 `[T, V]` logits 上算 logprob，再按 sample 切，不对每条样本重复走 softmax。
- vocab parallel logprob / entropy 用自定义 autograd function，避免不必要的大中间张量。
- `log_probs_chunk_size` 可把 logprob 计算分块，控制峰值显存。
- 当 `entropy_coef=0` 时，entropy 可作为指标返回，但不保留 entropy backward 所需的大激活。
- top-p rollout replay 只影响 logprob 的 normalization mask，entropy 仍用完整 logits。

### 7.2 PPO clipped objective

slime 里定义：

```text
ppo_kl_t = old_logp_t - logp_t
ratio_t = exp(logp_t - old_logp_t) = exp(-ppo_kl_t)

loss1_t = - ratio_t * A_t
loss2_t = - clip(ratio_t, 1 - eps_clip, 1 + eps_clip_high) * A_t
pg_loss_t = max(loss1_t, loss2_t)
```

最终 policy gradient loss 是 masked reducer 后的均值：

```text
L_pg = reducer(pg_loss_t)
```

`eps_clip_high` 如果不设，默认等于 `eps_clip`。因此既支持经典对称 clip，也支持上界更宽的非对称 clip，例如 GRPO 常见 `eps_clip=0.2, eps_clip_high=0.28`。

实现也支持 dual-clip PPO：`policy_loss_function` 会把 `args.eps_clip_c` 传给 `compute_policy_loss`。默认值为 `None` 时就是标准 PPO clip；显式配置后会对负 advantage 场景再加一层下界约束。

### 7.3 entropy

entropy 项：

```text
entropy_loss = reducer(entropy_t)
loss = L_pg - entropy_coef * entropy_loss
```

因为训练最小化 loss，所以 `- entropy_coef * entropy` 会鼓励更高熵。默认 `entropy_coef=0`。

### 7.4 KL loss

如果 `--use-kl-loss` 开启，会额外计算 actor 当前 logprob 与 reference logprob 的近似 KL：

```text
kl_loss_t = approx_kl(logp_t, ref_logp_t)
loss += kl_loss_coef * reducer(kl_loss_t)
```

支持的 KL estimator：

- `k1`: `logp - ref_logp`
- `k2`: `0.5 * (logp - ref_logp)^2`
- `k3` / `low_var_kl`: `exp(ref_logp - logp) - 1 - (ref_logp - logp)`

`low_var_kl` 会 clamp 到 `[-10, 10]` 增强数值稳定性。`--use-unbiased-kl` 会乘上 `exp(logp - old_logp)` 作为 importance ratio。

### 7.5 TIS / off-policy 修正

当 rollout 策略和训练时 old policy 不完全一致时，可开启 TIS：

```text
tis = exp(train_old_logp - rollout_logp)
tis_weight = clamp(tis, tis_clip_low, tis_clip)
pg_loss *= tis_weight
```

这相当于用截断重要性采样降低 off-policy mismatch 的偏差，同时控制方差。slime 还支持自定义 TIS / rejection sampling 函数，并在 rejection 后重建 reducer，保证 numerator / denominator 语义一致。

### 7.6 OPSM

`--use-opsm` 是 Off-Policy Sequence Masking：

1. 计算 sequence-level KL。
2. 如果样本 advantage 为负且 sequence KL 超过阈值 `opsm_delta`，将该序列的 policy loss mask 掉。

它主要针对 off-policy 或策略漂移较大的坏样本，避免负 advantage 在高 mismatch 区间产生不稳定更新。

## 8. value loss 计算

critic 使用 `value_loss_function`。输入是：

- 当前 value head 输出：`values`
- old value：`old_values`
- GAE 得到的 returns：`returns`

PPO-style clipped value loss：

```text
values_clipped = old_values + clamp(values - old_values, -value_clip, value_clip)
surr1 = (values_clipped - returns)^2
surr2 = (values - returns)^2
value_loss_t = max(surr1, surr2)
L_v = reducer(value_loss_t)
```

同时记录：

```text
value_clipfrac = mean(abs(values - old_values) > value_clip)
```

这个设计和 PPO policy clip 类似，限制 critic 单步 value 更新幅度，避免 value function 过快漂移。

## 9. reducer 与 loss 归一化

slime 的 reducer 是理解 loss 数值的关键。

默认不是全 token 平均，而是 per-sample / per-rollout mean：

```text
sum_i [ sum_t loss_{i,t} * mask_{i,t} / denom_i ]
```

其中 `denom_i` 默认是 sample 自己的 `loss_mask.sum()`；在 rollout manager 预先提供 `rollout_mask_sums` 后，同一 rollout 的 sibling sample 共用整个 rollout 的 mask 总和作为 denominator。这样一个 rollout 拆成多个 sample 时，不会因为拆分数量更多而在 loss 中权重更大。

如果开启 `--calculate-per-token-loss`，reducer 改为 token sum，最后由 Megatron reduce 阶段除以全局 token 数：

```text
sum_{i,t} loss_{i,t} * mask_{i,t} / sum_{i,t} mask_{i,t}
```

训练 loss 返回 Megatron 前还会按 micro-batch、global batch size、DP/CP 做缩放，以适配 Megatron 的 gradient accumulation 和 parallel reduction。

## 10. 优化手段总结

### 10.1 算法稳定性

- PPO ratio clipping：限制 actor 相对 old policy 的更新幅度。
- value clipping：限制 critic value 漂移。
- KL reward shaping 或 KL loss：约束 actor 不要偏离 reference。
- advantage whitening：减小 reward scale 对梯度的影响。
- entropy bonus：可选鼓励探索。
- critic-only warmup：PPO 初期先让 critic 学一段。
- TIS / OPSM：处理 rollout policy 与 train policy mismatch。
- `keep_old_actor`：可显式维护 old actor，用于更标准的 PPO ratio。

### 10.2 长序列与并行优化

- packed sequence：多样本拼接，减少 padding 浪费。
- Context Parallelism 支持：response logprob/value/advantage 在 CP 切片间 gather / slice，保证语义等价。
- `chunked_gae`：长 response 的 GAE 用 chunk scan 优化。
- full logits 一次性 logprob：减少 per-sample softmax 调用。
- vocab parallel fused logprob/entropy autograd：减少显存和通信开销。
- `log_probs_chunk_size`：控制 logprob 计算峰值显存。
- `allgather_cp` 死锁保护：空 token rank 也加 `0 * logits.sum()` 保证 backward 图完整。

### 10.3 系统优化

- actor / critic / rollout engine 通过 Ray actor 编排。
- PPO 下自动 train offload，降低 actor+critic 共存显存压力。
- actor 更新后同步权重到 SGLang rollout engine，支持 disk / NCCL / colocate tensor 等模式。
- 支持 dynamic batch size、按 token 数构造 micro-batch。
- Megatron optimizer / scheduler 按实际 step global batch size 推进。
- 可选 stateless Adam、optimizer state reset。
- 支持 routing replay / rollout routing replay，降低 MoE 训练和 rollout 路由不一致带来的 mismatch。

## 11. GRPO 实现与 PPO 差异

### 11.1 GRPO 的 reward normalization

GRPO 的核心在 rollout 后处理。对于同一 prompt 的 `n_samples_per_prompt` 个 response：

```text
reward_group = [r_1, ..., r_K]
normalized_reward_i = r_i - mean(reward_group)
```

如果 `grpo_std_normalization=True`，进一步：

```text
normalized_reward_i /= std(reward_group) + 1e-6
```

这就是 group-relative advantage 的来源。若 `n_samples_per_prompt == 1`，代码会自动关闭 std normalization，因为单样本组没有有意义的组内 std。

### 11.2 GRPO returns / advantages

GRPO 分支：

```text
returns_i,t = normalized_reward_i
advantages_i,t = returns_i,t
```

也就是说，GRPO 没有 value function，没有 GAE，也没有 critic returns。一个 response 内所有有效 token 拿到同一个 group-relative scalar advantage。

KL 如果通过 `kl_coef` 做 reward shaping，在当前 `get_grpo_returns` 里不会被扣进 returns；GRPO 更常见的路径是 `--use-kl-loss` + `--kl-loss-coef`，即 KL 作为 actor loss 的额外项，或者设置 coef 为 0 只做观测。

### 11.3 GRPO policy loss

GRPO 仍然走 `policy_loss_function`，所以 actor loss 形式和 PPO 一样是 PPO-style ratio clipping：

```text
ratio_t = exp(logp_t - old_logp_t)
pg_loss_t = max(-ratio_t * A_t,
                -clip(ratio_t, 1 - eps_clip, 1 + eps_clip_high) * A_t)
```

差异只在 `A_t`：

- PPO: `A_t` 来自 critic value + token reward + GAE。
- GRPO: `A_t` 来自组内 reward normalization，广播到所有 response token。

### 11.4 PPO vs GRPO 对比表


| 维度            | PPO in slime                                         | GRPO in slime                                   |
| ------------- | ---------------------------------------------------- | ----------------------------------------------- |
| 是否需要 critic   | 需要，`advantage_estimator=ppo` 自动 `use_critic=True`    | 不需要                                             |
| reward 使用方式   | scalar reward 加到最后 token，KL 可作为 token reward shaping | reward 先做组内相对归一化，再广播到 token                     |
| advantage     | GAE: `A_t = delta_t + gamma lambda A_{t+1}`          | `A_t = normalized_group_reward`                 |
| returns       | `R_t = A_t + V_t`                                    | `returns = advantages`                          |
| value loss    | 有 clipped value loss                                 | 无                                               |
| policy loss   | PPO clipped objective                                | 同一个 PPO clipped objective                       |
| KL 控制         | `kl_coef` reward shaping 或 `kl_loss_coef` final loss | 通常用 `use_kl_loss` / `kl_loss_coef`，也可只记录 KL     |
| 资源            | actor + critic + rollout，并且 actor/critic 并列占 GPU     | actor + rollout，资源更省                            |
| 初期稳定性         | 依赖 critic 质量，可用 critic-only warmup                   | 依赖组内 reward 方差，需要每 prompt 多采样                   |
| 长 response 信号 | GAE 给每个 token 不同 advantage                           | 同一 response 内 token advantage 相同                |
| 适用场景          | 有稳定 value baseline、想做更经典 PPO 的任务                     | 数学/代码等 outcome reward、每 prompt 多 response 比较的任务 |


## 12. old policy logprob 相关开关

PPO-style policy loss 的核心 ratio 是：

```text
ratio = exp(logp_new - logp_old)
```

其中 `logp_new` 是当前正在训练的 actor 对已采样 token 的 logprob；真正容易混淆的是 `logp_old`。它应该代表“生成这批 rollout 数据的行为策略”的 logprob。`--keep-old-actor` 和 `--use-rollout-logprobs` 都是在解决 `logp_old` 怎么来的问题，所以它们不是 PPO 专用，GRPO 也会用。原因是 PPO、GRPO、GSPO、CISPO 都复用 `policy_loss_function` 这条 actor policy loss 路径。

### 12.1 默认路径：训练前重算 old logprob

在普通同步在线训练里，常见流程是：

```text
actor 权重同步到 rollout engine
  -> rollout engine 用这版 actor 生成数据
  -> actor 还没更新
  -> 训练前用当前 actor 对这批数据 forward 一次，得到 old_log_probs
  -> 再进入 actor update
```

这时当前 actor 就是 rollout actor，训练前 forward 一次得到的 `log_probs` 可以作为 `old_log_probs`。因此常规同步训练不一定需要 `keep_old_actor`，也不一定需要 `use_rollout_logprobs`。

### 12.2 `--keep-old-actor`

`keep_old_actor=True` 会在 trainer 内维护一份旧 actor 权重快照，训练前切到 `old_actor` 重新 forward 当前 batch，得到 `rollout_data["log_probs"]` 作为 ratio 分母。

它解决的是“当前 actor 不等于生成这批数据的 actor”的问题。例如：

```text
actor_v10 生成 rollout 数据
actor 已经训练到 v11 / v12
现在从 buffer / async 队列里拿到 actor_v10 的旧数据训练
```

如果这时直接用当前 actor forward，得到的是 `actor_v12` 的 logprob，不是生成数据时的 `actor_v10` logprob：

```text
错误近似：ratio = pi_v12_new / pi_v12_detached
正确目标：ratio = pi_v12_new / pi_v10_old
```

在没有 rollout logprob 的情况下，只能保留某个 old actor 快照来近似或匹配行为策略。

但这个功能使用场景比较窄。它要求保存的 `old_actor` 和这批 stale 数据的真实行为策略版本对得上。如果 buffer 里混了多个 actor 版本的数据，只维护一份 `old_actor` 也不严格够用，除非 buffer / update interval 的设计能保证当前取出的数据都来自同一个旧版本。

仓库里主线 `scripts/run-*.sh` 基本没有开 `--keep-old-actor`；它出现在 rollout buffer plugin 示例 `slime_plugins/rollout_buffer/rollout_buffer_example.sh`，属于 async / buffer 场景下的工程选项。

### 12.3 `--use-rollout-logprobs`

`use_rollout_logprobs=True` 表示直接使用 rollout engine 生成 token 时记录的 `sample.rollout_log_probs` 作为 `old_log_probs`，训练侧不再重算 old logprob：

```text
old_log_probs = batch["rollout_log_probs"]
ratio = exp(logp_new - rollout_log_probs)
```

它的优点是 `logp_old` 精确对应真实采样那一刻的行为策略，尤其适合：

- 异步 rollout。
- rollout buffer / stale data。
- multi-agent / agentic 多轮轨迹。
- 外部 rollout engine。
- 训练侧 Megatron 和推理侧 SGLang 可能存在 logprob mismatch 的场景。

代价是 rollout 必须返回每个 response token 的 logprob，并且 logprob 口径要和训练侧 loss 对齐。例如 rollout 使用 top-p 时，训练侧需要配合 top-p replay 让 normalization 范围一致。

官方示例里 `examples/multi_agent/run-qwen3-30B-A3B-multi-agent.sh` 在 GRPO 配置中开启了 `--use-rollout-logprobs`。`examples/train_infer_mismatch_helper/README.md` 也明确说明，这个 flag 会跳过训练引擎重算 old logprob，直接使用 rollout logprob 进入 PPO/GRPO loss。

### 12.4 和 TIS 的关系

`use_rollout_logprobs` 和 `use_tis` 在参数校验里互斥。原因是两者对 rollout logprob 的使用方式不同：

- `use_rollout_logprobs=True`: 直接把 rollout policy 当 PPO ratio 的 old policy 分母。
- `use_tis=True`: 保留训练侧 old policy 作为 PPO clipping anchor，同时用 `train_old_logp / rollout_logp` 做额外 off-policy importance sampling 修正。

也就是说：

```text
use_rollout_logprobs:
  2-policy bypass: pi_new / pi_rollout

use_tis:
  3-policy correction: (pi_old / pi_rollout) * PPO(pi_new / pi_old)
```

### 12.5 实用判断

可以按下面的优先级理解：

```text
同步在线训练：
  默认重算 old_log_probs 通常够用。

复杂 rollout / async / buffer / agent / 外部 engine：
  优先记录 rollout_log_probs，并开启 --use-rollout-logprobs。

无法可靠记录 rollout_log_probs，但又要训练 stale 数据：
  才考虑 --keep-old-actor。
```

因此，`keep_old_actor` 更像一个 fallback / 特殊场景功能；在通常会记录 rollout logprob 的现代 rollout 管线里，它确实相对少用。

## 13. 关键结论

1. slime 的 PPO 是标准 actor-critic PPO 结构，但工程上和 Megatron / SGLang / Ray 深度结合。
2. actor 和 critic 复用 `MegatronTrainRayActor` 及 Megatron 训练骨架，通过 `role` 分出不同 output head、派生数据和 loss；两套模型参数与 optimizer 仍完全独立。
3. PPO 的核心路径是：rollout reward -> critic value -> KL-shaped token reward -> GAE -> clipped policy loss + clipped value loss。
4. critic 到 actor 当前只通过 `external_data` 传 old values；基础 rollout 数据走 `rollout_data_ref`，advantages/returns 由两边在本地分别计算。
5. RolloutManager 的训练负载均衡遵循“先组成 micro-batch，再分配给 DP rank”：用 token budget 控制显存，用相同 micro-batch 数保证 PP/VPP 调度，再可选按 estimated FLOPs 平衡 DP workloads。
6. policy loss 对 PPO 和 GRPO 是共用的，都是 PPO-style clipped objective；算法差异主要来自 advantage estimator。
7. GRPO 在 slime 里本质是“组内 reward baseline + critic-free PPO-style policy update”，省掉了 critic 和 value loss。
8. slime 的很多复杂度来自大模型训练现实问题：packed sequence、CP/TP/PP、offload、rollout/train policy mismatch、per-rollout reducer、权重同步。
9. 看 loss 数值时必须确认是否开启 `--calculate-per-token-loss`。默认 per-rollout/per-sample mean 和 per-token mean 的量纲不同。
10. `--keep-old-actor` 和 `--use-rollout-logprobs` 都是 old policy logprob 的来源选择，不是 PPO 专用；GRPO 也会使用它们来构造 policy ratio。

## 14. PPO 配置关注点

建议重点检查这些参数：

- `--advantage-estimator ppo`
- `--num-critic-only-steps`
- `--gamma`、`--lambd`
- `--normalize-advantages`
- `--eps-clip`、`--eps-clip-high`
- `--value-clip`
- `--kl-coef` 或 `--use-kl-loss --kl-loss-coef`
- `--kl-loss-type`
- `--entropy-coef`
- `--calculate-per-token-loss`
- `--use-rollout-logprobs`、`--use-tis`、`--use-opsm`
- actor / critic 的 `lr`、`load`、`save`，可通过 `--megatron-config-path` 分角色覆盖。

一个简化 PPO 心智模型：

```text
rollout samples
  -> reward / mask / token packing
  -> critic forward: V_old
  -> KL-shaped token reward
  -> GAE: advantages, returns
  -> critic update: clipped value loss
  -> actor forward: logp_new
  -> ratio = exp(logp_new - logp_old)
  -> actor update: clipped policy loss + optional entropy/KL/TIS/OPSM
  -> sync actor weights to rollout engines
```

一个简化 GRPO 心智模型：

```text
rollout K samples per prompt
  -> group normalize rewards
  -> broadcast normalized reward as token advantage
  -> actor forward: logp_new
  -> ratio = exp(logp_new - logp_old)
  -> clipped policy loss + optional entropy/KL/TIS/OPSM
  -> sync actor weights to rollout engines
```
