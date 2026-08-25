# 模块 5：Debug、指标、Trace、Profiling 与测试

## 目标

分析发生 reward 异常、loss 异常、权重同步异常、rollout 卡死时，应该从哪些指标、trace 和测试入手。

## 重点文件

- `slime/ray/rollout.py`
- `slime/utils/metric_utils.py`
- `slime/utils/train_metric_utils.py`
- `slime/utils/trace_utils.py`
- `slime/utils/profile_utils.py`
- `slime/utils/logging_utils.py`
- `slime/utils/health_monitor.py`
- `slime/utils/train_dump_utils.py`
- `tests/`
- `tests/utils/`
- `tests/plugin_contracts/`
- `docs/zh/developer_guide/debug.md`
- `docs/zh/developer_guide/trace.md`
- `docs/zh/developer_guide/profiling.md`
- `docs/zh/developer_guide/ci.md`

## 待分析问题

- rollout 质量指标、SGLang 性能指标、训练 loss 指标分别在哪里产生？
- zero std、pass rate、reward 分布、response length 这些指标如何定位问题？
- debug rollout-only、train-only、rollout-then-train replay 的适用场景是什么？
- trace/profiling 怎么定位 rollout 慢、训练慢、同步慢？
- 哪些测试保护权重同步，哪些测试保护 loss，哪些测试保护 rollout contract？
