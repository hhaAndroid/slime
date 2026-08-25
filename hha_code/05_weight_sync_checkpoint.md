# 模块 4：权重同步、Checkpoint 与 Offload

## 目标

分析 Megatron 训练出来的权重如何正确、高效地同步给 SGLang，以及 checkpoint/offload 如何配合。

## 重点文件

- `slime/backends/megatron_utils/actor.py`
- `slime/backends/megatron_utils/update_weight/`
- `slime/backends/sglang_utils/sglang_engine.py`
- `slime/backends/megatron_utils/checkpoint.py`
- `slime/backends/megatron_utils/hf_checkpoint_saver.py`
- `slime/backends/megatron_utils/megatron_to_hf/`
- `slime/utils/tensor_backper.py`
- `slime/utils/reloadable_process_group.py`
- `tools/convert_hf_to_torch_dist.py`
- `tools/convert_torch_dist_to_hf.py`
- `docs/zh/advanced/delta-weight-sync.md`
- `docs/zh/advanced/fault-tolerance.md`

## 待分析问题

- colocate 为什么走 tensor update，非 colocate 为什么走 distributed/disk？
- full update、delta update、NCCL transport、disk transport 各自的边界是什么？
- Megatron sharded 参数如何转换成 SGLang/HF 需要的权重？
- `megatron_to_hf_mode=raw/bridge` 对权重同步路径有什么影响？
- checkpoint 保存、恢复、resume rollout id 的逻辑在哪里？
- offload train/rollout 时，显存、CPU backup、process group 如何处理？
