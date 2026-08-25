
#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
pkill -9 python
sleep 3
pkill -9 ray
pkill -9 python

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16
export SGLANG_ENABLE_SPEC_V2=1
export CONDA_PREFIX=/mnt/shared-storage-user/huanghaian/miniconda3/envs/slime_pt211
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX}/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH}
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}

# Save local script output (stdout+stderr) to a single log.
LOG_DIR="${LOG_DIR:-./logs}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/qwen3.5-35B-rl-eagle_${TS}.log"
mkdir -p "${LOG_DIR}"
exec > >(stdbuf -oL -eL tee -a "${LOG_FILE}") 2>&1
echo "[log] writing local output to: ${LOG_FILE}"

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3.5-35B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint /mnt/shared-storage-user/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307
   #--hf-checkpoint /root/Qwen3-4B-FP8
   --ref-load /mnt/shared-storage-user/llmrazor-share/model/Qwen3.5-35B-A3B_torch_dist
   --load ./Qwen3.5-35B-A3B-mtp_slime/
   --save ./Qwen3.5-35B-A3B-mtp_slime/
   --save-interval 2000
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/shared-storage-user/llmrazor-share/data/slime_data/dapo-math-17k/dapo-math-17k.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type deepscaler
   --num-rollout 3000
   --rollout-batch-size 32
   --n-samples-per-prompt 8
   --rollout-max-response-len 8192
   --rollout-temperature 1

   --global-batch-size 256
   --balance-data

#    --debug-rollout-only
#    --save-debug-rollout-data ./debug/data_{rollout_id}.pt
   --debug-train-only
   --load-debug-rollout-data ./debug/data_{rollout_id}.pt
)

EVAL_ARGS=()

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
#    --use-kl-loss
   --kl-loss-coef 0.00
#    --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
   --use-rollout-routing-replay
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --use-distributed-optimizer
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group mimo-7B-rl-test
   # --wandb-key ${WANDB_API_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.7

   # for speculative decoding
#    --sglang-speculative-algorithm EAGLE
#    --sglang-speculative-num-steps 3
#    --sglang-speculative-eagle-topk 1
#    --sglang-speculative-num-draft-tokens 4
#    --sglang-mamba-scheduler-strategy extra_buffer

   # sometimes flashinfer has IMA bugs. Use fa3 as instead
   --sglang-attention-backend fa3
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

SPEC_ARGS=(
#    --enable-mtp-training
#    --mtp-loss-scaling-factor 0.2
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

# \"PYTHONPATH\": \"/mnt/shared-storage-user/huanghaian/code/slime_package/Megatron-LM/\",
# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_NET\": \"Socket\",
    \"NCCL_NET_PLUGIN\": \"none\",
    \"NCCL_IB_DISABLE\": \"1\",
    \"NCCL_COLLNET_ENABLE\": \"0\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --submission-id "qwen3.5-35B-rl-eagle_${TS}" \
   --no-wait \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${SPEC_ARGS[@]}

# Ray job logs are produced on the Ray side; follow them into a separate log file.
RAY_JOB_ID="qwen3.5-35B-rl-eagle_${TS}"
RAY_LOG_FILE="${LOG_DIR}/${RAY_JOB_ID}.ray.log"
echo "[log] following ray job logs (${RAY_JOB_ID}) into: ${RAY_LOG_FILE}"
ray job logs -f "${RAY_JOB_ID}" 2>&1 | stdbuf -oL -eL tee -a "${RAY_LOG_FILE}"
