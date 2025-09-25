#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

CKPT_ARGS=(
   --hf-checkpoint /mnt/shared-storage-user/llmrazor-share/model/Qwen2.5-Math-7B
   --ref-load /mnt/shared-storage-user/llmrazor-share/model/Qwen2.5-Math-7B
)

ROLLOUT_ARGS=(
   --prompt-data /mnt/shared-storage-user/huanghaian/code/slime/data/dapo-math-17k_process.jsonl
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type dapo
   --reward-key score
   --eval-reward-key acc
   --num-rollout 200
   --rollout-batch-size 512
   --n-samples-per-prompt 16
   --rollout-max-response-len 8192
   --rollout-temperature 1.0
   --balance-data

   --global-batch-size 8192
   --max-tokens-per-gpu 32768
)

EVAL_ARGS=(
   --eval-prompt-data aime /mnt/shared-storage-user/huanghaian/code/slime/data/aime-2024_process.jsonl
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 8192
   --eval-top-p 0.7
   --eval-interval 5
)

GRPO_ARGS=(
   #--loss-type sft_loss
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.8
   --sglang-server-concurrency 128
)

MASTER_ADDR=10.103.20.71
# launch the master node of ray in container
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
        "SLIME_BACKEND": "xtuner"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 4 \
   --rollout-num-gpus 4 \
   --train-optimizer-steps 16 \
   --pack-max-length 32768 \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]}