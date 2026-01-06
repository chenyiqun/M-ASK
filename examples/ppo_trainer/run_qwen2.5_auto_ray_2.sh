#!/bin/bash
set -x

##################################
# 训练脚本调试信息
##################################
echo "===== DEBUG TRAIN SCRIPT ====="
echo "[DEBUG] Hostname: $(hostname)"
echo "[DEBUG] Date: $(date)"
echo "[DEBUG] JOB_ID=${JOB_ID}"
echo "[DEBUG] TARGET_WORLD_SIZE=${TARGET_WORLD_SIZE}"
echo "[DEBUG] GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "[DEBUG] MASTER_PORT=${MASTER_PORT}"
echo "[DEBUG] RAY_PORT=${RAY_PORT}"
echo "=============================="

##### 数据路径 #####
gsm8k_train_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/train.parquet"
gsm8k_test_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/test.parquet"
math_train_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/math/train.parquet"
math_test_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/math/test.parquet"

train_files="['$math_train_path']"
test_files="['$math_test_path']"

LOG_FILE="$RESULT_DIR/train_$(date +%Y%m%d_%H%M%S)_rank${RANK}.log"
echo "[INFO] 日志将保存到 $LOG_FILE"

##### 启动训练 #####
python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-14B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-14B-Instruct \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["wandb"]' \
    trainer.project_name='test_experiments' \
    trainer.experiment_name='Qwen2.5-14B-Instruct_math' \
    trainer.n_gpus_per_node=${GPUS_PER_NODE} \
    trainer.nnodes=${TARGET_WORLD_SIZE} \
    trainer.save_freq=100000 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    >> "$LOG_FILE" 2>&1

echo "开始 sleep 1h"
sleep 3600
echo "结束"
