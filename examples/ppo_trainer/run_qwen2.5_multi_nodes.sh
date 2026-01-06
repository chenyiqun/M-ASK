# #!/bin/bash
# set -x

# ########################
# # 节点 IP 配置（方便修改）
# ########################
# MASTER_NODE_IP="10.144.203.245"   # 主节点 IP
# WORKER_NODE_IP="10.144.204.18"   # 第二节点 IP

# ########################
# # 分布式配置
# ########################
# export MASTER_ADDR="$MASTER_NODE_IP"
# export MASTER_PORT=29500
# export WORLD_SIZE=2
# export NPROC_PER_NODE=1

# # 自动检测 IP（通用正则）
# current_ip=$(hostname -I | grep -o "10\.144\.[0-9]\+\.[0-9]\+" | head -1)

# if [[ "$current_ip" == "$MASTER_NODE_IP" ]]; then
#     export RANK=0
#     echo "Running on MASTER node ($MASTER_NODE_IP) as RANK=0"
# elif [[ "$current_ip" == "$WORKER_NODE_IP" ]]; then
#     export RANK=1
#     echo "Running on WORKER node ($WORKER_NODE_IP) as RANK=1"
# else
#     echo "Unknown node IP: $current_ip"
#     echo "Please run this script on either $MASTER_NODE_IP or $WORKER_NODE_IP"
#     exit 1
# fi

# ########################
# # 数据路径配置
# ########################
# gsm8k_train_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/train.parquet"
# gsm8k_test_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/test.parquet"

# train_files="['$gsm8k_train_path']"
# test_files="['$gsm8k_test_path']"

# ########################
# # 运行参数日志
# ########################
# echo "Starting torchrun with:"
# echo "  MASTER_ADDR=$MASTER_ADDR"
# echo "  MASTER_PORT=$MASTER_PORT"
# echo "  WORLD_SIZE=$WORLD_SIZE"
# echo "  RANK=$RANK"
# echo "  NPROC_PER_NODE=$NPROC_PER_NODE"

# ########################
# # 启动分布式训练
# ########################
# torchrun \
#     --nnodes=$WORLD_SIZE \
#     --nproc_per_node=$NPROC_PER_NODE \
#     --node_rank=$RANK \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=gae \
#     data.train_files="$train_files" \
#     data.val_files="$test_files" \
#     data.train_batch_size=1024 \
#     data.max_prompt_length=1024 \
#     data.max_response_length=1024 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     actor_rollout_ref.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-3B-Instruct \
#     actor_rollout_ref.model.enable_gradient_checkpointing=False \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
#     critic.optim.lr=1e-5 \
#     critic.model.use_remove_padding=True \
#     critic.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-3B-Instruct \
#     critic.model.enable_gradient_checkpointing=False \
#     critic.ppo_micro_batch_size_per_gpu=8 \
#     critic.model.fsdp_config.param_offload=False \
#     critic.model.fsdp_config.optimizer_offload=False \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger='["wandb"]' \
#     trainer.project_name='test_experiments' \
#     trainer.experiment_name='Qwen2.5-3B-Instruct_test_2nodes_torchrun' \
#     trainer.n_gpus_per_node=8 \
#     trainer.nnodes=2 \
#     trainer.save_freq=100000 \
#     trainer.test_freq=10 \
#     trainer.total_epochs=15

# Master 节点
MASTER_ADDR=10.144.203.245
MASTER_PORT=29500
RANK=0
WORLD_SIZE=2
python -m torch.distributed.run --nproc_per_node=1 --nnodes=2 --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT test_dist.py
