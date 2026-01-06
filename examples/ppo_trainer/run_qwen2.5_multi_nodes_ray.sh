#!/bin/bash
set -xe

##### 配置部分 #####
MASTER_NODE_IP="10.144.202.44"     # Master 节点 IP
WORKER_NODE_IP="10.144.201.24"      # Worker 节点 IP
WORKER_USER="root"                  # Worker 节点用户名（如果不能免密，这里只能手动在两个节点运行）
# WORKER_SCRIPT_PATH="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/examples/ppo_trainer/run_qwen2.5_multi_nodes_ray.sh"

MASTER_PORT=29504                   # torchrun 通信端口
RAY_PORT=6383                       # Ray 集群端口
WORLD_SIZE=2
GPUS_PER_NODE=8                     # 每节点 GPU 数量
NPROC_PER_NODE=1

##### 检测当前节点 IP #####
current_ip=$(hostname -I | grep -o "10\.144\.[0-9]\+\.[0-9]\+" | head -1)

##### 启动 Ray #####
if [[ "$current_ip" == "$MASTER_NODE_IP" ]]; then
    export RANK=0
    echo "[MASTER] 启动 Ray Head 节点"
    ray stop || true
    ray start --head --node-ip-address="$MASTER_NODE_IP" --port=$RAY_PORT --num-gpus=$GPUS_PER_NODE

    echo "[MASTER] 启动 Worker Ray 节点 via SSH..."
    ssh $WORKER_USER@$WORKER_NODE_IP "ray stop || true && ray start --address=$MASTER_NODE_IP:$RAY_PORT --num-gpus=$GPUS_PER_NODE" &

    sleep 5
    echo "[MASTER] Ray 集群状态:"
    ray status

elif [[ "$current_ip" == "$WORKER_NODE_IP" ]]; then
    export RANK=1
    echo "[WORKER] 启动 Ray Worker 节点"
    ray stop || true
    ray start --address=$MASTER_NODE_IP:$RAY_PORT --num-gpus=$GPUS_PER_NODE

    ray status
else
    echo "Unknown node IP: $current_ip"
    echo "请在 $MASTER_NODE_IP 或 $WORKER_NODE_IP 上运行该脚本"
    exit 1
fi

##### 数据路径 #####
gsm8k_train_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/train.parquet"
gsm8k_test_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/test.parquet"
math_train_path=/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/math/train.parquet
math_test_path=/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/math/test.parquet

# train_files="['$gsm8k_train_path']"
# test_files="['$gsm8k_test_path']"
train_files="['$math_train_path']"
test_files="['$math_test_path']"


LOG_DIR="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/results"
mkdir -p "$LOG_DIR"  # 确保目录存在
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] 日志将保存到 $LOG_FILE"


##### 启动训练 #####
echo "[INFO] 启动 torchrun 分布式训练..."

nohup bash -c "
torchrun \
    --nnodes=$WORLD_SIZE \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$RANK \
    --master_addr=$MASTER_NODE_IP \
    --master_port=$MASTER_PORT \
    -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files=\"$train_files\" \
    data.val_files=\"$test_files\" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-14B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-14B-Instruct \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=4 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='[\"wandb\"]' \
    trainer.project_name='test_experiments' \
    trainer.experiment_name='Qwen2.5-14B-Instruct_math' \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$WORLD_SIZE \
    trainer.save_freq=100000 \
    trainer.test_freq=10 \
    trainer.total_epochs=15
" > $LOG_FILE 2>&1 &