# #!/bin/bash
# set -xe

# ##### 配置 #####
# TARGET_WORLD_SIZE=2                       # 目标节点数
# GPUS_PER_NODE=8                           # 每节点 GPU 数量
# MASTER_PORT=29503                         # torchrun master 通信端口
# RAY_PORT=6382                             # Ray 集群端口
# WORKER_USER="root"                  # Worker 节点用户名（如果不能免密，这里只能手动在两个节点运行）

# SHARED_DIR="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/results/${MASTER_PORT}_${RAY_PORT}_${TARGET_WORLD_SIZE}_${GPUS_PER_NODE}"
# IP_LIST_FILE="$SHARED_DIR/node_ips.txt"

# mkdir -p "$SHARED_DIR"

# # 获取当前节点 IP
# CURRENT_IP=$(hostname -I | awk '{print $1}')

# # 写入当前IP到共享文件（去重排序）
# grep -q "$CURRENT_IP" "$IP_LIST_FILE" 2>/dev/null || { echo "$CURRENT_IP" >> "$IP_LIST_FILE"; sort -u "$IP_LIST_FILE" -o "$IP_LIST_FILE"; }
# echo "[INFO] 当前节点 $CURRENT_IP 已写入 $IP_LIST_FILE"

# ##### 打印当前节点 GPU 可见情况 #####
# echo "[DEBUG] 当前节点 $CURRENT_IP CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# nvidia-smi || echo "[WARN] nvidia-smi 执行失败，可能该节点无 GPU 或驱动不可用"

# ##### 循环等待节点到齐 #####
# while true; do
#     CURRENT_NODE_COUNT=$(wc -l < "$IP_LIST_FILE")
#     echo "[INFO] 当前已注册节点数: $CURRENT_NODE_COUNT / $TARGET_WORLD_SIZE"

#     if [[ $CURRENT_NODE_COUNT -ge $TARGET_WORLD_SIZE ]]; then
#         echo "[READY] 节点已全部到齐，继续执行..."
#         break
#     else
#         echo "[WAIT] 节点尚未全部到齐，等待其他节点运行该脚本..."
#         sleep 10
#     fi
# done

# ##### 获取 Master 节点和当前节点 Rank #####
# MASTER_NODE_IP=$(head -n1 "$IP_LIST_FILE")
# NODE_RANK=$(grep -n "$CURRENT_IP" "$IP_LIST_FILE" | cut -d: -f1)
# NODE_RANK=$((NODE_RANK-1))  # 行号改成从0开始的rank
# echo "[INFO] MASTER=${MASTER_NODE_IP}, 当前节点 RANK=${NODE_RANK}"

# ##### 启动 Ray #####
# if [[ "$CURRENT_IP" == "$MASTER_NODE_IP" ]]; then
#     echo "[MASTER] 启动 Ray Head 节点..."
#     ray stop || true
#     ray start --head \
#         --node-ip-address="$MASTER_NODE_IP" \
#         --port=$RAY_PORT \
#         --num-gpus=$GPUS_PER_NODE

#     echo "[MASTER] 启动 Worker 节点 via SSH..."
#     while read ip; do
#         [[ "$ip" == "$MASTER_NODE_IP" ]] && continue
#         echo " -> 启 Worker 节点: $ip"
#         ssh $WORKER_USER@$ip "ray stop || true && ray start \
#             --address=$MASTER_NODE_IP:$RAY_PORT \
#             --node-ip-address=$ip \
#             --num-gpus=$GPUS_PER_NODE" &
#     done < "$IP_LIST_FILE"
#     wait
#     sleep 3
#     ray status

# else
#     echo "[WORKER] 启动 Ray Worker 节点..."
#     ray stop || true
#     ray start \
#         --address=$MASTER_NODE_IP:$RAY_PORT \
#         --node-ip-address=$CURRENT_IP \
#         --num-gpus=$GPUS_PER_NODE
#     sleep 3
#     ray status
# fi

# # ##### 启动 Ray #####
# # if [[ "$CURRENT_IP" == "$MASTER_NODE_IP" ]]; then
# #     echo "[MASTER] 启动 Ray Head 节点..."
# #     ray stop || true
# #     ray start --head \
# #         --node-ip-address="$MASTER_NODE_IP" \
# #         --port=$RAY_PORT \
# #         --num-gpus=$GPUS_PER_NODE

# #     echo "[MASTER] 等待所有 Worker 节点加入 Ray 集群..."
# #     until [[ $(ray status | grep "Active:" | grep -v "(no)" | wc -l) -ge $TARGET_WORLD_SIZE ]] \
# #           && [[ $(ray status | grep "GPU" | awk -F'/' '{sum += $2} END {print sum}') -ge $((TARGET_WORLD_SIZE * GPUS_PER_NODE)) ]]; do
# #         echo "[WAIT] 集群节点或GPU还未全部就绪..."
# #         sleep 5
# #     done

# #     touch "$SHARED_DIR/master_ready"
# # else
# #     echo "[WORKER] 等待 Master 完成启动并GPU就绪..."
# #     while [[ ! -f "$SHARED_DIR/master_ready" ]]; do
# #         sleep 5
# #     done

# #     ray stop || true
# #     ray start \
# #         --address=$MASTER_NODE_IP:$RAY_PORT \
# #         --node-ip-address=$CURRENT_IP \
# #         --num-gpus=$GPUS_PER_NODE

# #     until ray status | grep "$CURRENT_IP" | grep "GPU"; do
# #         echo "[WAIT] 本节点GPU还未被集群识别..."
# #         sleep 5
# #     done
# # fi

# sleep 5
# echo "[DEBUG] Ray 集群状态："
# ray status || true

# # ##### 检查 GPU 总数 #####
# # TOTAL_GPUS=$(ray status | grep "GPU" | awk -F'/' '{sum += $2} END {print sum}')
# # if [[ $TOTAL_GPUS -lt $((TARGET_WORLD_SIZE * GPUS_PER_NODE)) ]]; then
# #     echo "[ERROR] GPU 总数不足 ($TOTAL_GPUS / $((TARGET_WORLD_SIZE * GPUS_PER_NODE)))，退出..."
# #     exit 1
# # fi

# LOG_FILE="$SHARED_DIR/train_$(date +%Y%m%d_%H%M%S)_rank${NODE_RANK}.log"
# echo "[INFO] 节点 $CURRENT_IP (rank=${NODE_RANK}) 启动 torchrun..."

# ##### 打印 CUDA/GPU 状态，方便调试 #####
# echo "[DEBUG] 启动训练前节点 $CURRENT_IP GPU 状态："
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# nvidia-smi || echo "[WARN] nvidia-smi 执行失败"

# ##### 数据路径 #####
# gsm8k_train_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/train.parquet"
# gsm8k_test_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/test.parquet"
# train_files="['$gsm8k_train_path']"
# test_files="['$gsm8k_test_path']"

# torchrun \
#     --nnodes=$TARGET_WORLD_SIZE \
#     --nproc_per_node=1 \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_NODE_IP \
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
#     actor_rollout_ref.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
#     critic.optim.lr=1e-5 \
#     critic.model.use_remove_padding=True \
#     critic.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct \
#     critic.model.enable_gradient_checkpointing=False \
#     critic.ppo_micro_batch_size_per_gpu=8 \
#     critic.model.fsdp_config.param_offload=False \
#     critic.model.fsdp_config.optimizer_offload=False \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger='["wandb"]' \
#     trainer.project_name='test_experiments' \
#     trainer.experiment_name='Qwen2.5-7B-Instruct_test_ray_multi_nodes' \
#     trainer.n_gpus_per_node=$GPUS_PER_NODE \
#     trainer.nnodes=$TARGET_WORLD_SIZE \
#     trainer.save_freq=100000 \
#     trainer.test_freq=10 \
#     trainer.total_epochs=15 \
#     >> "${LOG_FILE}" 2>&1





# #!/bin/bash
# set -xe

# ##### 配置 #####
# TARGET_WORLD_SIZE=2                       # 目标节点数
# GPUS_PER_NODE=8                           # 每节点 GPU 数量
# MASTER_PORT=29503                         # torchrun master 通信端口
# RAY_PORT=6382                             # Ray 集群端口

# SHARED_DIR="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/results/${MASTER_PORT}_${RAY_PORT}_${TARGET_WORLD_SIZE}_${GPUS_PER_NODE}"
# IP_LIST_FILE="$SHARED_DIR/node_ips.txt"

# mkdir -p "$SHARED_DIR"

# ##### 获取当前节点 IP（集群内网段优先） #####
# CURRENT_IP=$(hostname -I | tr ' ' '\n' | grep '^10\.' | head -n1)
# echo "[INFO] 当前节点 IP: $CURRENT_IP"

# ##### 写入当前IP到共享文件（去重排序）#####
# grep -q "$CURRENT_IP" "$IP_LIST_FILE" 2>/dev/null || {
#     echo "$CURRENT_IP" >> "$IP_LIST_FILE"
#     sort -u "$IP_LIST_FILE" -o "$IP_LIST_FILE"
# }
# echo "[INFO] 已写入 $CURRENT_IP 到 $IP_LIST_FILE"

# ##### 打印当前节点 GPU 可见情况 #####
# echo "[DEBUG] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# nvidia-smi || echo "[WARN] nvidia-smi 执行失败，可能该节点无 GPU 或驱动不可用"

# ##### 循环等待节点到齐 #####
# while true; do
#     CURRENT_NODE_COUNT=$(wc -l < "$IP_LIST_FILE")
#     echo "[INFO] 已注册节点数: $CURRENT_NODE_COUNT / $TARGET_WORLD_SIZE"
#     if [[ $CURRENT_NODE_COUNT -ge $TARGET_WORLD_SIZE ]]; then
#         echo "[READY] 节点已全部到齐"
#         break
#     else
#         echo "[WAIT] 节点尚未全部到齐..."
#         sleep 5
#     fi
# done

# ##### 从MPI环境变量获取RANK #####
# if [[ -n "$OMPI_COMM_WORLD_RANK" ]]; then
#     NODE_RANK=$OMPI_COMM_WORLD_RANK
# elif [[ -n "$PMI_RANK" ]]; then
#     NODE_RANK=$PMI_RANK
# else
#     # 回退用 IP 列表匹配位置
#     NODE_RANK=$(grep -n "$CURRENT_IP" "$IP_LIST_FILE" | cut -d: -f1)
#     NODE_RANK=$((NODE_RANK-1))
# fi

# MASTER_NODE_IP=$(head -n1 "$IP_LIST_FILE")
# echo "[INFO] MASTER=$MASTER_NODE_IP, NODE_RANK=$NODE_RANK"

# ##### 如果是 root 用户，启用 OMPI 允许 root 运行 #####
# if [[ "$(id -u)" -eq 0 ]]; then
#     echo "[WARN] 当前用户是 root，将启用 MPI 允许 root 运行选项"
#     export OMPI_ALLOW_RUN_AS_ROOT=1
#     export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# fi

# ##### 启动前清理残留 Ray 状态 #####
# ray stop || true
# rm -rf /tmp/ray
# # ##### 启动前强制清理残留 Ray/Redis 状态（新增更强清理） #####
# # pkill -9 -f ray || true
# # pkill -9 -f redis || true
# # ray stop || true
# # rm -rf /tmp/ray

# # 安装netcat
# apt-get update && apt-get install -y netcat-openbsd

# ##### 启动 Ray #####
# if [[ "$NODE_RANK" -eq 0 ]]; then
#     echo "[MASTER] 启动 Ray Head 节点..."
#     ray start --head \
#         --node-ip-address="$MASTER_NODE_IP" \
#         --port=$RAY_PORT \
#         --num-gpus=$GPUS_PER_NODE
#     echo "[MASTER] Head 启动完成，等待 Worker 连接..."
# else
#     echo "[WORKER] 等待 Head 节点 GCS 服务可用..."
#     # 等待端口可连
#     until nc -z $MASTER_NODE_IP $RAY_PORT; do
#         echo "[WAIT] GCS 未就绪，重试中..."
#         sleep 2
#     done
#     sleep 5  # 新增额外等待，确保 Redis 完全初始化
#     echo "[WORKER] 启动 Ray Worker 节点 (rank=$NODE_RANK)..."
#     ray start \
#         --address=$MASTER_NODE_IP:$RAY_PORT \
#         --node-ip-address=$CURRENT_IP \
#         --num-gpus=$GPUS_PER_NODE
# fi

# sleep 5
# echo "[DEBUG] 当前 Ray 集群状态:"
# ray status || true

# LOG_FILE="$SHARED_DIR/train_$(date +%Y%m%d_%H%M%S)_rank${NODE_RANK}.log"
# echo "[INFO] 节点 $CURRENT_IP (rank=${NODE_RANK}) 启动 torchrun..."

# ##### 打印 CUDA/GPU 状态，方便调试 #####
# echo "[DEBUG] 启动训练前 GPU 状态："
# echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# nvidia-smi || echo "[WARN] nvidia-smi 执行失败"

# ##### 数据路径 #####
# gsm8k_train_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/train.parquet"
# gsm8k_test_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/test.parquet"
# train_files="['$gsm8k_train_path']"
# test_files="['$gsm8k_test_path']"

# torchrun \
#     --nnodes=$TARGET_WORLD_SIZE \
#     --nproc_per_node=1 \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_NODE_IP \
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
#     actor_rollout_ref.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
#     critic.optim.lr=1e-5 \
#     critic.model.use_remove_padding=True \
#     critic.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct \
#     critic.model.enable_gradient_checkpointing=False \
#     critic.ppo_micro_batch_size_per_gpu=8 \
#     critic.model.fsdp_config.param_offload=False \
#     critic.model.fsdp_config.optimizer_offload=False \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger='["wandb"]' \
#     trainer.project_name='test_experiments' \
#     trainer.experiment_name='Qwen2.5-7B-Instruct_test_ray_multi_nodes' \
#     trainer.n_gpus_per_node=$GPUS_PER_NODE \
#     trainer.nnodes=$TARGET_WORLD_SIZE \
#     trainer.save_freq=100000 \
#     trainer.test_freq=10 \
#     trainer.total_epochs=15 \
#     >> "${LOG_FILE}" 2>&1




# #!/bin/bash
# set -xe

# ##### 配置 #####
# TARGET_WORLD_SIZE=2         # 节点总数
# GPUS_PER_NODE=8             # 每节点 GPU 数量
# MASTER_PORT=29503           # torchrun master 通信端口
# RAY_PORT=6382               # Ray 集群端口

# SHARED_DIR="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/results/${MASTER_PORT}_${RAY_PORT}_${TARGET_WORLD_SIZE}_${GPUS_PER_NODE}"
# IP_LIST_FILE="$SHARED_DIR/node_ips.txt"
# mkdir -p "$SHARED_DIR"

# ##### 获取当前节点 IP #####
# CURRENT_IP=$(hostname -I | tr ' ' '\n' | grep '^10\.' | head -n1)
# echo "[INFO] 当前节点 IP: $CURRENT_IP"

# ##### 从 MPI/Slurm 环境获取 RANK #####
# if [[ -n "$OMPI_COMM_WORLD_RANK" ]]; then
#     NODE_RANK=$OMPI_COMM_WORLD_RANK
# elif [[ -n "$PMI_RANK" ]]; then
#     NODE_RANK=$PMI_RANK
# elif [[ -n "$SLURM_PROCID" ]]; then
#     NODE_RANK=$SLURM_PROCID
# else
#     echo "[WARN] 无法从环境获取 rank，使用 IP 顺序推算"
#     grep -q "$CURRENT_IP" "$IP_LIST_FILE" 2>/dev/null || echo "$CURRENT_IP" >> "$IP_LIST_FILE"
#     sort -u "$IP_LIST_FILE" -o "$IP_LIST_FILE"
#     NODE_RANK=$(grep -n "$CURRENT_IP" "$IP_LIST_FILE" | cut -d: -f1)
#     NODE_RANK=$((NODE_RANK-1))
# fi
# echo "[INFO] NODE_RANK=$NODE_RANK"

# MASTER_NODE_IP=""
# if [[ "$NODE_RANK" -eq 0 ]]; then
#     MASTER_NODE_IP="$CURRENT_IP"
# else
#     # 等待 head 节点写入 master IP
#     while [[ ! -s "$IP_LIST_FILE" ]]; do
#         echo "[WAIT] 等待 head 节点初始化共享文件..."
#         sleep 2
#     done
#     MASTER_NODE_IP=$(head -n1 "$IP_LIST_FILE")
# fi
# echo "[INFO] MASTER_NODE_IP=$MASTER_NODE_IP"

# ##### 仅 head 节点执行初始化 #####
# if [[ "$NODE_RANK" -eq 0 ]]; then
#     echo "[HEAD] 初始化共享文件..."
#     : > "$IP_LIST_FILE"
#     echo "$CURRENT_IP" >> "$IP_LIST_FILE"
#     sort -u "$IP_LIST_FILE" -o "$IP_LIST_FILE"

#     # 安装依赖（最好提前在镜像完成，这里只是演示）
#     if ! command -v nc >/dev/null; then
#         echo "[HEAD] 安装 netcat..."
#         apt-get update && apt-get install -y netcat-openbsd
#     fi
# else
#     # Worker 写入自己 IP
#     grep -q "$CURRENT_IP" "$IP_LIST_FILE" 2>/dev/null || {
#         echo "$CURRENT_IP" >> "$IP_LIST_FILE"
#         sort -u "$IP_LIST_FILE" -o "$IP_LIST_FILE"
#     }
# fi

# ##### 等待所有节点到齐 #####
# while true; do
#     CURRENT_NODE_COUNT=$(wc -l < "$IP_LIST_FILE")
#     echo "[INFO] 已注册节点数: $CURRENT_NODE_COUNT / $TARGET_WORLD_SIZE"
#     if [[ $CURRENT_NODE_COUNT -ge $TARGET_WORLD_SIZE ]]; then
#         echo "[READY] 节点全部到齐"
#         break
#     fi
#     sleep 5
# done

# ##### 如果是 root 用户，容许 root 运行 MPI #####
# if [[ "$(id -u)" -eq 0 ]]; then
#     export OMPI_ALLOW_RUN_AS_ROOT=1
#     export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# fi

# ##### 清理旧 Ray 状态 #####
# ray stop || true
# rm -rf /tmp/ray

# ##### 启动 Ray 集群 #####
# if [[ "$NODE_RANK" -eq 0 ]]; then
#     echo "[MASTER] 启动 Ray head..."
#     ray start --head \
#         --node-ip-address="$MASTER_NODE_IP" \
#         --port=$RAY_PORT \
#         --num-gpus=$GPUS_PER_NODE
# else
#     echo "[WORKER] 等待 head 节点就绪..."
#     until nc -z $MASTER_NODE_IP $RAY_PORT; do
#         echo "[WAIT] GCS 未就绪，重试中..."
#         sleep 2
#     done
#     sleep 5
#     echo "[WORKER] 启动 Ray worker..."
#     ray start \
#         --address=$MASTER_NODE_IP:$RAY_PORT \
#         --node-ip-address=$CURRENT_IP \
#         --num-gpus=$GPUS_PER_NODE
# fi

# ##### 同步：等待所有节点的 ray 启动完成 #####
# echo "[SYNC] 等待所有节点完成 Ray 启动..."
# if command -v mpirun >/dev/null; then
#     mpirun -np $TARGET_WORLD_SIZE /bin/true
# elif [[ -n "$SLURM_PROCID" ]]; then
#     srun hostname >/dev/null
# fi

# sleep 5
# echo "[DEBUG] Ray 状态:"
# ray status || true

# ##### 打印 GPU 状态 #####
# echo "[DEBUG] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# nvidia-smi || echo "[WARN] 无法调用 nvidia-smi"

# LOG_FILE="$SHARED_DIR/train_$(date +%Y%m%d_%H%M%S)_rank${NODE_RANK}.log"

# ##### 数据路径 #####
# gsm8k_train_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/train.parquet"
# gsm8k_test_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/test.parquet"
# train_files="['$gsm8k_train_path']"
# test_files="['$gsm8k_test_path']"

# ##### 启动分布式训练 #####
# torchrun \
#     --nnodes=$TARGET_WORLD_SIZE \
#     --nproc_per_node=1 \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_NODE_IP \
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
#     actor_rollout_ref.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
#     critic.optim.lr=1e-5 \
#     critic.model.use_remove_padding=True \
#     critic.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct \
#     critic.model.enable_gradient_checkpointing=False \
#     critic.ppo_micro_batch_size_per_gpu=8 \
#     critic.model.fsdp_config.param_offload=False \
#     critic.model.fsdp_config.optimizer_offload=False \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger='["wandb"]' \
#     trainer.project_name='test_experiments' \
#     trainer.experiment_name='Qwen2.5-7B-Instruct_test_ray_multi_nodes' \
#     trainer.n_gpus_per_node=$GPUS_PER_NODE \
#     trainer.nnodes=$TARGET_WORLD_SIZE \
#     trainer.save_freq=100000 \
#     trainer.test_freq=10 \
#     trainer.total_epochs=15 \
#     >> "${LOG_FILE}" 2>&1



# #!/bin/bash
# set -xe

# ##### 配置 #####
# GPUS_PER_NODE=8                               # 每节点 GPU 数量
# MASTER_PORT=29503                             # torchrun master 通信端口
# RAY_PORT=6382                                 # Ray 集群端口

# ##### 获取 MPI/Slurm 信息 #####
# if [[ -n "$OMPI_COMM_WORLD_SIZE" ]]; then
#     TARGET_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
# elif [[ -n "$SLURM_NTASKS" ]]; then
#     TARGET_WORLD_SIZE=$SLURM_NTASKS
# else
#     echo "[ERROR] 无法确定 TARGET_WORLD_SIZE，请确保在 MPI/Slurm 环境中运行"
#     exit 1
# fi

# if [[ -n "$OMPI_COMM_WORLD_RANK" ]]; then
#     NODE_RANK=$OMPI_COMM_WORLD_RANK
# elif [[ -n "$SLURM_PROCID" ]]; then
#     NODE_RANK=$SLURM_PROCID
# else
#     echo "[ERROR] 无法确定 NODE_RANK，请确保在 MPI/Slurm 环境中运行"
#     exit 1
# fi

# ##### 获取当前节点 IP #####
# CURRENT_IP=$(hostname -I | tr ' ' '\n' | grep '^10\.' | head -n1)
# echo "[INFO] 当前节点 IP: $CURRENT_IP, NODE_RANK=$NODE_RANK"

# ##### 确定 MASTER_NODE_IP #####
# if [[ "$NODE_RANK" -eq 0 ]]; then
#     MASTER_NODE_IP="$CURRENT_IP"
#     echo "[HEAD] MASTER_NODE_IP=$MASTER_NODE_IP"
# fi

# # 如果不是 HEAD 节点，需要获取 MASTER_NODE_IP
# # 在 MPI 下，mpirun 可以直接传 MASTER_NODE_IP 变量
# if [[ "$NODE_RANK" -ne 0 ]]; then
#     if [[ -n "$MASTER_NODE_IP" ]]; then
#         echo "[WORKER] 来自环境的 MASTER_NODE_IP=$MASTER_NODE_IP"
#     elif [[ -n "$MASTER_IP_ARG" ]]; then
#         MASTER_NODE_IP="$MASTER_IP_ARG"
#         echo "[WORKER] 从参数 MASTER_IP_ARG 获取 MASTER_NODE_IP=$MASTER_NODE_IP"
#     else
#         echo "[ERROR] Work 节点未获得 MASTER_NODE_IP，请用 mpirun/xargs 传递"
#         exit 1
#     fi
# fi

# ##### 容许 root MPI 运行 #####
# if [[ "$(id -u)" -eq 0 ]]; then
#     export OMPI_ALLOW_RUN_AS_ROOT=1
#     export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# fi

# ##### 清理旧 Ray 状态 #####
# ray stop || true
# rm -rf /tmp/ray

# ##### 安装依赖（最好提前装好） #####
# if ! command -v nc >/dev/null; then
#     echo "[INFO] 安装 netcat..."
#     apt-get update && apt-get install -y netcat-openbsd
# fi

# ##### 启动 Ray 集群 #####
# if [[ "$NODE_RANK" -eq 0 ]]; then
#     echo "[MASTER] 启动 Ray Head 节点..."
#     ray start --head \
#         --node-ip-address="$MASTER_NODE_IP" \
#         --port=$RAY_PORT \
#         --num-gpus=$GPUS_PER_NODE
#     echo "[MASTER] 等待 Worker 连接..."
# else
#     echo "[WORKER] 等待 Head 节点 $MASTER_NODE_IP:$RAY_PORT 就绪..."
#     until nc -z $MASTER_NODE_IP $RAY_PORT; do
#         echo "[WAIT] Head 未就绪，重试中..."
#         sleep 2
#     done
#     sleep 5
#     echo "[WORKER] 启动 Ray Worker..."
#     ray start \
#         --address=$MASTER_NODE_IP:$RAY_PORT \
#         --node-ip-address=$CURRENT_IP \
#         --num-gpus=$GPUS_PER_NODE
# fi

# ##### MPI Barrier 同步，确保所有节点完成 Ray 启动 #####
# echo "[SYNC] 等待所有节点完成 Ray 启动..."
# if command -v mpirun >/dev/null; then
#     mpirun -np $TARGET_WORLD_SIZE /bin/true
# elif command -v srun >/dev/null; then
#     srun hostname >/dev/null
# fi

# sleep 5
# echo "[DEBUG] Ray 集群状态:"
# ray status || true

# ##### GPU 状态 #####
# echo "[DEBUG] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# nvidia-smi || echo "[WARN] 无法调用 nvidia-smi"

# LOG_FILE="train_$(date +%Y%m%d_%H%M%S)_rank${NODE_RANK}.log"

# ##### 数据路径 #####
# gsm8k_train_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/train.parquet"
# gsm8k_test_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/test.parquet"
# train_files="['$gsm8k_train_path']"
# test_files="['$gsm8k_test_path']"

# ##### 启动分布式训练 #####
# torchrun \
#     --nnodes=$TARGET_WORLD_SIZE \
#     --nproc_per_node=1 \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_NODE_IP \
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
#     actor_rollout_ref.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
#     critic.optim.lr=1e-5 \
#     critic.model.use_remove_padding=True \
#     critic.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct \
#     critic.model.enable_gradient_checkpointing=False \
#     critic.ppo_micro_batch_size_per_gpu=8 \
#     critic.model.fsdp_config.param_offload=False \
#     critic.model.fsdp_config.optimizer_offload=False \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger='["wandb"]' \
#     trainer.project_name='test_experiments' \
#     trainer.experiment_name='Qwen2.5-7B-Instruct_test_ray_multi_nodes' \
#     trainer.n_gpus_per_node=$GPUS_PER_NODE \
#     trainer.nnodes=$TARGET_WORLD_SIZE \
#     trainer.save_freq=100000 \
#     trainer.test_freq=10 \
#     trainer.total_epochs=15 \
#     >> "${LOG_FILE}" 2>&1





# #!/bin/bash
# set -xe

# ##############################
# # 配置参数
# ##############################
# GPUS_PER_NODE=8
# MASTER_PORT=29503
# RAY_PORT=6382

# ##############################
# # 从 MPI/Slurm 获取参数
# ##############################

# # 获取总节点数
# if [[ -n "$OMPI_COMM_WORLD_SIZE" ]]; then
#     TARGET_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
# elif [[ -n "$SLURM_NTASKS" ]]; then
#     TARGET_WORLD_SIZE=$SLURM_NTASKS
# else
#     echo "[ERROR] 无法获取 TARGET_WORLD_SIZE，请确保在 MPI/Slurm 环境运行"
#     exit 1
# fi

# # 获取当前节点 rank
# if [[ -n "$OMPI_COMM_WORLD_RANK" ]]; then
#     NODE_RANK=$OMPI_COMM_WORLD_RANK
# elif [[ -n "$SLURM_PROCID" ]]; then
#     NODE_RANK=$SLURM_PROCID
# else
#     echo "[ERROR] 无法获取 NODE_RANK，请确保在 MPI/Slurm 环境运行"
#     exit 1
# fi

# CURRENT_IP=$(hostname -I | tr ' ' '\n' | grep '^10\.' | head -n1)
# echo "[INFO] 当前节点 IP: $CURRENT_IP, NODE_RANK=$NODE_RANK"

# # Head 节点 IP
# if [[ "$NODE_RANK" -eq 0 ]]; then
#     MASTER_NODE_IP="$CURRENT_IP"
#     echo "[HEAD] MASTER_NODE_IP=$MASTER_NODE_IP"
# else
#     MASTER_NODE_IP=${MASTER_NODE_IP:?必须设置 MASTER_NODE_IP}
#     echo "[WORKER] MASTER_NODE_IP=$MASTER_NODE_IP"
# fi

# ##############################
# # 允许 root 运行 MPI/Ray
# ##############################
# if [[ "$(id -u)" -eq 0 ]]; then
#     export OMPI_ALLOW_RUN_AS_ROOT=1
#     export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# fi

# ##############################
# # 清理旧 Ray 状态
# ##############################
# ray stop || true
# rm -rf /tmp/ray

# ##############################
# # 安装依赖（仅 Head 安装）
# ##############################
# if ! command -v nc >/dev/null; then
#     if [[ "$NODE_RANK" -eq 0 ]]; then
#         echo "[HEAD] 安装 netcat..."
#         apt-get update && apt-get install -y netcat-openbsd
#     else
#         echo "[WORKER] 跳过安装 netcat（假定提前安装好）"
#     fi
# fi

# ##############################
# # 启动 Ray
# ##############################
# if [[ "$NODE_RANK" -eq 0 ]]; then
#     echo "[MASTER] 启动 Ray Head 节点..."
#     ray start --head \
#         --node-ip-address="$MASTER_NODE_IP" \
#         --port=$RAY_PORT \
#         --num-gpus=$GPUS_PER_NODE
#     echo "[MASTER] 等待 Worker 连接..."
# else
#     echo "[WORKER] 等待 Head 节点 ${MASTER_NODE_IP}:${RAY_PORT} 就绪..."
#     until nc -z $MASTER_NODE_IP $RAY_PORT; do
#         echo "[WAIT] Head 未就绪，重试中..."
#         sleep 2
#     done
#     sleep 5  # 确保 Redis 完全初始化
#     echo "[WORKER] 启动 Ray Worker 节点..."
#     ray start \
#         --address=$MASTER_NODE_IP:$RAY_PORT \
#         --node-ip-address=$CURRENT_IP \
#         --num-gpus=$GPUS_PER_NODE
# fi

# ##############################
# # 安全 Barrier 同步（非递归 MPI）
# ##############################
# echo "[SYNC] 等待 Ray 集群稳定..."
# sleep 5  # 简单延迟代替嵌套 mpirun barrier

# echo "[DEBUG] Ray 集群状态:"
# ray status || true

# ##############################
# # 打印 GPU 状态
# ##############################
# echo "[DEBUG] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# nvidia-smi || echo "[WARN] nvidia-smi 执行失败"

# ##############################
# # 数据路径
# ##############################
# gsm8k_train_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/train.parquet"
# gsm8k_test_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/test.parquet"
# train_files="['$gsm8k_train_path']"
# test_files="['$gsm8k_test_path']"

# LOG_FILE="train_$(date +%Y%m%d_%H%M%S)_rank${NODE_RANK}.log"

# ##############################
# # 启动分布式训练
# ##############################
# torchrun \
#     --nnodes=$TARGET_WORLD_SIZE \
#     --nproc_per_node=1 \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_NODE_IP \
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
#     actor_rollout_ref.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=256 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.actor.use_kl_loss=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
#     critic.optim.lr=1e-5 \
#     critic.model.use_remove_padding=True \
#     critic.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct \
#     critic.model.enable_gradient_checkpointing=False \
#     critic.ppo_micro_batch_size_per_gpu=8 \
#     critic.model.fsdp_config.param_offload=False \
#     critic.model.fsdp_config.optimizer_offload=False \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger='["wandb"]' \
#     trainer.project_name='test_experiments' \
#     trainer.experiment_name='Qwen2.5-7B-Instruct_test_ray_multi_nodes' \
#     trainer.n_gpus_per_node=$GPUS_PER_NODE \
#     trainer.nnodes=$TARGET_WORLD_SIZE \
#     trainer.save_freq=100000 \
#     trainer.test_freq=10 \
#     trainer.total_epochs=15 \
#     >> "${LOG_FILE}" 2>&1




#!/bin/bash
set -xe

######################
# 配置参数
######################
GPUS_PER_NODE=8
MASTER_PORT=29503
RAY_PORT=6382

######################
# 获取 MPI/Slurm 总节点数
######################
if [[ -n "$OMPI_COMM_WORLD_SIZE" ]]; then
    TARGET_WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
elif [[ -n "$SLURM_NTASKS" ]]; then
    TARGET_WORLD_SIZE=$SLURM_NTASKS
else
    echo "[ERROR] 无法获取 TARGET_WORLD_SIZE，请确保在 MPI/Slurm 环境运行"
    exit 1
fi

######################
# 获取当前进程 rank
######################
if [[ -n "$OMPI_COMM_WORLD_RANK" ]]; then
    NODE_RANK=$OMPI_COMM_WORLD_RANK
elif [[ -n "$SLURM_PROCID" ]]; then
    NODE_RANK=$SLURM_PROCID
else
    echo "[ERROR] 无法获取 NODE_RANK，请确保在 MPI/Slurm 环境运行"
    exit 1
fi

######################
# 获取当前节点 IP
######################
CURRENT_IP=$(hostname -I | tr ' ' '\n' | grep '^10\.' | head -n1)
echo "[INFO] 当前节点 IP: $CURRENT_IP, NODE_RANK=$NODE_RANK"

######################
# MASTER_NODE_IP
######################
if [[ "$NODE_RANK" -eq 0 ]]; then
    MASTER_NODE_IP="$CURRENT_IP"
else
    MASTER_NODE_IP=${MASTER_NODE_IP:?必须设置 MASTER_NODE_IP}
fi

# ######################
# # 允许 root 运行 MPI / Ray
# ######################
# if [[ "$(id -u)" -eq 0 ]]; then
#     export OMPI_ALLOW_RUN_AS_ROOT=1
#     export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
# fi

# ######################
# # 清理旧 Ray 状态
# ######################
# ray stop || true
# rm -rf /tmp/ray

# ######################
# # 安装依赖（仅 Head）
# ######################
# if ! command -v nc >/dev/null; then
#     if [[ "$NODE_RANK" -eq 0 ]]; then
#         echo "[HEAD] 安装 netcat..."
#         apt-get update && apt-get install -y netcat-openbsd
#     else
#         echo "[WORKER] 跳过安装 netcat（假定镜像里已存在）"
#     fi
# fi

# ######################
# # 端口范围配置（避免冲突）
# ######################
# HEAD_WORKER_MIN_PORT=30000
# HEAD_WORKER_MAX_PORT=30100
# WORKER_MIN_PORT=30101
# WORKER_MAX_PORT=30200
# HEAD_METRICS_PORT=18000
# WORKER_METRICS_PORT=18001

# ######################
# # 启动 Ray
# ######################
# if [[ "$NODE_RANK" -eq 0 ]]; then
#     echo "[MASTER] 启动 Ray Head 节点..."
#     ray start --head \
#         --node-ip-address="$MASTER_NODE_IP" \
#         --port=$RAY_PORT \
#         --num-gpus=$GPUS_PER_NODE \
#         --min-worker-port=${HEAD_WORKER_MIN_PORT} \
#         --max-worker-port=${HEAD_WORKER_MAX_PORT} \
#         --metrics-export-port=${HEAD_METRICS_PORT}
# else
#     echo "[WORKER] 等待 Head 节点就绪..."
#     until nc -z $MASTER_NODE_IP $RAY_PORT; do
#         echo "[WAIT] Head 未就绪，重试中..."
#         sleep 2
#     done
#     sleep 5
#     echo "[WORKER] 启动 Ray Worker 节点..."
#     ray start \
#         --address=$MASTER_NODE_IP:$RAY_PORT \
#         --node-ip-address=$CURRENT_IP \
#         --num-gpus=$GPUS_PER_NODE \
#         --min-worker-port=${WORKER_MIN_PORT} \
#         --max-worker-port=${WORKER_MAX_PORT} \
#         --metrics-export-port=${WORKER_METRICS_PORT}
# fi

# ######################
# # 简单同步（避免嵌套 mpirun）
# ######################
# echo "[SYNC] 等待 Ray 集群稳定..."
# sleep 5

# echo "[DEBUG] Ray 集群状态:"
# ray status || true

######################
# 数据路径
######################
gsm8k_train_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/train.parquet"
gsm8k_test_path="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data/gsm8k/test.parquet"
train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

RESULTS_DIR="/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/results"
mkdir -p "$RESULTS_DIR"
LOG_FILE="$RESULTS_DIR/train_$(date +%Y%m%d_%H%M%S)_rank${NODE_RANK}.log"


######################
# 启动分布式训练（交给 verl.trainer.main_ppo 管理 Ray）
######################
echo "[INFO] 启动真实训练（Ray 由 verl.trainer.main_ppo 统一启动）..."
torchrun \
    --nnodes=$TARGET_WORLD_SIZE \
    --nproc_per_node=1 \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_NODE_IP \
    --master_port=$MASTER_PORT \
    -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=/mnt/tidal-alsh01/usr/chenyiqun/base_models/Qwen/Qwen2.5-7B-Instruct \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["wandb"]' \
    trainer.project_name='test_experiments' \
    trainer.experiment_name='Qwen2.5-7B-Instruct_test_ray_multi_nodes' \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$TARGET_WORLD_SIZE \
    trainer.save_freq=100000 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    2>&1 | tee "$LOG_FILE"

echo "[INFO] Rank $NODE_RANK 日志保存在 $LOG_FILE"