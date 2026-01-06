#!/bin/bash
set -x

##################################
# Rank/环境调试打印
##################################
echo "===== DEBUG RANK INFO ====="
echo "[DEBUG] Hostname: $(hostname)"
echo "[DEBUG] Date: $(date)"
echo "[DEBUG] JOB_ID=${JOB_ID}"
echo "[DEBUG] IP_FILE=${IP_FILE}"
echo "[DEBUG] OMPI_COMM_WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-N/A}"
echo "[DEBUG] OMPI_COMM_WORLD_RANK=${OMPI_COMM_WORLD_RANK:-N/A}"
echo "[DEBUG] OMPI_COMM_WORLD_LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-N/A}"
echo "[DEBUG] SLURM_NODEID=${SLURM_NODEID:-N/A}"
echo "[DEBUG] SLURM_PROCID=${SLURM_PROCID:-N/A}"
echo "[DEBUG] RANK=${RANK:-N/A}"
echo "[DEBUG] LOCAL_RANK=${LOCAL_RANK:-N/A}"
echo "[DEBUG] GLOBAL_RANK will be calculated next..."
echo "============================"

##### 基本配置 #####
MASTER_PORT=29505
RAY_PORT=6385
GPUS_PER_NODE=8
NPROC_PER_NODE=1
WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-${SLURM_NTASKS:-2}}

# 从提交脚本传入参数
: "${JOB_ID:?JOB_ID 未设置}"
: "${IP_FILE:?IP_FILE 未设置}"
mkdir -p "$(dirname "$IP_FILE")"

##### 获取 RANK 信息 #####
GLOBAL_RANK="${RANK:-${OMPI_COMM_WORLD_RANK:-0}}"
LOCAL_RANK="${LOCAL_RANK:-${OMPI_COMM_WORLD_LOCAL_RANK:-0}}"
echo "[DEBUG] After calculation: GLOBAL_RANK=$GLOBAL_RANK, LOCAL_RANK=$LOCAL_RANK"

##### 获取本节点 IP #####
CURRENT_NODE=$(hostname)
CURRENT_IP=$(hostname -I | grep -o "10\.[0-9]\+\.[0-9]\+\.[0-9]\+" | head -1)
if [[ -z "$CURRENT_IP" ]]; then
    echo "[ERROR] 节点 $CURRENT_NODE 无法获取IP" >&2
    exit 1
fi

# 写入IP文件（加锁防竞争）
(
flock 200
grep -qx "$CURRENT_IP" "$IP_FILE" 2>/dev/null || echo "$CURRENT_IP" >> "$IP_FILE"
) 200>"$IP_FILE.lock"

echo "[INFO] 节点 $CURRENT_NODE IP: $CURRENT_IP 写入 $IP_FILE"

##### 等待所有节点都写完 #####
while [[ $(wc -l < "$IP_FILE") -lt "$WORLD_SIZE" ]]; do
    echo "[INFO] 当前IP记录数量$(wc -l < "$IP_FILE")/需要$WORLD_SIZE，等待..."
    sleep 5
done

##### 读取节点IP（并与 GLOBAL_RANK 对齐） #####
# 所有节点先加载 ALL_IPS 仅用于调试展示
mapfile -t ALL_IPS < "$IP_FILE"
echo "[INFO] 当前 ALL_IPS 列表: ${ALL_IPS[*]}"

# 如果是 rank0 节点，写 master_ip.txt
MASTER_IP_FILE="$(dirname "$IP_FILE")/master_ip.txt"

# Rank 0 节点写入 Master IP
if [[ "$GLOBAL_RANK" -eq 0 ]]; then
    MASTER_IP="$CURRENT_IP"
    (
        flock 201
        echo "$MASTER_IP" > "$MASTER_IP_FILE"
        sync  # 确保写入落盘（处理 NFS 延迟）
    ) 201>"$MASTER_IP_FILE.lock"
    echo "[INFO] 我是 GLOBAL_RANK=0，已写 MASTER_IP=$MASTER_IP 到 $MASTER_IP_FILE"
else
    # 等待有效的 MASTER_IP 文件
    until [[ -s "$MASTER_IP_FILE" ]] && grep -Eq "^10\.[0-9]+\.[0-9]+\.[0-9]+$" "$MASTER_IP_FILE"; do
        echo "[WORKER] 等待有效的 MASTER_IP 文件（当前状态：文件大小=$(stat -c%s "$MASTER_IP_FILE" 2>/dev/null || echo 0) 内容='$(cat "$MASTER_IP_FILE" 2>/dev/null || echo "")'）..."
        sleep 3
    done
    MASTER_IP=$(cat "$MASTER_IP_FILE")
    echo "[INFO] 从 $MASTER_IP_FILE 读取 MASTER_IP=$MASTER_IP"
fi
# if [[ "$GLOBAL_RANK" -eq 0 ]]; then
#     MASTER_IP="$CURRENT_IP"
#     echo "$MASTER_IP" > "$MASTER_IP_FILE"
#     echo "[INFO] 我是 GLOBAL_RANK=0，已写 MASTER_IP=$MASTER_IP 到 $MASTER_IP_FILE"
# else
#     # 等待 master_ip.txt 文件出现
#     until [[ -s "$MASTER_IP_FILE" ]]; do
#         echo "[INFO] 等待 master_ip.txt 被 rank0 节点写入..."
#         sleep 5
#     done
#     MASTER_IP=$(cat "$MASTER_IP_FILE")
#     echo "[INFO] 从 $MASTER_IP_FILE 读取 MASTER_IP=$MASTER_IP"
# fi

# 生成 WORKER_IPS（ALL_IPS 去掉 MASTER_IP）
WORKER_IPS=()
for ip in "${ALL_IPS[@]}"; do
    if [[ "$ip" != "$MASTER_IP" ]]; then
        WORKER_IPS+=("$ip")
    fi
done
echo "[INFO] WORKER_IPS 列表: ${WORKER_IPS[*]}"


#### 判断角色
if [[ "$GLOBAL_RANK" -eq 0 ]]; then
    ROLE="Master"
else
    ROLE="Worker"
fi
echo "[INFO] 当前角色: $ROLE"

##################################
# !!! Node Leader 判定 !!!
# 每个node第一个抢到锁的进程执行Ray/Torchrun
##################################
NODE_LOCK="/tmp/node_start_${CURRENT_NODE}.lock"
NODE_LEADER=false
if ( set -o noclobber; echo "$$" > "$NODE_LOCK") 2> /dev/null; then
    NODE_LEADER=true
    echo "[DEBUG] 我是节点 $CURRENT_NODE 的 Node Leader"
else
    echo "[DEBUG] 节点 $CURRENT_NODE 已有进程启动任务，跳过Ray/torchrun启动"
fi

# 安装nc
apt update -y
apt install -y netcat-openbsd

##### 启动 Ray（每节点一次）
RAY_READY_FLAG="$(dirname "$IP_FILE")/ray_head_ready.flag"

if $NODE_LEADER; then
    if [[ "$ROLE" == "Master" ]]; then
        echo "[MASTER] 启动 Ray Head ($CURRENT_IP)"
        ray stop || true
        rm -rf /tmp/ray
        ray start --head --node-ip-address="$CURRENT_IP" --port=$RAY_PORT --num-gpus=$GPUS_PER_NODE

        echo "[MASTER] 等待 Ray Head 初始化完成..."
        # 原来这里 nc 检测端口 -> 改成 ray status 检测
        until ray status --address=$CURRENT_IP:$RAY_PORT >/dev/null 2>&1; do
            echo "[MASTER] 等待 Ray cluster 启动完成..."
            sleep 3
        done

        # 额外延时，确保稳定
        sleep 5

        # 写一个 Ray Ready 标志文件，告诉 Worker 可以启动了
        touch "$RAY_READY_FLAG"
        echo "[MASTER] Ray Head 已就绪，写入标志文件 $RAY_READY_FLAG"

    else
        echo "[WORKER] 等待 Master Ray Head 就绪（检测标志文件 + 集群状态）..."
        # 等 Master 写好了 Ready 标志文件
        until [[ -f "$RAY_READY_FLAG" ]]; do
            echo "[WORKER] 等待 Ray Head Ready 标志文件..."
            sleep 3
        done

        # 再检测 Ray cluster 是否健康
        until ray status --address=$MASTER_IP:$RAY_PORT >/dev/null 2>&1; do
            echo "[WORKER] Ray cluster 尚未完全就绪，等待中..."
            sleep 3
        done

        echo "[WORKER] 启动 Ray Worker（连接 $MASTER_IP）"
        ray stop || true
        rm -rf /tmp/ray
        ray start --address=$MASTER_IP:$RAY_PORT --num-gpus=$GPUS_PER_NODE
    fi
fi

LOG_DIR="$(dirname "$IP_FILE")"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S)_node${GLOBAL_RANK}_local${LOCAL_RANK}.log"
echo "[INFO] 日志将保存到 $LOG_FILE"

###################################
# 安装依赖
###################################
bash /mnt/tidal-alsh01/usr/chenyiqun/research_project/knowledge_search/rl/verl/scripts/install_vllm_sglang_mcore_0.7.sh
pip install --no-deps -e /mnt/tidal-alsh01/usr/chenyiqun/research_project/knowledge_search/rl/verl
pip install "numpy<2.0"
pip install --upgrade deepspeed
pip install datasets==4.2.0 pyarrow==21.0.0

##### 数据路径 #####
hotpot_qa_train_path=/mnt/tidal-alsh01/usr/chenyiqun/datasets/QA_datasets/verl_format_data/hotpot_qa/train_verl.parquet
hotpot_qa_test_path=/mnt/tidal-alsh01/usr/chenyiqun/datasets/QA_datasets/verl_format_data/hotpot_qa/test_verl.parquet
train_files="['$hotpot_qa_train_path']"
test_files="['$hotpot_qa_test_path']"

##### 启动训练（Master 节点只让 LOCAL_RANK=0 执行，其他节点 NODE_LEADER 执行） #####
if { [[ "$GLOBAL_RANK" -eq 0 && "$LOCAL_RANK" -eq 0 ]] || { [[ "$GLOBAL_RANK" -ne 0 ]] && $NODE_LEADER; }; }; then

    echo "[INFO] Launching torchrun:"
    echo "[INFO] Host: $(hostname)"
    echo "[INFO] GLOBAL_RANK=$GLOBAL_RANK, LOCAL_RANK=$LOCAL_RANK, ROLE=$ROLE, NODE_LEADER=$NODE_LEADER"
    echo "[INFO] MASTER_IP=$MASTER_IP, MASTER_PORT=$MASTER_PORT"

    torchrun \
        --nnodes=$WORLD_SIZE \
        --nproc_per_node=$NPROC_PER_NODE \
        --node_rank=$GLOBAL_RANK \
        --master_addr=$MASTER_IP \
        --master_port=$MASTER_PORT \
        -m verl.trainer.main_ppo \
        algorithm.adv_estimator=gae \
        data.train_files="$train_files" \
        data.val_files="$test_files" \
        data.train_batch_size=768 \
        data.val_batch_size=1024 \
        data.max_prompt_length=3072 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=/mnt/tidal-alsh01/usr/chenyiqun/research_project/knowledge_search/sft_ckpt/Qwen2.5-7B-Instruct/no_end_new_prompt \
        actor_rollout_ref.model.enable_gradient_checkpointing=False \
        actor_rollout_ref.actor.optim.lr=5e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=128 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.model.path=/mnt/tidal-alsh01/usr/chenyiqun/research_project/knowledge_search/sft_ckpt/Qwen2.5-7B-Instruct/no_end_new_prompt \
        critic.model.enable_gradient_checkpointing=False \
        critic.ppo_micro_batch_size_per_gpu=4 \
        critic.model.fsdp_config.param_offload=False \
        critic.model.fsdp_config.optimizer_offload=False \
        algorithm.use_kl_in_reward=False \
        trainer.critic_warmup=0 \
        trainer.logger='["wandb"]' \
        trainer.project_name='knowledge_search_hotpotqa' \
        trainer.experiment_name='bs_768_lr5e-6_longtext' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=4 \
        trainer.save_freq=100000 \
        trainer.test_freq=6 \
        trainer.total_epochs=15 $@ \
        data.return_raw_chat=True \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.actor.ppo_epochs=1 \
        trainer.val_before_train=True \
        >> "$LOG_FILE" 2>&1
fi

echo "开始sleep 10 day"
sleep 864000
echo "结束"
