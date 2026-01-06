#!/bin/bash
set -e

# 节点列表文件，默认 hosts.txt
NODES_FILE=${NODES_FILE:-hosts.txt}
# 脚本路径
REMOTE_SCRIPT_PATH=${REMOTE_SCRIPT_PATH:-"/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl"}
# 数据路径
TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-"/mnt/tidal-alsh01/usr/chenyiqun/research_project/verl/data"}

HOSTS=($(cat "$NODES_FILE"))
NNODES=${#HOSTS[@]}

echo "[INFO] 检测节点数量: $NNODES"
echo "-------------------------------------"

# 先检查本机是否有 SSH 公钥，没有的话生成
if [ ! -f "$HOME/.ssh/id_rsa.pub" ]; then
    echo "[INFO] 本机不存在 SSH 公钥，开始生成..."
    ssh-keygen -t rsa -b 4096 -N "" -f "$HOME/.ssh/id_rsa"
fi

# 可选：推送公钥到每个节点（仅运行一次即可）
read -p "[询问] 是否推送 SSH 公钥到所有节点以配置免密登录？[y/N] " CONFIRM
if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    for HOST in "${HOSTS[@]}"; do
        echo "[INFO] 向 $HOST 推送公钥..."
        ssh-copy-id -i "$HOME/.ssh/id_rsa.pub" "$USER@$HOST" || {
            echo "⚠️ 无法推送公钥到 $HOST，请检查网络/账户/密码"
        }
    done
fi

# 遍历每个节点做检测
for NODE in "${HOSTS[@]}"; do
    echo "[INFO] >>> 检查节点: $NODE"

    # 检查 ping
    if ping -c 1 -W 2 "$NODE" &> /dev/null; then
        echo "✅ 网络连通性: OK"
    else
        echo "❌ 网络连通性: FAIL"
        continue
    fi

    # 检查 SSH 免密
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$NODE" "echo 1" &> /dev/null; then
        echo "✅ SSH免密登录: OK"
    else
        echo "❌ SSH免密登录: FAIL"
        continue
    fi

    # GPU检查
    echo "[INFO] 检查 GPU..."
    if ssh "$NODE" "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"; then
        GPU_COUNT=$(ssh "$NODE" "nvidia-smi -L | wc -l")
        echo "✅ 检测到 GPU 数量: $GPU_COUNT"
    else
        echo "❌ GPU检查失败"
        continue
    fi

    # Python/PyTorch 检查
    echo "[INFO] 检查 Python/PyTorch..."
    if ssh "$NODE" "python3 --version"; then
        ssh "$NODE" "python3 -c 'import torch; print(\"PyTorch版本:\", torch.__version__); print(\"GPU可用:\", torch.cuda.is_available())'" || {
            echo "❌ PyTorch未安装或GPU不可用"
            continue
        }
    else
        echo "❌ Python未安装"
        continue
    fi

    # 文件路径检查
    echo "[INFO] 检查路径一致性..."
    ssh "$NODE" "[ -d \"$REMOTE_SCRIPT_PATH\" ] && echo '✅ 脚本路径存在' || echo '❌ 脚本路径不存在'"
    ssh "$NODE" "[ -d \"$TRAIN_DATA_PATH\" ] && echo '✅ 数据路径存在' || echo '❌ 数据路径不存在'"

    echo "-------------------------------------"
done

echo "[INFO] 所有节点检测完成"
