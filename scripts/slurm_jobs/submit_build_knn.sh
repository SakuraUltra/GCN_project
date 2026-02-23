#!/bin/bash
#SBATCH --job-name=build_knn
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_knn_%j.log

# 激活环境
source venv_t4/bin/activate

# 设置 Python 路径
export PYTHONPATH="/users/sl3753/scratch/GCN_project:$PYTHONPATH"

# 创建日志目录
mkdir -p logs

# 获取参数
NODES_DIR=$1
K=${2:-8}
METRIC=${3:-cosine}

# 检查参数
if [ -z "$NODES_DIR" ]; then
    echo "用法: sbatch submit_build_knn.sh <nodes_dir> [k] [metric]"
    echo "示例: sbatch submit_build_knn.sh outputs/graph_nodes/776_baseline_grid_8x8 8 cosine"
    exit 1
fi

echo "开始构建 kNN 图..."
echo "节点目录: $NODES_DIR"
echo "k 值: $K"
echo "相似度度量: $METRIC"
echo "时间: $(date)"
echo "================================================================"

# 运行 kNN 构建（带早期停止）
python scripts/build_knn_graph.py \
    --nodes-dir "$NODES_DIR" \
    --k "$K" \
    --metric "$METRIC" \
    --detach

echo "================================================================"
echo "完成时间: $(date)"
