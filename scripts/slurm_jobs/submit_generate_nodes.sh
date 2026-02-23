#!/bin/bash
#SBATCH --job-name=gen_nodes
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/generate_nodes_%j.log

# 激活环境
source venv_t4/bin/activate

# 设置 Python 路径
export PYTHONPATH="/users/sl3753/scratch/GCN_project:$PYTHONPATH"

# 创建日志目录
mkdir -p logs

# 获取参数
DATASET=$1
GRID_H=$2
GRID_W=$3

# 检查参数
if [ -z "$DATASET" ] || [ -z "$GRID_H" ] || [ -z "$GRID_W" ]; then
    echo "用法: sbatch submit_generate_nodes.sh <dataset> <grid_h> <grid_w>"
    echo "示例: sbatch submit_generate_nodes.sh 776 8 8"
    exit 1
fi

echo "开始生成图节点..."
echo "数据集: $DATASET"
echo "网格大小: ${GRID_H}x${GRID_W}"
echo "时间: $(date)"
echo "================================================================"

# 运行节点生成
python scripts/generate_graph_nodes.py \
    --dataset "$DATASET" \
    --grid-h "$GRID_H" \
    --grid-w "$GRID_W"

echo "================================================================"
echo "完成时间: $(date)"
