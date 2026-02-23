#!/bin/bash
#SBATCH --job-name=extract_feat
#SBATCH --partition=gpuplus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00

# 提取CNN特征图脚本
# 用于图准备阶段 Step 1

PROJECT_DIR="/users/sl3753/scratch/GCN_project"
cd ${PROJECT_DIR}

# 解析参数
DATASET=${1:-"776"}  # 默认VeRi-776
CHECKPOINT=${2:-""}  # checkpoint路径

if [ "$DATASET" == "776" ]; then
    DATA_ROOT="data/dataset/776_DataSet"
    if [ -z "$CHECKPOINT" ]; then
        CHECKPOINT="outputs/bot_baseline_1_1/776/baseline_run_01/best_model.pth"
    fi
    OUTPUT_DIR="outputs/features/776_baseline"
elif [ "$DATASET" == "VehicleID" ]; then
    DATA_ROOT="data/dataset/VehicleID_V1.0"
    if [ -z "$CHECKPOINT" ]; then
        CHECKPOINT="outputs/bot_baseline_1_1/VehicleID/baseline_run_01/best_model.pth"
    fi
    OUTPUT_DIR="outputs/features/VehicleID_baseline"
else
    echo "错误: 未知数据集 $DATASET"
    echo "用法: sbatch submit_extract_features.sh [776|VehicleID] [checkpoint_path]"
    exit 1
fi

echo "========================================"
echo "特征提取任务"
echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "数据集: $DATASET"
echo "Checkpoint: $CHECKPOINT"
echo "输出目录: $OUTPUT_DIR"
echo "========================================"

# 检查checkpoint是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: Checkpoint文件不存在: $CHECKPOINT"
    exit 1
fi

# 加载模块
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.6.0

# 激活虚拟环境
source ${PROJECT_DIR}/venv_t4/bin/activate

# 验证GPU
echo ""
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# 运行特征提取
echo ""
echo "开始提取特征..."
python scripts/extract_features.py \
    --config configs/baseline_configs/bot_baseline.yaml \
    --checkpoint ${CHECKPOINT} \
    --data_root ${DATA_ROOT} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 128 \
    --num_workers 12

echo ""
echo "========================================"
echo "特征提取完成！"
echo "输出目录: $OUTPUT_DIR"
echo "========================================"
