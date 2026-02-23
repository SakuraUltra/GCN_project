#!/bin/bash
#SBATCH --job-name=bot_gcn_vid_a40
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/bot_gcn_vehicleid_a40_%j.out
#SBATCH --error=logs/bot_gcn_vehicleid_a40_%j.err

# BoT-GCN Training on VehicleID with A40
# Step 4: Train CNN+GCN Model

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 激活环境
source venv_t4/bin/activate

# 创建日志目录
mkdir -p logs

# 训练配置
CONFIG="configs/gcn_transformer_configs/bot_gcn_vehicleid.yaml"
OUTPUT_DIR="outputs/bot_gcn_vehicleid_a40"

echo ""
echo "Configuration:"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT_DIR"
echo ""

# 训练
echo "Starting training..."
python scripts/training/train_bot_gcn.py \
    --config $CONFIG \
    --output_dir $OUTPUT_DIR

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
