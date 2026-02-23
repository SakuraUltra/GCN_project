#!/bin/bash
#SBATCH --job-name=bot_gcn_776
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# BoT-GCN Training on VeRi-776
# All logs will be saved in outputs/*/training.log

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 激活环境
source venv_t4/bin/activate

# 训练配置
CONFIG="configs/gcn_transformer_configs/bot_gcn_776.yaml"
OUTPUT_DIR="outputs/bot_gcn_776_v2"

echo ""
echo "Configuration:"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT_DIR"
echo "  Log: $OUTPUT_DIR/training.log"
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
