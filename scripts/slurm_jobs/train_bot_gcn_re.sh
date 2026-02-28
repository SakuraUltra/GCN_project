#!/bin/bash
#SBATCH --job-name=bot_gcn_re
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/bot_gcn_re_%j.out
#SBATCH --error=logs/bot_gcn_re_%j.err

# BoT-GCN Training with Random Erasing
# Ablation Study: Effect of Random Erasing on Occlusion Robustness

echo "=========================================="
echo "BoT-GCN Training with Random Erasing"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Environment
source /users/sl3753/scratch/GCN_project/venv_t4/bin/activate
cd /users/sl3753/scratch/GCN_project

# GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""

# Training
echo "Starting training..."
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/bot_gcn_776_re.yaml

echo ""
echo "Training completed at: $(date)"
echo "=========================================="
