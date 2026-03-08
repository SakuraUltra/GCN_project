#!/bin/bash
#SBATCH --job-name=vitbase_gcn
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vitbase_gcn_h100_%j.out
#SBATCH --error=logs/vitbase_gcn_h100_%j.err

# ViT-Base/21k + GCN Training on H100
# VeRi-776 Vehicle Re-ID with Larger Transformer Backbone

echo "=========================================="
echo "ViT-Base/21k GCN Training on H100"
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
echo "Starting ViT-Base training..."
python -u scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/bot_vitbase_gcn_776.yaml

echo ""
echo "Training completed at: $(date)"
echo "=========================================="
