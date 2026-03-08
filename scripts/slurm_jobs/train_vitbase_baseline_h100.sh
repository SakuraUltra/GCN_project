#!/bin/bash
#SBATCH --job-name=vitbase_baseline
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vitbase_baseline_h100_%j.out
#SBATCH --error=logs/vitbase_baseline_h100_%j.err

# 激活虚拟环境
source /users/sl3753/scratch/GCN_project/venv_t4/bin/activate
cd /users/sl3753/scratch/GCN_project

# 打印环境信息
echo "=========================================="
echo "ViT-Base Baseline Training on H100 (with RE)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: bot_vitbase_baseline_776.yaml"
echo "Model: ViT-Base/21k Baseline (NO GCN)"
echo "=========================================="

# 运行训练
echo ""
echo "Starting ViT-Base Baseline training with Random Erasing..."
nvidia-smi

python3 scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/bot_vitbase_baseline_776.yaml

echo ""
echo "Training completed at $(date)"
