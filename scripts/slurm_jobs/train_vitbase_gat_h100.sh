#!/bin/bash
#SBATCH --job-name=vitbase_gat
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/vitbase_gat_h100_%j.out
#SBATCH --error=logs/vitbase_gat_h100_%j.err

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 进入项目目录
cd /users/sl3753/scratch/GCN_project

# 激活虚拟环境
source venv_t4/bin/activate

# 显示 GPU 信息
nvidia-smi

# 开始训练
echo "Starting ViT-Base + GAT training on H100..."
python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/bot_vitbase_gat_776.yaml \
    2>&1 | tee outputs/logs/vitbase_gat_h100_run1.log

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
