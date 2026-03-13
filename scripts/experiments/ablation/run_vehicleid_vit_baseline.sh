#!/bin/bash
#SBATCH --job-name=vid_vit_baseline
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/vit_baseline_vehicleid_%j.out
#SBATCH --error=logs/vit_baseline_vehicleid_%j.err

# VehicleID ViT-Base Baseline Training (Unlimited epochs with early stopping)
echo "=========================================="
echo "Job: VehicleID ViT-Base Baseline (H100)"
echo "Config: EPOCHS=999, Early Stopping: patience=50, min_delta=0.002"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

nvidia-smi

cd /users/sl3753/scratch/GCN_project
source venv_t4/bin/activate

python scripts/training/train_bot_gcn.py \
    --config configs/baseline_configs/vit_baseline_vehicleid.yaml

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
