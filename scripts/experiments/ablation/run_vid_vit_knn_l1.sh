#!/bin/bash
#SBATCH --job-name=vid_vit_knn_l1
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/vid_vit_knn_l1_%j.out
#SBATCH --error=logs/vid_vit_knn_l1_%j.err

# VehicleID ViT+GCN kNN L1
echo "=========================================="
echo "Job: VehicleID ViT+GCN kNN L1"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

cd /users/sl3753/scratch/GCN_project
source venv_t4/bin/activate

python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_vehicleid_vit_gcn_knn_l1.yaml

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
