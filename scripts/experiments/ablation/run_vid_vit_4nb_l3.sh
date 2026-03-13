#!/bin/bash
#SBATCH --job-name=vid_vit_4nb_l3
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/vid_vit_4nb_l3_%j.out
#SBATCH --error=logs/vid_vit_4nb_l3_%j.err

# VehicleID ViT+GCN 4nb L3
echo "=========================================="
echo "Job: VehicleID ViT+GCN 4nb L3"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

cd /users/sl3753/scratch/GCN_project
source venv_t4/bin/activate

python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_vehicleid_vit_gcn_4nb_l3.yaml

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
