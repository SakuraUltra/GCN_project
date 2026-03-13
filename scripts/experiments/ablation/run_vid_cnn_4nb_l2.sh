#!/bin/bash
#SBATCH --job-name=vid_cnn_4nb_l2
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/vid_cnn_4nb_l2_%j.out
#SBATCH --error=logs/vid_cnn_4nb_l2_%j.err

# VehicleID CNN+GCN 4nb L2
echo "=========================================="
echo "Job: VehicleID CNN+GCN 4nb L2"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

cd /users/sl3753/scratch/GCN_project
source venv_t4/bin/activate

python scripts/training/train_bot_gcn.py \
    --config configs/gcn_transformer_configs/abl_vehicleid_cnn_gcn_4nb_l2.yaml

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
