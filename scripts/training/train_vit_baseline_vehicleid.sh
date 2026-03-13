#!/bin/bash
#SBATCH --job-name=vit_baseline_vehicleid
#SBATCH --output=logs/vit_baseline_vehicleid_%j.out
#SBATCH --error=logs/vit_baseline_vehicleid_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=gpuplus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

# Navigate to project directory
cd /users/sl3753/scratch/GCN_project
source venv_t4/bin/activate

# Training command
echo "=========================================="
echo "Starting ViT Baseline Training for VehicleID"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=========================================="

python3 scripts/training/train_bot_gcn.py \
    --config configs/baseline_configs/vit_baseline_vehicleid.yaml

echo "=========================================="
echo "Training Completed"
echo "End Time: $(date)"
echo "=========================================="
