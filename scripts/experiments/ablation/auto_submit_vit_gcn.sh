#!/bin/bash

# Automatic submission script for VehicleID ViT+GCN ablation experiments
# Monitors baseline training completion and automatically submits 4 GCN experiments

BASELINE_OUTPUT_DIR="outputs/bot_baseline_1_1/VehicleID/vit_baseline_run_01"
BASELINE_CHECKPOINT="${BASELINE_OUTPUT_DIR}/best_model.pth"
BASELINE_LOG="${BASELINE_OUTPUT_DIR}/training.log"
CHECK_INTERVAL=1800  # Check every 30 minutes

echo "=========================================="
echo "Auto-Submission Script for ViT+GCN Ablation"
echo "=========================================="
echo "Baseline checkpoint: ${BASELINE_CHECKPOINT}"
echo "Check interval: ${CHECK_INTERVAL} seconds (30 minutes)"
echo "Start monitoring at: $(date)"
echo "=========================================="

# Function to check if training is complete
check_training_complete() {
    # Check if best_model.pth exists
    if [ ! -f "${BASELINE_CHECKPOINT}" ]; then
        return 1
    fi
    
    # Check if training log contains completion marker
    if [ -f "${BASELINE_LOG}" ]; then
        if grep -q "Training session completed" "${BASELINE_LOG}" || \
           grep -q "Early stopping triggered" "${BASELINE_LOG}"; then
            return 0
        fi
    fi
    
    return 1
}

# Function to verify checkpoint is valid
verify_checkpoint() {
    python3 -c "
import torch
import sys
try:
    ckpt = torch.load('${BASELINE_CHECKPOINT}', map_location='cpu', weights_only=False)
    sd = ckpt.get('state_dict', ckpt.get('model', ckpt))
    
    # Check for ViT-Base cls_token dimension
    if 'backbone.vit.cls_token' in sd:
        cls_token_shape = sd['backbone.vit.cls_token'].shape
        if cls_token_shape != torch.Size([1, 1, 768]):
            print(f'ERROR: cls_token shape {cls_token_shape}, expected [1, 1, 768]')
            sys.exit(1)
    
    # Check total parameters
    total_params = sum(p.numel() for p in sd.values())
    if total_params < 100_000_000:  # Should be ~115M
        print(f'ERROR: Only {total_params} parameters, expected >100M')
        sys.exit(1)
    
    print(f'✓ Checkpoint validated: {total_params:,} parameters')
    sys.exit(0)
except Exception as e:
    print(f'ERROR: {e}')
    sys.exit(1)
"
    return $?
}

# Function to submit all 4 GCN experiments
submit_experiments() {
    echo ""
    echo "=========================================="
    echo "Submitting 4 ViT+GCN Ablation Experiments"
    echo "=========================================="
    
    cd /users/sl3753/scratch/GCN_project
    
    # Submit 4-neighbor L1
    echo "Submitting: ViT+GCN 4-neighbor L1..."
    sbatch scripts/experiments/ablation/run_vehicleid_vit_4nb_l1.sh
    sleep 2
    
    # Submit 4-neighbor L2
    echo "Submitting: ViT+GCN 4-neighbor L2..."
    sbatch scripts/experiments/ablation/run_vehicleid_vit_4nb_l2.sh
    sleep 2
    
    # Submit 4-neighbor L3
    echo "Submitting: ViT+GCN 4-neighbor L3..."
    sbatch scripts/experiments/ablation/run_vehicleid_vit_4nb_l3.sh
    sleep 2
    
    # Submit kNN L1
    echo "Submitting: ViT+GCN kNN L1..."
    sbatch scripts/experiments/ablation/run_vehicleid_vit_knn_l1.sh
    
    echo ""
    echo "=========================================="
    echo "All 4 experiments submitted at: $(date)"
    echo "=========================================="
    
    # Show queue status
    sleep 2
    echo ""
    echo "Current job queue:"
    squeue -u sl3753
}

# Main monitoring loop
while true; do
    if check_training_complete; then
        echo ""
        echo "✓ Baseline training completed at: $(date)"
        echo "Verifying checkpoint..."
        
        if verify_checkpoint; then
            echo "✓ Checkpoint verified successfully"
            submit_experiments
            
            echo ""
            echo "=========================================="
            echo "Auto-submission completed successfully!"
            echo "=========================================="
            exit 0
        else
            echo "✗ Checkpoint verification failed!"
            echo "Please check the checkpoint manually."
            exit 1
        fi
    else
        echo "[$(date)] Baseline training not complete yet, checking again in 30 minutes..."
        sleep ${CHECK_INTERVAL}
    fi
done
