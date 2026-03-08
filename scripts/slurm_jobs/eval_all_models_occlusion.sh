#!/bin/bash
#SBATCH --job-name=eval_all_occ
#SBATCH --output=logs/eval_all_models_%j.out
#SBATCH --error=logs/eval_all_models_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

echo "=================================================="
echo "Batch Occlusion Evaluation for All 10 Models"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "=================================================="

# 激活虚拟环境
source venv_t4/bin/activate

# 打印环境信息
echo ""
echo "Environment Info:"
python --version
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# 运行批量评估
echo ""
echo "Starting batch evaluation..."
echo "Models: 10"
echo "Occlusion levels: 11 (0%, 3%, 6%, ..., 30%)"
echo "Total evaluations: 110"
echo ""

python scripts/testing/evaluate_all_models_occlusion.py \
    --gpu 0 \
    --batch-size 64 \
    --num-workers 8

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Job finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=================================================="

exit $EXIT_CODE
