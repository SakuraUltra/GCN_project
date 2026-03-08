#!/bin/bash
#SBATCH --job-name=occ_eval_all
#SBATCH --output=logs/occlusion_eval_%j.out
#SBATCH --error=logs/occlusion_eval_%j.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

echo "=================================================="
echo "Batch Occlusion Robustness Evaluation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=================================================="

# 进入项目目录
cd /users/sl3753/scratch/GCN_project || exit 1

# 激活虚拟环境
source venv_t4/bin/activate

# 打印环境信息
echo ""
echo "Environment Info:"
python --version
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "GPU: $CUDA_VISIBLE_DEVICES"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi
echo ""

# 运行批量评估脚本
bash scripts/testing/run_occlusion_evaluation.sh

EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=================================================="

exit $EXIT_CODE
