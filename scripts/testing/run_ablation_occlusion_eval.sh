#!/bin/bash
#SBATCH --job-name=abl_occ_eval
#SBATCH --output=logs/ablation_occlusion_eval_%j.out
#SBATCH --error=logs/ablation_occlusion_eval_%j.err
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sl3753@york.ac.uk

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# 激活虚拟环境
source venv_t4/bin/activate

# 进入项目目录
cd /users/sl3753/scratch/GCN_project

# 打印环境信息
echo ""
echo "Environment Info:"
echo "Python: $(which python3)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

echo ""
echo "=========================================="
echo "Starting Ablation Occlusion Evaluation"
echo "=========================================="

# 运行评估脚本
python3 scripts/testing/evaluate_occlusion_abl19.py \
    --occ_root outputs/occlusion_tests_v2 \
    --output_dir outputs/ablation_occlusion_results

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Job Finished"
echo "End Time: $(date)"
echo "Exit Code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Evaluation completed successfully!"
    echo ""
    echo "Results saved to: outputs/ablation_occlusion_results/"
    ls -lh outputs/ablation_occlusion_results/
else
    echo "❌ Evaluation failed with exit code $EXIT_CODE"
fi

exit $EXIT_CODE
