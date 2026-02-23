#!/bin/bash
# 批量运行 BoT-GCN 消融实验 (VeRi-776)
# 
# 消融因素:
# 1. Baseline vs GCN
# 2. Pooling: mean vs max vs attention
# 3. Fusion: concat vs gated
# 4. Grid Size: 4x4 vs 8x8

echo "=========================================="
echo "BoT-GCN 消融实验 - VeRi-776"
echo "=========================================="

CONFIG="configs/gcn_transformer_configs/bot_gcn_776.yaml"

# 1. Baseline (No GCN)
echo ""
echo "1/9: Baseline (No GCN)"
sbatch scripts/slurm_jobs/submit_train_bot_gcn_776.sh \
    --use_gcn false \
    --output_dir outputs/ablation_776/baseline_no_gcn

# 2. GCN + Mean Pooling + Concat Fusion (4x4)
echo "2/9: GCN + Mean + Concat (4x4)"
sbatch scripts/slurm_jobs/submit_train_bot_gcn_776.sh \
    --pooling_type mean \
    --fusion_type concat \
    --grid_h 4 --grid_w 4 \
    --output_dir outputs/ablation_776/gcn_mean_concat_4x4

# 3. GCN + Max Pooling + Concat Fusion (4x4)
echo "3/9: GCN + Max + Concat (4x4)"
sbatch scripts/slurm_jobs/submit_train_bot_gcn_776.sh \
    --pooling_type max \
    --fusion_type concat \
    --grid_h 4 --grid_w 4 \
    --output_dir outputs/ablation_776/gcn_max_concat_4x4

# 4. GCN + Attention Pooling + Concat Fusion (4x4)
echo "4/9: GCN + Attention + Concat (4x4)"
sbatch scripts/slurm_jobs/submit_train_bot_gcn_776.sh \
    --pooling_type attention \
    --fusion_type concat \
    --grid_h 4 --grid_w 4 \
    --output_dir outputs/ablation_776/gcn_attention_concat_4x4

# 5. GCN + Mean Pooling + Gated Fusion (4x4)
echo "5/9: GCN + Mean + Gated (4x4)"
sbatch scripts/slurm_jobs/submit_train_bot_gcn_776.sh \
    --pooling_type mean \
    --fusion_type gated \
    --grid_h 4 --grid_w 4 \
    --output_dir outputs/ablation_776/gcn_mean_gated_4x4

# 6. GCN + Mean Pooling + Concat Fusion (8x8)
echo "6/9: GCN + Mean + Concat (8x8)"
sbatch scripts/slurm_jobs/submit_train_bot_gcn_776.sh \
    --pooling_type mean \
    --fusion_type concat \
    --grid_h 8 --grid_w 8 \
    --output_dir outputs/ablation_776/gcn_mean_concat_8x8

# 7. GCN + Add Fusion (4x4) - 简单相加
echo "7/9: GCN + Add Fusion (4x4)"
sbatch scripts/slurm_jobs/submit_train_bot_gcn_776.sh \
    --pooling_type mean \
    --fusion_type add \
    --grid_h 4 --grid_w 4 \
    --output_dir outputs/ablation_776/gcn_mean_add_4x4

# 8. GCN + No Fusion (仅图嵌入, 4x4)
echo "8/9: GCN + No Fusion (Graph Only, 4x4)"
sbatch scripts/slurm_jobs/submit_train_bot_gcn_776.sh \
    --pooling_type mean \
    --fusion_type none \
    --grid_h 4 --grid_w 4 \
    --output_dir outputs/ablation_776/gcn_mean_none_4x4

# 9. GCN + 2 Layers (vs 1 layer baseline)
echo "9/9: GCN 2 Layers + Mean + Concat (4x4)"
sbatch scripts/slurm_jobs/submit_train_bot_gcn_776.sh \
    --pooling_type mean \
    --fusion_type concat \
    --gcn_num_layers 2 \
    --grid_h 4 --grid_w 4 \
    --output_dir outputs/ablation_776/gcn_2layers_mean_concat_4x4

echo ""
echo "=========================================="
echo "✓ 已提交 9 个消融实验任务"
echo "=========================================="
echo ""
echo "监控任务: squeue -u $USER"
echo "查看结果: ls -lh outputs/ablation_776/"
