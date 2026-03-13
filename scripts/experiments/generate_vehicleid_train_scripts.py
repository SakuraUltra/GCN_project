#!/usr/bin/env python3
"""
批量生成 VehicleID 独立训练脚本（8个）
"""

CONFIGS = [
    ("vid_cnn_4nb_l1", "abl_vehicleid_cnn_gcn_4nb_l1.yaml", "VehicleID CNN+GCN 4nb L1"),
    ("vid_cnn_4nb_l2", "abl_vehicleid_cnn_gcn_4nb_l2.yaml", "VehicleID CNN+GCN 4nb L2"),
    ("vid_cnn_4nb_l3", "abl_vehicleid_cnn_gcn_4nb_l3.yaml", "VehicleID CNN+GCN 4nb L3"),
    ("vid_cnn_knn_l1", "abl_vehicleid_cnn_gcn_knn_l1.yaml", "VehicleID CNN+GCN kNN L1"),
    ("vid_vit_4nb_l1", "abl_vehicleid_vit_gcn_4nb_l1.yaml", "VehicleID ViT+GCN 4nb L1"),
    ("vid_vit_4nb_l2", "abl_vehicleid_vit_gcn_4nb_l2.yaml", "VehicleID ViT+GCN 4nb L2"),
    ("vid_vit_4nb_l3", "abl_vehicleid_vit_gcn_4nb_l3.yaml", "VehicleID ViT+GCN 4nb L3"),
    ("vid_vit_knn_l1", "abl_vehicleid_vit_gcn_knn_l1.yaml", "VehicleID ViT+GCN kNN L1"),
]

TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err

# {description}
echo "=========================================="
echo "Job: {description}"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "=========================================="

cd /users/sl3753/scratch/GCN_project
source venv_t4/bin/activate

python scripts/training/train_bot_gcn.py \\
    --config configs/gcn_transformer_configs/{config_file}

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
"""

import os

output_dir = "scripts/experiments/ablation"
os.makedirs(output_dir, exist_ok=True)

print("🚀 生成 VehicleID 训练脚本...")
for job_name, config_file, description in CONFIGS:
    script_content = TEMPLATE.format(
        job_name=job_name,
        config_file=config_file,
        description=description
    )
    
    script_path = os.path.join(output_dir, f"run_{job_name}.sh")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # 设置执行权限
    os.chmod(script_path, 0o755)
    
    print(f"✅ {script_path}")

print(f"\n✨ 完成！共生成 {len(CONFIGS)} 个训练脚本")
