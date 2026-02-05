#!/bin/bash

# run_grid_ablation.sh
# 批量运行不同网格尺寸的实验脚本

# 显式退出，如果有任何指令失败
set -e

# 进入项目根目录以确保相对路径正确
cd /root/autodl-tmp/gcn_project

# 定义要测试的网格尺寸列表
# 格式: "H W"
SCALES=(
    # "4 2"  <-- 已跳过
    # "8 4"
    # "12 6"
    "16 8"
    "24 12"
    "32 16"
)

DATA_ROOT="data/VehicleID_V1.0"

echo "========================================================"
echo "🚀 Starting Grid Scale Ablation Experiments"
echo "========================================================"

for scale in "${SCALES[@]}"; do
    # 解析 H 和 W
    read -r H W <<< "$scale"
    
    echo ""
    echo "--------------------------------------------------------"
    echo "▶️  Running experiment for Grid Size: ${H}x${W}"
    echo "--------------------------------------------------------"
    
    # 运行训练脚本
    # 注意：我们假设 train_bot_baseline.py 已经修改为支持 --grid_h 和 --grid_w
    # 输出目录会自动按 outputs/grid_scale/${H}x${W}/... 结构生成
    python /root/autodl-tmp/gcn_project/scripts/train_bot_baseline.py \
        --data_root "$DATA_ROOT" \
        --grid_h "$H" \
        --grid_w "$W"

    echo "✅ Finished ${H}x${W}"
done

echo ""
echo "========================================================"
echo "🎉 All Grid Scale Experiments Completed!"
echo "========================================================"
