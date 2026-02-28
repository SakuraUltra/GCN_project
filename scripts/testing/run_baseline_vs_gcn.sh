#!/bin/bash
# 运行Baseline vs GCN对比评估

source venv_t4/bin/activate

OCCLUSION_DIR="outputs/occlusion_tests/veri776_query"
GALLERY_DIR="data/dataset/776_DataSet/image_test"
BASELINE_MODEL="outputs/bot_baseline_1_1/776/baseline_run_01/best_model.pth"
GCN_MODEL="outputs/bot_gcn_776_v2/best_model.pth"

echo "=================================================="
echo "Robustness Comparison: Baseline vs GCN"
echo "=================================================="

# 评估Baseline
echo ""
echo "Step 1/2: Evaluating Baseline model..."
python scripts/testing/evaluate_occlusion_robustness.py \
    --model $BASELINE_MODEL \
    --occlusion-dir $OCCLUSION_DIR \
    --gallery-dir $GALLERY_DIR \
    --output-dir outputs/robustness_comparison/baseline \
    --device cuda

# 评估GCN
echo ""
echo "Step 2/2: Evaluating GCN model..."
python scripts/testing/evaluate_occlusion_robustness.py \
    --model $GCN_MODEL \
    --occlusion-dir $OCCLUSION_DIR \
    --gallery-dir $GALLERY_DIR \
    --output-dir outputs/robustness_comparison/gcn \
    --device cuda

echo ""
echo "✅ Both evaluations completed!"
echo "Baseline results: outputs/robustness_comparison/baseline/"
echo "GCN results: outputs/robustness_comparison/gcn/"
