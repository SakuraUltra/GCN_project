#!/bin/bash
#SBATCH --job-name=occ_eval
#SBATCH --partition=gpuplus
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/occlusion_evaluation_%j.out
#SBATCH --error=logs/occlusion_evaluation_%j.err

# 遮挡鲁棒性批量评估脚本
# 协议：Random Erasing (Zhong et al., 2020)
# 遮挡等级：0%, 3%, 6%, 9%, 12%, 15%, 18%, 21%, 24%, 27%, 30%

set -e  # Exit on error

# 激活虚拟环境
source venv_t4/bin/activate

# 遮挡级别
OCCLUSION_LEVELS=(0 3 6 9 12 15 18 21 24 27 30)
OUTPUT_BASE="outputs/occlusion_results_v2"
GALLERY_DIR="data/dataset/776_DataSet/image_test"

# 定义10个模型（模型名称 模型路径 模型类型）
declare -a MODELS=(
  # CNN系列（无RE）
  "resnet_baseline|outputs/bot_baseline_1_1/776/baseline_run_01/best_model.pth|resnet_baseline"
  "resnet_gcn_nore|outputs/bot_gcn_776_v2/best_model.pth|resnet_gcn"
  # CNN系列（旧RE 0.02-0.33）
  "resnet_gcn_re_old|outputs/bot_gcn_776_Random/best_model.pth|resnet_gcn"
  # ViT系列（无RE）
  "vit_baseline_nore|outputs/bot_vitbase_baseline_nore_776/best_model.pth|vit_baseline"
  "vit_native768_nore|outputs/bot_vitbase_native768_nore_776/best_model.pth|vit_native768"
  # ViT系列（旧RE 0.02-0.33）
  "vit_baseline_re_old|outputs/bot_vitbase_baseline_776/best_model.pth|vit_baseline"
  "vit_gcn_re_old|outputs/bot_vitbase_gcn_776/best_model.pth|vit_gcn"
  # 新RE系列（0.02-0.2）
  "resnet_gcn_re_new|outputs/new_re/resnet_gcn_re/best_model.pth|resnet_gcn"
  "vit_baseline_re_new|outputs/new_re/vitbase_baseline_re/best_model.pth|vit_baseline"
  "vit_native768_gcn_re_new|outputs/new_re/vitbase_native768_gcn_re/best_model.pth|vit_gcn"
)

echo "=================================================="
echo "批量遮挡鲁棒性评估"
echo "=================================================="
echo "模型数量: ${#MODELS[@]}"
echo "遮挡级别: ${#OCCLUSION_LEVELS[@]} (0%-30%)"
echo "总测试数: $((${#MODELS[@]} * ${#OCCLUSION_LEVELS[@]}))"
echo "开始时间: $(date)"
echo "=================================================="

# 创建输出目录
mkdir -p "$OUTPUT_BASE"

# 统计变量
TOTAL_TESTS=$((${#MODELS[@]} * ${#OCCLUSION_LEVELS[@]}))
CURRENT_TEST=0
FAILED_TESTS=0

# 遍历所有模型
for MODEL_INFO in "${MODELS[@]}"; do
  # 解析模型信息（用|分隔）
  IFS='|' read -r MODEL_NAME MODEL_PATH MODEL_TYPE <<< "$MODEL_INFO"
  
  echo ""
  echo "=================================================="
  echo "Testing Model: $MODEL_NAME"
  echo "Type: $MODEL_TYPE"
  echo "Path: $MODEL_PATH"
  echo "=================================================="
  
  # 检查模型文件是否存在
  if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Error: Model file not found: $MODEL_PATH"
    FAILED_TESTS=$((FAILED_TESTS + ${#OCCLUSION_LEVELS[@]}))
    continue
  fi
  
  MODEL_OUT="$OUTPUT_BASE/$MODEL_NAME"
  mkdir -p "$MODEL_OUT"
  
  # 遍历所有遮挡级别
  for LEVEL in "${OCCLUSION_LEVELS[@]}"; do
    CURRENT_TEST=$((CURRENT_TEST + 1))
    
    QUERY_DIR="outputs/occlusion_tests_v2/query_$(printf '%02d' $LEVEL)pct"
    LEVEL_OUT="$MODEL_OUT/level_${LEVEL}pct"
    
    echo ""
    echo "[$CURRENT_TEST/$TOTAL_TESTS] Testing ${MODEL_NAME} at ${LEVEL}% occlusion..."
    
    # 检查遮挡数据集是否存在
    if [ ! -d "$QUERY_DIR" ]; then
      echo "❌ Error: Query directory not found: $QUERY_DIR"
      FAILED_TESTS=$((FAILED_TESTS + 1))
      continue
    fi
    
    # 创建输出目录（tee需要目录先存在）
    mkdir -p "$LEVEL_OUT"
    
    # 运行测试
    if python scripts/testing/test_single_model.py \
      --model-path "$MODEL_PATH" \
      --model-type "$MODEL_TYPE" \
      --query-dir "$QUERY_DIR" \
      --gallery-dir "$GALLERY_DIR" \
      --output-dir "$LEVEL_OUT" \
      --batch-size 64 \
      --num-workers 8 \
      2>&1 | tee "$LEVEL_OUT/test.log"; then
      
      echo "✅ Success: $MODEL_NAME @ ${LEVEL}%"
    else
      echo "❌ Failed: $MODEL_NAME @ ${LEVEL}%"
      FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
  done
  
  echo ""
  echo "=== $MODEL_NAME 完成 ==="
done

# 生成汇总报告
echo ""
echo "=================================================="
echo "生成汇总报告..."
echo "=================================================="

SUMMARY_FILE="$OUTPUT_BASE/summary_report.csv"
echo "model_name,occlusion_level,mAP,rank1,rank5,rank10" > "$SUMMARY_FILE"

for MODEL_INFO in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_NAME MODEL_PATH MODEL_TYPE <<< "$MODEL_INFO"
  
  for LEVEL in "${OCCLUSION_LEVELS[@]}"; do
    RESULT_FILE="$OUTPUT_BASE/$MODEL_NAME/level_${LEVEL}pct/results.json"
    
    if [ -f "$RESULT_FILE" ]; then
      # 使用Python解析JSON
      METRICS=$(python3 -c "
import json
with open('$RESULT_FILE') as f:
    data = json.load(f)
    print(f\"{data['mAP']:.4f},{data['rank1']:.4f},{data['rank5']:.4f},{data['rank10']:.4f}\")
")
      echo "$MODEL_NAME,$LEVEL,$METRICS" >> "$SUMMARY_FILE"
    fi
  done
done

echo "✅ 汇总报告已保存: $SUMMARY_FILE"

# 最终统计
echo ""
echo "=================================================="
echo "批量评估完成"
echo "=================================================="
echo "总测试数: $TOTAL_TESTS"
echo "成功: $((TOTAL_TESTS - FAILED_TESTS))"
echo "失败: $FAILED_TESTS"
echo "结束时间: $(date)"
echo "输出目录: $OUTPUT_BASE"
echo "=================================================="

# 显示前10行汇总
echo ""
echo "汇总预览 (前10行):"
head -11 "$SUMMARY_FILE" | column -t -s,

exit 0
