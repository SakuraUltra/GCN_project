#!/bin/bash
# 检查遮挡评估中失败的测试

OUTPUT_BASE="outputs/occlusion_results_v2"

echo "=================================================="
echo "检查遮挡评估结果"
echo "=================================================="
echo ""

# 统计变量
TOTAL_TESTS=0
SUCCESS_TESTS=0
FAILED_TESTS=0

# 存储失败的测试
declare -a FAILED_LIST

# 遍历所有模型目录
for MODEL_DIR in "$OUTPUT_BASE"/*; do
    if [ ! -d "$MODEL_DIR" ]; then
        continue
    fi
    
    MODEL_NAME=$(basename "$MODEL_DIR")
    
    # 遍历所有遮挡级别
    for LEVEL_DIR in "$MODEL_DIR"/level_*; do
        if [ ! -d "$LEVEL_DIR" ]; then
            continue
        fi
        
        LEVEL=$(basename "$LEVEL_DIR")
        TOTAL_TESTS=$((TOTAL_TESTS + 1))
        
        # 检查是否有 results.json
        if [ -f "$LEVEL_DIR/results.json" ]; then
            # 检查 JSON 是否完整（有 mAP 字段）
            if grep -q '"mAP"' "$LEVEL_DIR/results.json"; then
                SUCCESS_TESTS=$((SUCCESS_TESTS + 1))
            else
                FAILED_TESTS=$((FAILED_TESTS + 1))
                FAILED_LIST+=("$MODEL_NAME|$LEVEL")
                echo "❌ $MODEL_NAME @ $LEVEL - 结果不完整"
            fi
        else
            FAILED_TESTS=$((FAILED_TESTS + 1))
            FAILED_LIST+=("$MODEL_NAME|$LEVEL")
            
            # 检查错误日志
            if [ -f "$LEVEL_DIR/test.log" ]; then
                ERROR_LINE=$(grep -n "Error\|Traceback\|RuntimeError" "$LEVEL_DIR/test.log" | head -1)
                if [ -n "$ERROR_LINE" ]; then
                    echo "❌ $MODEL_NAME @ $LEVEL - 错误: $(echo $ERROR_LINE | cut -d: -f2- | head -c 80)"
                else
                    echo "❌ $MODEL_NAME @ $LEVEL - 未生成结果"
                fi
            else
                echo "❌ $MODEL_NAME @ $LEVEL - 未运行"
            fi
        fi
    done
done

echo ""
echo "=================================================="
echo "统计结果"
echo "=================================================="
echo "总测试数: $TOTAL_TESTS"
echo "成功: $SUCCESS_TESTS ($(( SUCCESS_TESTS * 100 / (TOTAL_TESTS > 0 ? TOTAL_TESTS : 1) ))%)"
echo "失败: $FAILED_TESTS ($(( FAILED_TESTS * 100 / (TOTAL_TESTS > 0 ? TOTAL_TESTS : 1) ))%)"
echo ""

if [ $FAILED_TESTS -gt 0 ]; then
    echo "=================================================="
    echo "失败的测试列表 (可用于重新运行)"
    echo "=================================================="
    for FAILED in "${FAILED_LIST[@]}"; do
        echo "$FAILED"
    done
    echo ""
    
    # 保存到文件
    FAILED_FILE="$OUTPUT_BASE/failed_tests.txt"
    printf "%s\n" "${FAILED_LIST[@]}" > "$FAILED_FILE"
    echo "失败列表已保存到: $FAILED_FILE"
fi

echo ""
echo "完成时间: $(date)"
