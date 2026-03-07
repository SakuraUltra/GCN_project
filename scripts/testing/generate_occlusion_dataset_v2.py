"""
遮挡数据集生成脚本 v2
协议：Random Erasing (Zhong et al., 2020) Re-ID标准

参数：
- 遮挡等级：0%, 3%, 6%, 9%, 12%, 15%, 18%, 21%, 24%, 27%, 30%（共11级）
- 面积比：固定（按每张图自身 W×H 计算）
- 位置：随机
- 长宽比：(0.3, 3.3) 随机采样
- 填充值：随机像素 [0, 255]（RE-R模式）
- 每张图遮挡一次
- 0% 等级：直接字节复制原始图片，不经任何PIL处理

输入：原始 VeRi-776 query 目录
输出：outputs/occlusion_tests_v2/query_00pct/
      outputs/occlusion_tests_v2/query_03pct/
      ...
      outputs/occlusion_tests_v2/query_30pct/
"""

import os
import math
import random
import shutil
import numpy as np
from PIL import Image

OCCLUSION_LEVELS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

def apply_occlusion(img_path, level):
    """对单张图片应用固定面积比遮挡，随机位置，随机像素填充"""
    img = Image.open(img_path).convert('RGB')
    W, H = img.size
    
    if level == 0:
        return img, None  # 0% 直接返回原图
    
    target_area = W * H * (level / 100.0)
    
    # 多次尝试找到合适的遮挡框
    max_attempts = 50
    best_h, best_w = None, None
    best_error = float('inf')
    
    for _ in range(max_attempts):
        # 随机长宽比 (0.3, 3.3)
        ratio = random.uniform(0.3, 3.3)
        h = int(round(math.sqrt(target_area * ratio)))
        w = int(round(math.sqrt(target_area / ratio)))
        
        # 必须在边界内
        if h > H or w > W:
            continue
        
        # 计算误差
        actual_area = h * w
        error = abs(actual_area - target_area) / target_area
        
        # 如果误差 < 1%，直接使用
        if error < 0.01:
            best_h, best_w = h, w
            break
        
        # 记录最佳结果
        if error < best_error:
            best_error = error
            best_h, best_w = h, w
    
    # 使用最佳结果（如果50次都失败，使用误差最小的）
    if best_h is not None and best_w is not None:
        h, w = best_h, best_w
    else:
        # 极端情况：使用正方形遮挡
        side = int(round(math.sqrt(target_area)))
        h = min(side, H)
        w = min(side, W)
    
    # 随机位置
    x = random.randint(0, W - w) if W > w else 0
    y = random.randint(0, H - h) if H > h else 0
    
    # 随机像素填充 RE-R
    img_array = np.array(img)
    img_array[y:y+h, x:x+w] = np.random.randint(
        0, 256, (h, w, 3), dtype=np.uint8
    )
    
    # 返回遮挡信息用于验证
    occlusion_info = {
        'x': x, 'y': y, 'w': w, 'h': h,
        'actual_ratio': (h * w) / (W * H) * 100,
        'target_ratio': level
    }
    
    return Image.fromarray(img_array), occlusion_info

def generate_dataset(src_dir, out_base):
    import json
    all_stats = []  # 记录所有遮挡统计信息
    
    for level in OCCLUSION_LEVELS:
        out_dir = os.path.join(out_base, f'query_{level:02d}pct')
        os.makedirs(out_dir, exist_ok=True)
        
        img_files = [f for f in os.listdir(src_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        level_ratios = []
        
        for fname in img_files:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(out_dir, fname)
            
            if level == 0:
                shutil.copy2(src_path, dst_path)  # 字节级复制，不经PIL
            else:
                img, info = apply_occlusion(src_path, level)
                img.save(dst_path, quality=95)
                if info:
                    level_ratios.append(info['actual_ratio'])
        
        # 统计本级别
        if level_ratios:
            avg_ratio = np.mean(level_ratios)
            std_ratio = np.std(level_ratios)
            all_stats.append({
                'target': level,
                'actual_mean': float(avg_ratio),
                'actual_std': float(std_ratio),
                'count': len(level_ratios)
            })
        
        print(f"Level {level:2d}%: {len(img_files)} images → {out_dir}")
        if level_ratios:
            print(f"         实际遮挡: {avg_ratio:.2f}% ± {std_ratio:.2f}%")
    
    # 保存统计信息
    stats_file = os.path.join(out_base, 'occlusion_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n统计信息已保存到: {stats_file}")

if __name__ == '__main__':
    SRC = 'data/dataset/776_DataSet/image_query'  # VeRi-776 原始query目录
    OUT = 'outputs/occlusion_tests_v2'
    generate_dataset(SRC, OUT)
    print("=== 遮挡数据集生成完成 ===")
