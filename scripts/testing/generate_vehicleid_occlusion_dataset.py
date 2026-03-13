#!/usr/bin/env python3
"""
VehicleID 遮挡数据集生成脚本
- 读取三档 test split 的 query 列表（800/1600/2400 IDs）
- 对所有 query 图像生成 11 档遮挡版本（0%~30%，步长3%）
- 与 VeRi-776 遮挡协议完全一致（Random Erasing, sh=0.2, mode=random）

VehicleID 目录结构（假设标准版本）:
  VehicleID_V1.0/
  ├── image/                      # 所有图像（221,763张）
  ├── train_test_split/
  │   ├── train_list.txt
  │   ├── test_list_800.txt       # 格式: image_id vehicle_id
  │   ├── test_list_1600.txt
  │   └── test_list_2400.txt
  └── attribute/

输出:
  outputs/occlusion_tests_vehicleid/
  ├── split_800/
  │   ├── query_00pct/  query_03pct/ ... query_30pct/
  ├── split_1600/
  │   └── (同上)
  └── split_2400/
      └── (同上)
"""

import os, random, math
from pathlib import Path
from PIL import Image
import numpy as np

# ── 参数 ───────────────────────────────────────────────────────────────────
DATA_ROOT  = Path("data/dataset/VehicleID_V1.0")
OUTPUT_DIR = Path("outputs/occlusion_tests_vehicleid")
OCC_LEVELS = list(range(0, 31, 3))   # [0,3,6,...,30]
TEST_SPLITS = [800, 1600, 2400]

# Random Erasing 参数（与 VeRi-776 版本完全一致）
RE_P    = 1.0    # 生成数据集时固定应用（不随机跳过）
RE_SL   = 0.02
RE_SH   = 0.20   # 新 RE 参数
RE_R1   = 0.3
RE_MODE = "random"  # 填充模式

SEED = 42  # 固定随机种子，保证可复现


# ── VehicleID test split 解析 ──────────────────────────────────────────────
def parse_test_split(split_size: int):
    """
    解析 test_list_{split_size}.txt
    VehicleID 格式: <image_name> <vehicle_id>
    Query 定义：每个 vehicle_id 取第一张作为 query，其余作为 gallery。
    返回 query 图像路径列表。
    """
    txt_path = DATA_ROOT / "train_test_split" / f"test_list_{split_size}.txt"
    if not txt_path.exists():
        raise FileNotFoundError(f"找不到 {txt_path}")

    # 按 vehicle_id 分组
    from collections import defaultdict
    vid_to_imgs = defaultdict(list)
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_name, vid = parts[0], parts[1]
            vid_to_imgs[vid].append(img_name)

    # 每个 vehicle_id 的第一张作为 query
    query_imgs = []
    for vid, imgs in sorted(vid_to_imgs.items()):
        query_imgs.append(imgs[0])

    return query_imgs


# ── Random Erasing（固定应用）────────────────────────────────────────────
def _apply_patch(result: np.ndarray, x0: int, y0: int, eh: int, ew: int) -> np.ndarray:
    """
    在 result[x0:x0+eh, y0:y0+ew] 上施加随机填充。
    fill 的尺寸严格由切片实际大小决定，彻底避免 broadcast shape mismatch。
    """
    # 取实际切片尺寸（边界截断后可能比 eh/ew 小）
    actual_h = result[x0:x0+eh, :].shape[0]
    actual_w = result[:, y0:y0+ew].shape[1]
    c = result.shape[2]

    if RE_MODE == "random":
        fill = np.random.randint(0, 256, (actual_h, actual_w, c), dtype=np.uint8)
    elif RE_MODE == "zeros":
        fill = np.zeros((actual_h, actual_w, c), dtype=np.uint8)
    else:  # mean
        fill = np.full((actual_h, actual_w, c),
                       result.mean(axis=(0, 1)).astype(np.uint8))

    result[x0:x0+actual_h, y0:y0+actual_w] = fill
    return result


def random_erasing(img_array: np.ndarray, target_ratio: float, rng: random.Random) -> np.ndarray:
    """
    在图像上施加 Random Erasing，使遮挡面积约等于 target_ratio。
    target_ratio = 0 时原样返回。
    """
    if target_ratio == 0:
        return img_array.copy()

    h, w = img_array.shape[:2]
    img_area = h * w
    result = img_array.copy()

    # ── 高遮挡档位（> 15%）：直接用目标面积反推 patch 尺寸 ───────────────────
    if target_ratio > 0.15:
        erase_area = target_ratio * img_area
        aspect = rng.uniform(RE_R1, 1.0 / RE_R1)
        eh = min(int(math.sqrt(erase_area * aspect)), h)
        ew = min(int(math.sqrt(erase_area / aspect)), w)
        eh = max(eh, 1)
        ew = max(ew, 1)
        x0 = rng.randint(0, max(0, h - eh))
        y0 = rng.randint(0, max(0, w - ew))
        return _apply_patch(result, x0, y0, eh, ew)

    # ── 低遮挡档位（<= 15%）：标准 RE 随机采样 ──────────────────────────────
    for _ in range(100):
        sl = RE_SL
        sh = min(RE_SH, target_ratio * 1.5)
        erase_area = rng.uniform(sl, sh) * img_area
        aspect = rng.uniform(RE_R1, 1.0 / RE_R1)

        eh = int(math.sqrt(erase_area * aspect))
        ew = int(math.sqrt(erase_area / aspect))

        if eh < 1 or ew < 1 or eh > h or ew > w:
            continue

        actual_ratio = (eh * ew) / img_area
        if abs(actual_ratio - target_ratio) > target_ratio * 0.3:
            continue

        x0 = rng.randint(0, max(0, h - eh))
        y0 = rng.randint(0, max(0, w - ew))
        return _apply_patch(result, x0, y0, eh, ew)

    # 兜底：目标面积正方形（低档位不应触发，保险起见保留）
    erase_area = target_ratio * img_area
    eh = max(1, min(int(math.sqrt(erase_area)), h))
    ew = max(1, min(int(math.sqrt(erase_area)), w))
    x0 = rng.randint(0, max(0, h - eh))
    y0 = rng.randint(0, max(0, w - ew))
    return _apply_patch(result, x0, y0, eh, ew)


# ── 主函数 ─────────────────────────────────────────────────────────────────
def main():
    rng = random.Random(SEED)
    np.random.seed(SEED)

    total_generated = 0

    for split in TEST_SPLITS:
        print(f"\n{'='*60}")
        print(f"处理 test split = {split}")

        query_imgs = parse_test_split(split)
        print(f"  Query 数量: {len(query_imgs)}")

        for occ in OCC_LEVELS:
            occ_ratio = occ / 100.0
            out_dir = OUTPUT_DIR / f"split_{split}" / f"query_{occ:02d}pct"
            out_dir.mkdir(parents=True, exist_ok=True)

            count = 0
            for img_name in query_imgs:
                # VehicleID 图像可能存储为 image/<id>.jpg
                img_path = DATA_ROOT / "image" / f"{img_name}.jpg"
                if not img_path.exists():
                    img_path = DATA_ROOT / "image" / img_name
                if not img_path.exists():
                    print(f"  ⚠️  找不到图像: {img_name}")
                    continue

                img = Image.open(img_path).convert("RGB")
                arr = np.array(img)

                if occ_ratio > 0:
                    arr = random_erasing(arr, occ_ratio, rng)

                out_path = out_dir / f"{img_name}.jpg"
                Image.fromarray(arr).save(out_path, quality=95)
                count += 1

            total_generated += count
            print(f"  occ={occ:2d}%  生成: {count} 张 → {out_dir}")

    print(f"\n✅ 全部完成，共生成 {total_generated} 张遮挡图像")
    print(f"   输出目录: {OUTPUT_DIR}/")
    print(f"   目录结构: split_800/1600/2400 × query_00pct~query_30pct")


if __name__ == "__main__":
    main()
