#!/usr/bin/env python3
"""快速测试单个遮挡配置的评估"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

import torch
from models.bot_baseline.bot_model import BoTBaseline
from models.bot_baseline.veri_dataset import VeRiDataset, build_transforms
from eval.evaluator import ReIDEvaluator
from PIL import Image
import json

# 简单的OcclusionDataset
class SimpleOcclusionDataset(torch.utils.data.Dataset):
    def __init__(self, occlusion_dir, metadata_path, transform=None):
        self.occlusion_dir = Path(occlusion_dir)
        self.transform = transform
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.data = []
        for img_meta in self.metadata['images'][:100]:  # 只测试前100张
            img_path = self.occlusion_dir / Path(img_meta['relative_path'])
            if img_path.exists():
                filename = Path(img_meta['original_path']).stem
                parts = filename.split('_')
                if len(parts) >= 1:
                    pid = int(parts[0])
                    camid = 0
                    self.data.append((str(img_path), pid, camid))
        
        print(f"Loaded {len(self.data)} images (sample)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, pid, camid = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, pid, camid

print("="*80)
print("Quick Test: Single Occlusion Configuration")
print("="*80)

# 加载Baseline模型
print("\n1. Loading Baseline model...")
model_path = "outputs/bot_baseline_1_1/776/baseline_run_01/best_model.pth"
checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)

if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

model = BoTBaseline(num_classes=576)
model.load_state_dict(state_dict)
model = model.cuda()
model.eval()
print(f"✓ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")

# 准备数据
print("\n2. Preparing datasets...")
_, test_transform = build_transforms()

# Gallery
gallery_dataset = VeRiDataset('data/dataset/776_DataSet', mode='gallery', transform=test_transform)
gallery_loader = torch.utils.data.DataLoader(gallery_dataset, batch_size=64, shuffle=False, num_workers=4)
print(f"✓ Gallery: {len(gallery_dataset)} images")

# Query (遮挡)
query_dir = Path('outputs/occlusion_tests/veri776_query/ratio_00/center')
metadata_path = query_dir / 'metadata.json'
query_dataset = SimpleOcclusionDataset(query_dir, metadata_path, transform=test_transform)
query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=64, shuffle=False, num_workers=4)
print(f"✓ Query: {len(query_dataset)} images (sample)")

# 评估
print("\n3. Running evaluation...")
evaluator = ReIDEvaluator(model=model, use_flip_test=False, device='cuda')
metrics = evaluator.evaluate(query_loader, gallery_loader, metric='cosine')

print("\n" + "="*80)
print("RESULTS")
print("="*80)
print(f"mAP:    {metrics['mAP']:.4f}")
print(f"Rank-1: {metrics['rank1']:.4f}")
print(f"Rank-5: {metrics['rank5']:.4f}")
print(f"Rank-10: {metrics['rank10']:.4f}")
print("="*80)
print("\n✅ Test completed successfully!")
