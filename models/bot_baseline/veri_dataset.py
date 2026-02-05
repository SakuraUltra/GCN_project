"""
VeRi-776 and VehicleID Dataset Manager
Handles data loading, parsing, and preprocessing for both datasets
"""

import os
import glob
import re
import random
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np


class VeRiDataset(Dataset):
    """
    VeRi-776 Dataset for Vehicle Re-identification
    """
    def __init__(self, root, mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        
        # Dataset paths
        if mode == 'train':
            self.data_path = os.path.join(root, 'image_train')
        elif mode == 'query':
            self.data_path = os.path.join(root, 'image_query')
        elif mode == 'gallery':
            self.data_path = os.path.join(root, 'image_test')
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
        # Parse dataset
        self.data = self._parse_data()
        
        if mode == 'train':
            self.pids = sorted(list(set([item[1] for item in self.data])))
            self.pid2label = {pid: label for label, pid in enumerate(self.pids)}
            print(f"VeRi-776 Training set: {len(self.data)} images, {len(self.pids)} identities")
        else:
            print(f"VeRi-776 {mode.capitalize()} set: {len(self.data)} images")
    
    def _parse_data(self):
        img_paths = glob.glob(os.path.join(self.data_path, '*.jpg'))
        data = []
        for img_path in img_paths:
            filename = os.path.basename(img_path)
            pattern = r'(\d+)_c(\d+)_(\d+)_(\d+)\.jpg'
            match = re.match(pattern, filename)
            if match:
                vehicle_id = int(match.group(1))
                camera_id = int(match.group(2))
                data.append((img_path, vehicle_id, camera_id))
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, pid, camid = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.mode == 'train':
            label = self.pid2label[pid]
            return img, label, pid, camid
        else:
            return img, pid, camid


class VehicleIDDataset(Dataset):
    """
    VehicleID Dataset for Vehicle Re-identification
    Structure:
    data/VehicleID_V1.0/
    ├── image/ (or *.jpg at root)
    └── train_test_split/
        ├── train_list.txt
        ├── test_list_800.txt
        └── ...
    """
    
    def __init__(self, root, list_file=None, mode='train', transform=None, test_data=None):
        """
        Args:
            root: Root directory of VehicleID dataset
            list_file: Path to the list file (train_list.txt or test_list_*.txt)
            mode: 'train', 'query', or 'gallery'
            transform: Transformations
            test_data: Tuple of (img_path, pid) list, used when splitting test set into query/gallery
        """
        self.root = root
        self.mode = mode
        self.transform = transform
        
        # Determine image directory
        # Check if images are in 'image' folder or root
        if os.path.exists(os.path.join(root, 'image')):
            self.img_dir = os.path.join(root, 'image')
        else:
            self.img_dir = root
            
        if mode == 'train':
            if list_file is None:
                list_file = os.path.join(root, 'train_test_split', 'train_list.txt')
            self.data = self._parse_list(list_file)
            
            self.pids = sorted(list(set([item[1] for item in self.data])))
            self.pid2label = {pid: label for label, pid in enumerate(self.pids)}
            print(f"VehicleID Training set: {len(self.data)} images, {len(self.pids)} identities")
            
        elif mode in ['query', 'gallery']:
            # For query/gallery, we expect data passed from the splitting logic
            if test_data is None:
                raise ValueError("For VehicleID query/gallery mode, 'test_data' must be provided")
            self.data = test_data
            # Set mocked camid
            self.camid = 0 if mode == 'query' else 1
            print(f"VehicleID {mode.capitalize()} set: {len(self.data)} images")
            
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _parse_list(self, list_file):
        data = []
        if not os.path.exists(list_file):
            raise FileNotFoundError(f"List file not found: {list_file}")
            
        with open(list_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_id = parts[0]
                pid = int(parts[1])
                # Image file usually is img_id + ".jpg"
                img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
                if os.path.exists(img_path):
                    data.append((img_path, pid, 0))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # VehicleID doesn't have camid in list, so we mock it
        # Train: camid=0
        # Query: camid=0, Gallery: camid=1 (to ensure they don't match by camid filter if any)
        
        img_path, pid, _ = self.data[idx]
        if self.mode == 'train':
            camid = 0 
        elif self.mode == 'query':
            camid = 0
        else: # gallery
            camid = 1
            
        img = Image.open(img_path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.mode == 'train':
            label = self.pid2label[pid]
            return img, label, pid, camid
        else:
            return img, pid, camid


def build_transforms(height=256, width=256, random_erase=True):
    """Build data transforms"""
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    
    train_transforms = [
        T.Resize((height, width)),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalize,
    ]
    
    if random_erase:
        from data.transforms.data_transforms import RandomErasing
        train_transforms.append(RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406]))
    
    test_transforms = [
        T.Resize((height, width)),
        T.ToTensor(),
        normalize,
    ]
    
    return T.Compose(train_transforms), T.Compose(test_transforms)


def split_vehicleid_test(root, test_list_name='test_list_800.txt'):
    """
    Split VehicleID test list into Query and Gallery.
    Protocol: Randomly select 1 image per identity for Gallery, rest for Query.
    """
    list_path = os.path.join(root, 'train_test_split', test_list_name)
    if not os.path.exists(list_path):
        print(f"Warning: Test list {list_path} not found. Trying test_list_800.txt")
        list_path = os.path.join(root, 'train_test_split', 'test_list_800.txt')
    
    # Check for image dir
    if os.path.exists(os.path.join(root, 'image')):
        img_dir = os.path.join(root, 'image')
    else:
        img_dir = root

    pid_dict = defaultdict(list)
    with open(list_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                img_id = parts[0]
                pid = int(parts[1])
                img_path = os.path.join(img_dir, f"{img_id}.jpg")
                if os.path.exists(img_path):
                    pid_dict[pid].append((img_path, pid, 0))
    
    gallery_data = []
    query_data = []
    
    # Set seed for reproducibility
    random.seed(1) 
    
    for pid, items in pid_dict.items():
        if len(items) < 1:
            continue
        # Randomly sample 1 for gallery
        gallery_sample = random.choice(items)
        gallery_data.append(gallery_sample)
        
        # Rest for query
        for item in items:
            if item != gallery_sample:
                query_data.append(item)
    
    return query_data, gallery_data


def create_data_loaders(data_root, batch_size=64, num_workers=4, use_pk_sampler=True, p=16, k=4):
    """
    Create data loaders. 
    detects dataset type ('Structure A': VeRi vs 'Structure B': VehicleID) based on path.
    """
    
    # Detect Dataset Type
    is_vehicleID = 'VehicleID' in data_root
    
    train_transform, test_transform = build_transforms()
    
    if is_vehicleID:
        print(f"Detected VehicleID dataset at {data_root}")
        # Train Dataset
        train_dataset = VehicleIDDataset(data_root, mode='train', transform=train_transform)
        
        # Split Test Data
        query_data, gallery_data = split_vehicleid_test(data_root, 'test_list_800.txt')
        
        query_dataset = VehicleIDDataset(data_root, mode='query', transform=test_transform, test_data=query_data)
        gallery_dataset = VehicleIDDataset(data_root, mode='gallery', transform=test_transform, test_data=gallery_data)
        
        num_classes = len(train_dataset.pids)
        
    else:
        print(f"Detected VeRi-776 dataset at {data_root}")
        train_dataset = VeRiDataset(data_root, mode='train', transform=train_transform)
        query_dataset = VeRiDataset(data_root, mode='query', transform=test_transform)
        gallery_dataset = VeRiDataset(data_root, mode='gallery', transform=test_transform)
        num_classes = len(train_dataset.pids)

    # Dataloaders
    use_pin_memory = True
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        use_pin_memory = False
    
    if use_pk_sampler:
        from utils.pk_sampler import create_pk_dataloader
        train_loader = create_pk_dataloader(train_dataset, p=p, k=k, num_workers=num_workers)
        print(f"Using PK Sampler: P={p}, K={k}, Batch Size={p*k}")
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=use_pin_memory, drop_last=True
        )
    
    query_loader = torch.utils.data.DataLoader(
        query_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    
    gallery_loader = torch.utils.data.DataLoader(
        gallery_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory
    )
    
    return train_loader, query_loader, gallery_loader, num_classes
