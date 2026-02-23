"""
PK Sampler for Vehicle Re-identification
P identities × K instances per batch
Essential for Triplet Loss training
"""

import random
import numpy as np
from collections import defaultdict
import torch
from torch.utils.data.sampler import Sampler


class PKSampler(Sampler):
    """
    PK Sampler for ReID training
    
    Sample P identities, then sample K instances for each identity
    Batch size = P × K
    
    This ensures each batch contains both positive and negative pairs
    which is essential for triplet loss computation
    """
    
    def __init__(self, dataset, p=16, k=4):
        """
        Args:
            dataset: Training dataset
            p (int): Number of identities per batch  
            k (int): Number of instances per identity
        """
        self.dataset = dataset
        self.p = p
        self.k = k
        self.batch_size = p * k
        
        # Group data by identity
        self.pid_index = defaultdict(list)
        for idx, (_, pid, _) in enumerate(dataset.data):
            # Convert to label index
            if hasattr(dataset, 'pid2label'):
                label = dataset.pid2label[pid]
                self.pid_index[label].append(idx)
            else:
                self.pid_index[pid].append(idx)
        
        self.pids = list(self.pid_index.keys())
        
        # Calculate number of batches per epoch
        self.num_samples = len(self.pids) * self.k
        self.length = self.num_samples // self.batch_size
        
        print(f"PKSampler: {len(self.pids)} identities, P={p}, K={k}")
        print(f"Batch size: {self.batch_size}, Batches per epoch: {self.length}")
    
    def __iter__(self):
        """Generate batches for one epoch"""
        batch_indices = []
        
        for _ in range(self.length):
            # Sample P identities
            selected_pids = random.sample(self.pids, self.p)
            
            batch = []
            for pid in selected_pids:
                # Sample K instances for this identity
                pid_indices = self.pid_index[pid]
                if len(pid_indices) >= self.k:
                    # Sufficient instances: random sample
                    selected = random.sample(pid_indices, self.k)
                else:
                    # Insufficient instances: sample with replacement
                    selected = random.choices(pid_indices, k=self.k)
                
                batch.extend(selected)
            
            batch_indices.extend(batch)
        
        return iter(batch_indices)
    
    def __len__(self):
        return self.length * self.batch_size


def create_pk_dataloader(dataset, p=16, k=4, num_workers=4):
    """
    Create dataloader with PK sampling strategy
    
    Args:
        dataset: PyTorch dataset
        p: Number of identities per batch
        k: Number of instances per identity
        num_workers: Number of worker threads
    
    Returns:
        DataLoader with PK sampler
    """
    sampler = PKSampler(dataset, p=p, k=k)
    
    # Check if MPS is available and being used - MPS doesn't support pin_memory
    use_pin_memory = True
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        use_pin_memory = False
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=p * k,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        drop_last=False  # PKSampler handles batch formation
    )
    
    return dataloader