"""
landmark_dataset.py - PyTorch Dataset for landmark features
Much simpler than image dataset!
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from .landmark_config import *


class LandmarkDataset(Dataset):
    """
    PyTorch Dataset for landmark features
    Loads .npy files instead of images
    """
    
    def __init__(self, data_dir, class_names=CLASS_NAMES):
        """
        Args:
            data_dir: Directory with class subdirectories
            class_names: List of class names
        """
        self.data_dir = data_dir
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.samples = self._load_samples()
        
        print(f"✓ Landmark dataset loaded: {len(self.samples)} samples")
        for class_name in class_names:
            count = sum(1 for _, label in self.samples if label == self.class_to_idx[class_name])
            print(f"  - {class_name}: {count} samples")
    
    def _load_samples(self):
        """Load all .npy files"""
        samples = []
        
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(self.data_dir, class_name)
            
            if not os.path.exists(class_dir):
                continue
            
            for filename in os.listdir(class_dir):
                if filename.endswith('.npy'):
                    filepath = os.path.join(class_dir, filename)
                    samples.append((filepath, class_idx))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        
        # Load features
        features = np.load(filepath)
        features = torch.from_numpy(features).float()
        
        return features, label


def create_data_loaders(data_dir=DATA_DIR, batch_size=BATCH_SIZE):
    """
    Create train/val/test data loaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Load full dataset
    full_dataset = LandmarkDataset(data_dir)
    
    # Get all indices
    indices = list(range(len(full_dataset)))
    labels = [label for _, label in full_dataset.samples]
    
    # Split into train/val/test
    train_val_indices, test_indices = train_test_split(
        indices, test_size=TEST_SPLIT, random_state=42, stratify=labels
    )
    
    train_val_labels = [labels[i] for i in train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT),
        random_state=42,
        stratify=train_val_labels
    )
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    print(f"✓ Data loaders created:")
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Val: {len(val_dataset)} samples")
    print(f"  - Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader

