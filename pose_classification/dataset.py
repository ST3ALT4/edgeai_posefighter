"""
dataset.py - Custom PyTorch Dataset for Pose Classification
SYLLABUS COVERAGE: Custom Dataset, Data Augmentation
"""

import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from config import *


class PoseDataset(Dataset):
    """
    Custom Dataset for Pose Classification
    
    SYLLABUS COVERAGE: Custom Dataset for Retraining
    
    Directory structure:
    data/pose_dataset/
        fireball/
            img_001.jpg
            img_002.jpg
            ...
        lightning/
            img_001.jpg
            ...
    """
    
    def __init__(self, root_dir, transform=None, augment=False):
        """
        Args:
            root_dir: Root directory with class subdirectories
            transform: PyTorch transforms
            augment: Whether to apply data augmentation
        """
        self.root_dir = root_dir
        self.transform = transform
        self.augment = augment
        self.samples = []
        self.class_to_idx = {}
        
        # Load samples
        self._load_samples()
        
        print(f"✓ Dataset loaded: {len(self.samples)} samples")
        for class_name, class_idx in self.class_to_idx.items():
            count = sum(1 for _, label in self.samples if label == class_idx)
            print(f"  - {class_name}: {count} samples")
    
    def _load_samples(self):
        """Load all samples from directory structure"""
        # Get class directories
        class_dirs = sorted([d for d in os.listdir(self.root_dir)
                           if os.path.isdir(os.path.join(self.root_dir, d))])
        
        # Create class to index mapping
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(class_dirs)}
        
        # Load all image paths and labels
        for class_name in class_dirs:
            class_idx = self.class_to_idx[class_name]
            class_dir = os.path.join(self.root_dir, class_name)
            
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(augment=False):
    """
    Get image transforms
    
    SYLLABUS COVERAGE: Data Augmentation
    
    Args:
        augment: Whether to apply augmentation
    
    Returns:
        torchvision.transforms.Compose
    """
    if augment:
        # Training transforms with augmentation
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomRotation(AUGMENTATION_PARAMS['rotation']),
            transforms.RandomHorizontalFlip() if AUGMENTATION_PARAMS['horizontal_flip'] else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(
                brightness=AUGMENTATION_PARAMS['brightness'],
                contrast=AUGMENTATION_PARAMS['contrast']
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
    else:
        # Validation/test transforms (no augmentation)
        transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_data_loaders(data_dir, batch_size=BATCH_SIZE):
    """
    Create train, validation, and test data loaders
    
    Args:
        data_dir: Root directory with pose data
        batch_size: Batch size for training
    
    Returns:
        train_loader, val_loader, test_loader
    """
    from sklearn.model_selection import train_test_split
    
    # Load full dataset
    full_dataset = PoseDataset(data_dir, transform=None)
    
    # Split into train/val/test
    train_val_indices, test_indices = train_test_split(
        list(range(len(full_dataset))),
        test_size=TEST_SPLIT,
        random_state=42,
        stratify=[label for _, label in full_dataset.samples]
    )
    
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=VAL_SPLIT / (TRAIN_SPLIT + VAL_SPLIT),
        random_state=42,
        stratify=[full_dataset.samples[i] for i in train_val_indices]
    )
    
    # Create datasets with appropriate transforms
    train_dataset = torch.utils.data.Subset(
        PoseDataset(data_dir, transform=get_transforms(augment=True)),
        train_indices
    )
    val_dataset = torch.utils.data.Subset(
        PoseDataset(data_dir, transform=get_transforms(augment=False)),
        val_indices
    )
    test_dataset = torch.utils.data.Subset(
        PoseDataset(data_dir, transform=get_transforms(augment=False)),
        test_indices
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ Data loaders created:")
    print(f"  - Train: {len(train_dataset)} samples")
    print(f"  - Val: {len(val_dataset)} samples")
    print(f"  - Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader

