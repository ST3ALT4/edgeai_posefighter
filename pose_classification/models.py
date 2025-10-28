"""
models.py - ResNet Transfer Learning Model
SYLLABUS COVERAGE: CNN Architectures, Transfer Learning, Batch Normalization, Regularization
"""

import torch
import torch.nn as nn
from torchvision import models
from config import *


class PoseClassifierResNet(nn.Module):
    """
    Transfer Learning Model using ResNet
    
    SYLLABUS COVERAGE:
    - CNN Architecture (ResNet18/34/50)
    - Transfer Learning (pre-trained on ImageNet)
    - Batch Normalization
    - Dropout Regularization
    - Activation Functions
    """
    
    def __init__(self, num_classes=NUM_CLASSES, model_name='resnet18', 
                 freeze_backbone=FREEZE_BACKBONE, dropout_rate=DROPOUT_RATE):
        """
        Initialize ResNet-based pose classifier
        
        Args:
            num_classes: Number of pose classes (5 for our game)
            model_name: 'resnet18', 'resnet34', or 'resnet50'
            freeze_backbone: If True, freeze pretrained layers (transfer learning)
            dropout_rate: Dropout probability for regularization
        """
        super(PoseClassifierResNet, self).__init__()
        
        # Load pre-trained ResNet (SYLLABUS: Transfer Learning)
        print(f"Loading pre-trained {model_name}...")
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=True)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Freeze backbone layers (SYLLABUS: Transfer Learning)
        if freeze_backbone:
            print("Freezing backbone layers for transfer learning...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get number of features from last layer
        num_features = self.backbone.fc.in_features
        
        # Replace final fully connected layer with custom classifier
        # SYLLABUS: Batch Normalization, Dropout, Activation Functions
        self.backbone.fc = nn.Sequential(
            # First layer
            nn.Linear(num_features, 512),
            nn.ReLU(),  # SYLLABUS: Activation Function
            nn.BatchNorm1d(512),  # SYLLABUS: Batch Normalization
            nn.Dropout(dropout_rate),  # SYLLABUS: Regularization
            
            # Second layer
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(256, num_classes)
        )
        
        print(f"✓ Model initialized: {model_name}")
        print(f"  - Num classes: {num_classes}")
        print(f"  - Backbone frozen: {freeze_backbone}")
        print(f"  - Dropout rate: {dropout_rate}")
    
    def forward(self, x):
        """Forward pass through the network"""
        return self.backbone(x)
    
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        print("Unfreezing backbone for fine-tuning...")
        for param in self.backbone.parameters():
            param.requires_grad = True


def get_optimizer(model, optimizer_name='adam', lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    """
    Get optimizer for training
    
    SYLLABUS COVERAGE: Optimization Algorithms (SGD, Adam, RMSprop)
    
    Args:
        model: PyTorch model
        optimizer_name: 'sgd', 'adam', 'rmsprop', 'adamw'
        lr: Learning rate
        weight_decay: L2 regularization weight
    
    Returns:
        Optimizer instance
    """
    if optimizer_name == 'sgd':
        # Stochastic Gradient Descent with momentum
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True
        )
    elif optimizer_name == 'adam':
        # Adam optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay
        )
    elif optimizer_name == 'rmsprop':
        # RMSprop optimizer
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            alpha=0.99,
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adamw':
        # AdamW (Adam with weight decay)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    print(f"✓ Optimizer: {optimizer_name.upper()} (lr={lr}, weight_decay={weight_decay})")
    return optimizer


def get_scheduler(optimizer, step_size=SCHEDULER_STEP_SIZE, gamma=SCHEDULER_GAMMA):
    """
    Get learning rate scheduler
    
    Args:
        optimizer: PyTorch optimizer
        step_size: Period of learning rate decay
        gamma: Multiplicative factor of learning rate decay
    
    Returns:
        Scheduler instance
    """
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size,
        gamma=gamma
    )
    print(f"✓ Scheduler: StepLR (step_size={step_size}, gamma={gamma})")
    return scheduler

