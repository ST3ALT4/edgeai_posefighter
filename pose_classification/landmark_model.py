"""
landmark_model.py - Simple MLP for landmark classification
Much faster than ResNet, works for any person!
"""

import torch
import torch.nn as nn
from landmark_config import *


class LandmarkMLP(nn.Module):
    """
    Multi-Layer Perceptron for pose classification from landmarks
    
    SYLLABUS COVERAGE:
    - Fully Connected Networks
    - Batch Normalization
    - Dropout Regularization
    - ReLU Activation
    """
    
    def __init__(self, input_dim=FEATURE_DIM, hidden_dims=HIDDEN_DIMS, 
                 num_classes=NUM_CLASSES, dropout=DROPOUT_RATE):
        """
        Args:
            input_dim: Number of input features (16)
            hidden_dims: List of hidden layer sizes [128, 64, 32]
            num_classes: Number of output classes (3)
            dropout: Dropout probability
        """
        super(LandmarkMLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        print(f"✓ Landmark MLP initialized")
        print(f"  Input: {input_dim} features")
        print(f"  Hidden: {hidden_dims}")
        print(f"  Output: {num_classes} classes")
        print(f"  Dropout: {dropout}")
    
    def forward(self, x):
        """Forward pass"""
        return self.network(x)


def get_optimizer(model, optimizer_name='adam', lr=LEARNING_RATE, 
                 weight_decay=WEIGHT_DECAY):
    """Get optimizer for training"""
    if optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, 
            weight_decay=weight_decay
        )
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    elif optimizer_name == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    print(f"✓ Optimizer: {optimizer_name.upper()} (lr={lr})")
    return optimizer

