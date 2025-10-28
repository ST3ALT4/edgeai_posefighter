"""
landmark_config.py - Configuration for landmark-based classifier
BETTER: General model that works for any person!
"""

import torch

# Model settings
NUM_CLASSES = 3
CLASS_NAMES = ['block', 'fireball', 'lightning']
FEATURE_DIM = 16  # Number of engineered features

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 30  # Can train longer since it's fast
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

# Data split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# MLP architecture
HIDDEN_DIMS = [128, 64, 32]  # 3 hidden layers
DROPOUT_RATE = 0.3

# Optimization
OPTIMIZER = 'adam'
USE_SCHEDULER = True

# Paths
DATA_DIR = 'data/landmark_dataset'  # Saves as .npy files
MODEL_SAVE_PATH = 'models/landmark_classifier.pth'

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inference
CONFIDENCE_THRESHOLD = 0.75
SMOOTHING_WINDOW = 5
MIN_HOLD_FRAMES = 3

