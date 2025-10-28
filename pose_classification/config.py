"""
config.py - SIMPLIFIED: 3 Pose Configuration
"""

import torch

# Dataset settings - SIMPLIFIED TO 3 CLASSES
NUM_CLASSES = 3
CLASS_NAMES = ['block', 'fireball', 'lightning']

# Pose descriptions
POSE_DESCRIPTIONS = {
    'block': 'Arms crossed in front of chest (defensive)',
    'fireball': 'Both arms extended forward (Kamehameha)',
    'lightning': 'Both arms raised above head (Y-shape)'
}

IMAGE_SIZE = 224  # ResNet input size

# Training hyperparameters
BATCH_SIZE = 16  # Reduced for smaller dataset
EPOCHS = 15  # Fewer epochs needed
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

# Data split
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# Model settings
MODEL_NAME = 'resnet18'
FREEZE_BACKBONE = True
DROPOUT_RATE = 0.3

# Optimization
OPTIMIZER = 'adam'
USE_SCHEDULER = True
SCHEDULER_STEP_SIZE = 5
SCHEDULER_GAMMA = 0.1

# Mixed precision
USE_MIXED_PRECISION = True

# Data augmentation
USE_AUGMENTATION = True
AUGMENTATION_PARAMS = {
    'rotation': 10,
    'horizontal_flip': True,
    'brightness': 0.2,
    'contrast': 0.2
}

# Paths
DATA_DIR = 'data/pose_dataset_3class'
MODEL_SAVE_PATH = 'models/pose_classifier_3class.pth'
CHECKPOINT_DIR = 'checkpoints/'
LOGS_DIR = 'logs/'

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inference settings
CONFIDENCE_THRESHOLD = 0.7
SMOOTHING_WINDOW = 5
MIN_HOLD_FRAMES = 3

