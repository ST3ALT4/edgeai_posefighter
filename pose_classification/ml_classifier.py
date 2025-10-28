"""
ml_classifier.py - ML Classifier for Real-Time Inference
Uses trained ResNet model for pose classification
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from collections import deque

from models import PoseClassifierResNet
from config import *


class MLPoseClassifier:
    """
    Real-time pose classifier using trained ML model
    """
    
    def __init__(self, model_path=MODEL_SAVE_PATH, device=None):
        """
        Initialize classifier
        
        Args:
            model_path: Path to trained model
            device: torch.device (auto-detect if None)
        """
        self.device = device if device else DEVICE
        self.model = None
        self.transform = None
        self.class_names = CLASS_NAMES
        
        # Smoothing buffers
        self.prediction_history = {
            1: deque(maxlen=SMOOTHING_WINDOW),
            2: deque(maxlen=SMOOTHING_WINDOW)
        }
        
        # Hold counters
        self.hold_counters = {
            1: {},
            2: {}
        }
        
        # Load model
        self.load_model(model_path)
        
        # Setup transforms
        self.setup_transforms()
        
        print(f"✓ ML Classifier initialized")
        print(f"  Device: {self.device}")
        print(f"  Model: {model_path}")
    
    def load_model(self, model_path):
        """Load trained model"""
        # Create model
        self.model = PoseClassifierResNet(
            num_classes=NUM_CLASSES,
            model_name=MODEL_NAME
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded (Val Acc: {checkpoint['best_val_acc']:.2f}%)")
    
    def setup_transforms(self):
        """Setup image preprocessing transforms"""
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_frame(self, frame):
        """
        Preprocess webcam frame for model input
        
        Args:
            frame: OpenCV BGR image
        
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL
        pil_image = Image.fromarray(rgb)
        
        # Apply transforms
        tensor = self.transform(pil_image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def classify_pose(self, player_id, frame):
        """
        Classify pose from webcam frame
        
        Args:
            player_id: 1 or 2
            frame: OpenCV image (cropped to player region)
        
        Returns:
            Tuple of (move_name, confidence) or (None, 0)
        """
        if frame is None or frame.size == 0:
            return None, 0.0
        
        # Preprocess
        tensor = self.preprocess_frame(frame).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            confidence = confidence.item()
            predicted_class = predicted.item()
            move_name = self.class_names[predicted_class]
        
        # Apply smoothing and hold detection
        if confidence > CONFIDENCE_THRESHOLD:
            return self._smooth_prediction(player_id, move_name, confidence)
        
        return None, 0.0
    
    def _smooth_prediction(self, player_id, move_name, confidence):
        """Apply temporal smoothing"""
        # Add to history
        self.prediction_history[player_id].append((move_name, confidence))
        
        # Count occurrences
        move_counts = {}
        for move, conf in self.prediction_history[player_id]:
            if move not in move_counts:
                move_counts[move] = []
            move_counts[move].append(conf)
        
        # Find most common
        if move_counts:
            best_move = max(move_counts.keys(), key=lambda m: len(move_counts[m]))
            avg_confidence = sum(move_counts[best_move]) / len(move_counts[best_move])
            
            # Hold counter
            if best_move not in self.hold_counters[player_id]:
                self.hold_counters[player_id][best_move] = 0
            
            self.hold_counters[player_id][best_move] += 1
            
            # Trigger if held
            if self.hold_counters[player_id][best_move] >= MIN_HOLD_FRAMES:
                self.hold_counters[player_id][best_move] = 0
                return best_move, avg_confidence
        
        return None, 0.0

