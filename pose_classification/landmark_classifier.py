"""
landmark_classifier.py - Real-time inference with landmark-based model
FAST: 1-2ms per classification!
"""

import torch
import torch.nn.functional as F
from collections import deque

from .landmark_model import LandmarkMLP
from .landmark_features import LandmarkFeatureExtractor
from .landmark_config import *


class LandmarkClassifier:
    """
    Real-time pose classifier using landmarks
    Works for ANY person - completely general!
    """
    
    def __init__(self, model_path=MODEL_SAVE_PATH):
        self.device = DEVICE
        self.feature_extractor = LandmarkFeatureExtractor()
        self.class_names = CLASS_NAMES
        
        # Load model
        self.model = LandmarkMLP().to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Smoothing
        self.prediction_history = {
            1: deque(maxlen=SMOOTHING_WINDOW),
            2: deque(maxlen=SMOOTHING_WINDOW)
        }
        
        self.hold_counters = {1: {}, 2: {}}
        
        print(f"✓ Landmark Classifier initialized")
        print(f"  Model: {model_path}")
        print(f"  Best Val Acc: {checkpoint['best_val_acc']:.2f}%")
    
    def classify_pose(self, player_id, landmarks_dict):
        """
        Classify pose from landmarks
        
        Args:
            player_id: 1 or 2
            landmarks_dict: Landmarks from T1 (MediaPipe)
        
        Returns:
            (move_name, confidence) or (None, 0)
        """
        if landmarks_dict is None:
            return None, 0.0
        
        # Extract features
        features = self.feature_extractor.extract_features(landmarks_dict)
        if features is None:
            return None, 0.0
        
        # Convert to tensor
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            confidence = confidence.item()
            predicted_class = predicted.item()
            move_name = self.class_names[predicted_class]
        
        # Apply smoothing
        if confidence > CONFIDENCE_THRESHOLD:
            return self._smooth_prediction(player_id, move_name, confidence)
        
        return None, 0.0
    
    def _smooth_prediction(self, player_id, move_name, confidence):
        """Apply temporal smoothing"""
        self.prediction_history[player_id].append((move_name, confidence))
        
        # Count occurrences
        move_counts = {}
        for move, conf in self.prediction_history[player_id]:
            if move not in move_counts:
                move_counts[move] = []
            move_counts[move].append(conf)
        
        if move_counts:
            best_move = max(move_counts.keys(), key=lambda m: len(move_counts[m]))
            avg_conf = sum(move_counts[best_move]) / len(move_counts[best_move])
            
            # Hold counter
            if best_move not in self.hold_counters[player_id]:
                self.hold_counters[player_id][best_move] = 0
            
            self.hold_counters[player_id][best_move] += 1
            
            if self.hold_counters[player_id][best_move] >= MIN_HOLD_FRAMES:
                self.hold_counters[player_id][best_move] = 0
                return best_move, avg_conf
        
        return None, 0.0

