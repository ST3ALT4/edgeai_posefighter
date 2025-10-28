"""
classification_runner_ml.py - Run ML classifier with T1 & T2
Replaces rule-based classifier with trained ResNet model
"""

from multiprocessing import Process, Queue
import time
import cv2
import numpy as np

from pose_classification.ml_classifier import MLPoseClassifier
from pose_classification.config import *


class MLClassificationRunner:
    """
    Runs ML pose classification in separate process
    Reads pose data from T1, sends predictions to T2
    """
    
    def __init__(self, pose_queue: Queue, prediction_queue: Queue):
        """
        Args:
            pose_queue: Queue from T1 (pose detection)
            prediction_queue: Queue to T2 (game engine)
        """
        self.pose_queue = pose_queue
        self.prediction_queue = prediction_queue
        self.classifier = MLPoseClassifier()
        self.running = False
        
        # For visualization, we need access to frames
        # In production, T1 would send both pose data and cropped frames
        
    def run(self):
        """Main classification loop"""
        self.running = True
        print("✓ ML Classification system started (T3)")
        
        while self.running:
            try:
                # Get pose data from T1
                if not self.pose_queue.empty():
                    data = self.pose_queue.get_nowait()
                    
                    # data should contain:
                    # {
                    #     'player1': {landmarks...},
                    #     'player2': {landmarks...},
                    #     'player1_frame': cropped_frame,  # NEW!
                    #     'player2_frame': cropped_frame   # NEW!
                    # }
                    
                    # Classify both players
                    for player_id in [1, 2]:
                        player_key = f'player{player_id}'
                        frame_key = f'{player_key}_frame'
                        
                        if frame_key in data and data[frame_key] is not None:
                            frame = data[frame_key]
                            
                            # ML classification
                            move_name, confidence = self.classifier.classify_pose(
                                player_id, frame
                            )
                            
                            if move_name and confidence > CONFIDENCE_THRESHOLD:
                                # Send to game
                                try:
                                    if not self.prediction_queue.full():
                                        self.prediction_queue.put_nowait(
                                            (player_id, move_name)
                                        )
                                        print(f"✨ Player {player_id}: {move_name} "
                                              f"(ML confidence: {confidence:.2f})")
                                except:
                                    pass
                
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Classification error: {e}")
                time.sleep(0.1)
        
        print("✓ ML Classification stopped")
    
    def stop(self):
        """Stop classification"""
        self.running = False


def start_ml_classification_process(pose_queue: Queue, prediction_queue: Queue):
    """
    Entry point for multiprocessing
    
    Args:
        pose_queue: Queue from T1
        prediction_queue: Queue to T2
    """
    runner = MLClassificationRunner(pose_queue, prediction_queue)
    runner.run()

