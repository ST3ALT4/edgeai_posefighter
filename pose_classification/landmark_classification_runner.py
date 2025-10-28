"""
landmark_classification_runner.py - Integration with T1 & T2
Runs in separate process
"""

from multiprocessing import Process, Queue
import time

from landmark_classifier import LandmarkClassifier
from landmark_config import *


class LandmarkClassificationRunner:
    """
    Runs landmark-based classification in separate process
    Much faster than image-based!
    """
    
    def __init__(self, pose_queue: Queue, prediction_queue: Queue):
        self.pose_queue = pose_queue
        self.prediction_queue = prediction_queue
        self.classifier = LandmarkClassifier()
        self.running = False
    
    def run(self):
        """Main classification loop"""
        self.running = True
        print("✓ Landmark Classification started (T3)")
        
        while self.running:
            try:
                if not self.pose_queue.empty():
                    data = self.pose_queue.get_nowait()
                    
                    # Classify both players
                    for player_id in [1, 2]:
                        player_key = f'player{player_id}'
                        landmarks = data.get(player_key)
                        
                        if landmarks:
                            move_name, confidence = self.classifier.classify_pose(
                                player_id, landmarks
                            )
                            
                            if move_name and confidence > CONFIDENCE_THRESHOLD:
                                try:
                                    if not self.prediction_queue.full():
                                        self.prediction_queue.put_nowait(
                                            (player_id, move_name)
                                        )
                                        print(f"✨ P{player_id}: {move_name} "
                                              f"(conf: {confidence:.2f})")
                                except:
                                    pass
                
                time.sleep(0.01)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.1)
        
        print("✓ Landmark Classification stopped")
    
    def stop(self):
        self.running = False


def start_landmark_classification(pose_queue: Queue, prediction_queue: Queue):
    """Entry point for multiprocessing"""
    runner = LandmarkClassificationRunner(pose_queue, prediction_queue)
    runner.run()

