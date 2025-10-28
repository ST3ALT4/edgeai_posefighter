"""
pose_receiver.py - UPDATED for multiprocessing integration
"""

from multiprocessing import Queue
from typing import Dict, List, Tuple, Optional


class PoseReceiver:
    """
    Interface for receiving pose data from Team Member 1
    UPDATED to use multiprocessing.Queue
    """
    
    def __init__(self, pose_queue: Queue = None):
        """
        Initialize pose receiver
        
        Args:
            pose_queue: Multiprocessing Queue from T1 (optional for testing)
        """
        if pose_queue is None:
            # For testing without T1
            import queue
            self.pose_queue = queue.Queue(maxsize=10)
        else:
            # Production: use shared multiprocessing queue
            self.pose_queue = pose_queue
        
        # Prediction queue (for T3, still using regular queue for now)
        import queue
        self.prediction_queue = queue.Queue(maxsize=5)
        
        self.latest_poses = None
        self.running = False
        
    def start(self):
        """Start receiving data"""
        self.running = True
        print("PoseReceiver: Started")
    
    def stop(self):
        """Stop receiving data"""
        self.running = False
        print("PoseReceiver: Stopped")
    
    def get_latest_poses(self) -> Optional[Dict]:
        """
        Get the most recent pose data from T1
        
        Returns:
            Dictionary with pose data for both players
        """
        try:
            while not self.pose_queue.empty():
                self.latest_poses = self.pose_queue.get_nowait()
            return self.latest_poses
        except:
            return self.latest_poses
    
    def get_predictions(self) -> List[Tuple[int, str]]:
        """Get attack predictions from T3's pose classifier"""
        predictions = []
        try:
            while not self.prediction_queue.empty():
                pred = self.prediction_queue.get_nowait()
                predictions.append(pred)
        except:
            pass
        return predictions
    
    def put_prediction(self, player_id: int, move_name: str):
        """Put attack prediction into queue (called by T3)"""
        try:
            self.prediction_queue.put_nowait((player_id, move_name))
        except:
            pass

