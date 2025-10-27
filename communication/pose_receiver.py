"""
pose_receiver.py - Interface to receive pose data from T1 and predictions from T3
Tasks: T2.4, T2.5
"""

import queue
from typing import Dict, List, Tuple, Optional

class PoseReceiver:
    """
    Interface for receiving pose data from Team Member 1 and 
    predictions from Team Member 3
    
    In production, this would use multiprocessing.Queue or similar
    For now, provides mock interface for testing
    """
    
    def __init__(self):
        """Initialize pose receiver"""
        # These would be multiprocessing.Queue in production
        self.pose_queue = queue.Queue(maxsize=10)
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
            Dictionary with pose data for both players:
            {
                'player1': {'hip_x': float, 'hip_y': float, ...},
                'player2': {'hip_x': float, 'hip_y': float, ...}
            }
        """
        # Try to get latest from queue
        try:
            while not self.pose_queue.empty():
                self.latest_poses = self.pose_queue.get_nowait()
            return self.latest_poses
        except queue.Empty:
            return self.latest_poses
    
    def get_predictions(self) -> List[Tuple[int, str]]:
        """
        Get attack predictions from T3's pose classifier
        
        Returns:
            List of tuples: [(player_id, move_name), ...]
            Example: [(1, "fireball"), (2, "shield")]
        """
        predictions = []
        
        try:
            while not self.prediction_queue.empty():
                pred = self.prediction_queue.get_nowait()
                predictions.append(pred)
        except queue.Empty:
            pass
        
        return predictions
    
    def put_pose_data(self, pose_data: Dict):
        """
        Put pose data into queue (called by T1)
        
        Args:
            pose_data: Pose data dictionary
        """
        try:
            self.pose_queue.put_nowait(pose_data)
        except queue.Full:
            # Remove oldest and add new
            try:
                self.pose_queue.get_nowait()
                self.pose_queue.put_nowait(pose_data)
            except:
                pass
    
    def put_prediction(self, player_id: int, move_name: str):
        """
        Put attack prediction into queue (called by T3)
        
        Args:
            player_id: 1 or 2
            move_name: Name of predicted move
        """
        try:
            self.prediction_queue.put_nowait((player_id, move_name))
        except queue.Full:
            # Remove oldest and add new
            try:
                self.prediction_queue.get_nowait()
                self.prediction_queue.put_nowait((player_id, move_name))
            except:
                pass


# For testing: Mock pose data generator
class MockPoseGenerator:
    """Generate mock pose data for testing without T1"""
    
    def __init__(self, pose_receiver):
        """
        Initialize mock generator
        
        Args:
            pose_receiver: PoseReceiver instance to send data to
        """
        self.pose_receiver = pose_receiver
        self.frame = 0
        
    def update(self):
        """Generate and send mock pose data"""
        import math
        
        self.frame += 1
        
        # Simulate players moving back and forth
        player1_x = 0.5 + 0.3 * math.sin(self.frame * 0.05)
        player2_x = 0.5 + 0.3 * math.cos(self.frame * 0.05)
        
        mock_data = {
            'player1': {
                'hip_x': player1_x,
                'hip_y': 0.6,
            },
            'player2': {
                'hip_x': player2_x,
                'hip_y': 0.6,
            }
        }
        
        self.pose_receiver.put_pose_data(mock_data)
        
        # Occasionally trigger random attacks
        import random
        if random.random() < 0.01:  # 1% chance per frame
            player = random.choice([1, 2])
            move = random.choice(["fireball", "lightning", "shield", "ground_pound", "energy_beam"])
            self.pose_receiver.put_prediction(player, move)

