"""
test_t1.py - Test T1 pose detection independently
"""

import cv2
import time
from pose_detection.pose_detector import PoseDetectionSystem
from multiprocessing import Queue

def test_pose_detection():
    print("=" * 60)
    print("TESTING T1: POSE DETECTION")
    print("=" * 60)
    
    # Create dummy queue
    pose_queue = Queue(maxsize=10)
    
    # Initialize detector
    detector = PoseDetectionSystem(pose_queue)
    
    print("\nStarting webcam...")
    print("Instructions:")
    print("  - Stand in front of webcam")
    print("  - Move left/right to test player assignment")
    print("  - Have a friend join to test 2-player detection")
    print("  - Press 'Q' to quit")
    print()
    
    # Run detection (will show window)
    detector.run()
    
    # Check queue has data
    if not pose_queue.empty():
        data = pose_queue.get()
        print("\n✅ SUCCESS: Pose data received!")
        print(f"Player 1 detected: {data['player1'] is not None}")
        print(f"Player 2 detected: {data['player2'] is not None}")
        if data['player1']:
            print(f"P1 hip position: ({data['player1']['hip_x']:.2f}, {data['player1']['hip_y']:.2f})")
    else:
        print("\n❌ FAILED: No pose data received")

if __name__ == "__main__":
    test_pose_detection()

