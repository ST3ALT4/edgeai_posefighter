"""
main_integrated.py - Integrated game with T1 pose detection
COMPLETE INTEGRATION - Use this instead of main.py
"""

import pygame
import sys
from multiprocessing import Process, Queue
from game.game_engine import GameEngine


def main():
    """Initialize Pygame and start the game WITH pose detection"""
    
    # Create shared queue for pose data
    pose_queue = Queue(maxsize=10)
    
    # Start pose detection in separate process
    print("Starting pose detection process (T1)...")
    from pose_detection.pose_detector import start_pose_detection_process
    pose_process = Process(target=start_pose_detection_process, args=(pose_queue,))
    pose_process.daemon = True
    pose_process.start()
    
    # Small delay to let pose detection initialize
    import time
    time.sleep(2)
    
    # Initialize Pygame
    pygame.init()
    
    # Create and run the game engine with pose queue
    print("Starting game engine (T2)...")
    game = GameEngine(pose_queue=pose_queue)
    game.run()
    
    # Cleanup
    print("\nShutting down...")
    pose_process.terminate()
    pose_process.join(timeout=2)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

