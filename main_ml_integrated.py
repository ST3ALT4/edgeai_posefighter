"""
main_ml_integration.py - Complete system with ML pose classification
T1 (Pose Detection) + T2 (Game) + T3 (ML Classifier)
"""

import pygame
import sys
from multiprocessing import Process, Queue
from game.game_engine import GameEngine


def main():
    """Full system with ML classification"""
    
    print("=" * 60)
    print("POSE FIGHTERS - ML CLASSIFICATION ENABLED")
    print("=" * 60)
    
    # Create queues
    pose_queue = Queue(maxsize=10)
    prediction_queue = Queue(maxsize=5)
    
    # Start T1: Pose Detection
    print("[1/3] Starting Pose Detection (T1)...")
    from pose_detection.pose_detector import start_pose_detection_process
    pose_process = Process(target=start_pose_detection_process, args=(pose_queue,))
    pose_process.daemon = True
    pose_process.start()
    
    # Start T3: ML Classification
    print("[2/3] Starting ML Classification (T3)...")
    from pose_classification.classification_runner_ml import start_ml_classification_process
    classification_process = Process(
        target=start_ml_classification_process,
        args=(pose_queue, prediction_queue)
    )
    classification_process.daemon = True
    classification_process.start()
    
    # Start T2: Game Engine
    import time
    time.sleep(2)
    print("[3/3] Starting Game Engine (T2)...")
    pygame.init()
    
    game = GameEngine(
        pose_queue=pose_queue,
        prediction_queue=prediction_queue
    )
    
    print("\\n" + "=" * 60)
    print("✅ SYSTEM READY - ML CLASSIFIER ACTIVE")
    print("=" * 60)
    print("Trained model: models/pose_classifier_best.pth")
    print("Perform poses to trigger attacks automatically!")
    print("=" * 60)
    
    # Run game
    game.run()
    
    # Cleanup
    print("\\nShutting down...")
    pose_process.terminate()
    classification_process.terminate()
    pose_process.join(timeout=2)
    classification_process.join(timeout=2)
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

