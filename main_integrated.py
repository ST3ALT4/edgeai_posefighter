"""
main_landmark_integration.py - Full System Integration
T1 + T2 + T3 all together
"""

import pygame
import sys
from multiprocessing import Process, Queue
from game.game_engine import GameEngine


def main():
    print("\n" + "=" * 70)
    print("POSE FIGHTERS - FULL SYSTEM")
    print("=" * 70)
    print("T1: Pose Detection | T2: Game Engine | T3: ML Classification")
    print("=" * 70)
    
    # Create queues
    pose_queue = Queue(maxsize=10)
    prediction_queue = Queue(maxsize=5)
    
    # Start T1
    print("\n[1/3] Starting T1 (Pose Detection)...")
    from pose_detection.pose_detector import start_pose_detection_process
    pose_process = Process(target=start_pose_detection_process, args=(pose_queue,))
    pose_process.daemon = True
    pose_process.start()
    
    # Start T3
    print("[2/3] Starting T3 (ML Classification)...")
    from pose_classification.landmark_classification_runner import start_landmark_classification
    classification_process = Process(target=start_landmark_classification, args=(pose_queue, prediction_queue))
    classification_process.daemon = True
    classification_process.start()
    
    # Wait for initialization
    import time
    print("\nInitializing...")
    time.sleep(2)
    
    # Start T2
    print("[3/3] Starting T2 (Game Engine)...")
    pygame.init()
    game = GameEngine(pose_queue=pose_queue, prediction_queue=prediction_queue)
    
    print("\n" + "=" * 70)
    print("✅ ALL SYSTEMS ONLINE!")
    print("=" * 70)
    print("\n📹 Webcam: Shows pose detection")
    print("🎮 Game: Control with your body!")
    print("\n💪 Poses:")
    print("   🛡️  BLOCK: Arms crossed")
    print("   🔥 FIREBALL: Arms forward")
    print("   ⚡ LIGHTNING: Arms up")
    print("\n⌨️  Keyboard fallback:")
    print("   P1: Q/W/E  |  P2: U/I/O")
    print("=" * 70)
    
    # Run game
    try:
        game.run()
    except KeyboardInterrupt:
        print("\n\nGame stopped by user")
    
    # Cleanup
    print("\nShutting down...")
    pose_process.terminate()
    classification_process.terminate()
    pose_process.join(timeout=2)
    classification_process.join(timeout=2)
    pygame.quit()
    print("✓ Done!")


if __name__ == "__main__":
    main()
