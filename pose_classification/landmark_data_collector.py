"""
landmark_data_collector.py - Collect landmark data (not images!)
Much faster and more general than image-based approach
"""

import cv2
import numpy as np
import os
from datetime import datetime
from multiprocessing import Queue

from pose_detection.pose_detector import PoseDetectionSystem
from landmark_features import LandmarkFeatureExtractor
from landmark_config import CLASS_NAMES, DATA_DIR


class LandmarkDataCollector:
    """
    Collect pose landmark features for training
    Saves numpy arrays instead of images - much faster!
    """
    
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir
        self.feature_extractor = LandmarkFeatureExtractor()
        self.create_directories()
    
    def create_directories(self):
        """Create directories for each class"""
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(self.data_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
        print(f"✓ Created directories in {self.data_dir}")
    
    def collect_class_samples(self, class_name, num_samples=50):
        """
        Collect landmark samples for one class
        
        Args:
            class_name: Name of pose class
            num_samples: Number of samples to collect
        """
        if class_name not in CLASS_NAMES:
            print(f"Error: Unknown class '{class_name}'")
            return
        
        class_dir = os.path.join(self.data_dir, class_name)
        
        # Check existing samples
        existing = len([f for f in os.listdir(class_dir) 
                       if f.endswith('.npy')])
        
        print("\n" + "=" * 60)
        print(f"COLLECTING LANDMARK DATA: {class_name.upper()}")
        print("=" * 60)
        print(f"Target: {num_samples} samples")
        print(f"Existing: {existing}")
        print(f"\nInstructions:")
        print(f"  1. Perform '{class_name}' pose")
        print(f"  2. Press SPACE to capture")
        print(f"  3. Hold pose and vary slightly")
        print(f"  4. Press 'Q' to quit")
        print("=" * 60)
        
        input("\nPress ENTER to start...")
        
        # Initialize pose detection
        pose_queue = Queue(maxsize=10)
        detector = PoseDetectionSystem(pose_queue)
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        count = existing
        target = existing + num_samples
        
        print("\nWebcam opened. Start performing the pose!")
        
        while count < target:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect players and extract landmarks
            p1_bbox, p2_bbox = detector.detect_players(frame)
            landmarks = detector.extract_landmarks(frame, p1_bbox)
            
            # Display
            display = frame.copy()
            
            # Progress bar
            progress = (count / target) * 100
            cv2.rectangle(display, (10, 10), (10 + int(progress * 6), 40),
                         (0, 255, 0), -1)
            cv2.rectangle(display, (10, 10), (610, 40), (255, 255, 255), 2)
            
            # Info text
            info = [
                f"Class: {class_name.upper()}",
                f"Samples: {count}/{target}",
                f"Landmarks: {'YES' if landmarks else 'NO'}",
                "",
                "SPACE: Capture | Q: Quit"
            ]
            
            for i, text in enumerate(info):
                color = (0, 255, 0) if i < 3 else (255, 255, 255)
                cv2.putText(display, text, (10, 70 + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw bbox if detected
            if p1_bbox is not None:
                x1, y1, x2, y2 = map(int, p1_bbox)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display, "Detected", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            cv2.imshow('Landmark Collection', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' ') and landmarks is not None:
                # Extract features
                features = self.feature_extractor.extract_features(landmarks)
                
                if features is not None:
                    # Save as numpy file
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{class_name}_{timestamp}.npy"
                    filepath = os.path.join(class_dir, filename)
                    
                    np.save(filepath, features)
                    count += 1
                    print(f"✓ Saved {count}/{target}: {filename}")
                    
                    # Flash effect
                    white = np.ones_like(frame) * 255
                    cv2.imshow('Landmark Collection', white)
                    cv2.waitKey(100)
                else:
                    print("❌ Failed to extract features")
            
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        final_count = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
        print(f"\n✓ Collection complete for '{class_name}'")
        print(f"  Total samples: {final_count}")
    
    def collect_all_classes(self, samples_per_class=50):
        """Collect samples for all classes"""
        print("\n" + "=" * 60)
        print("LANDMARK DATA COLLECTION")
        print("=" * 60)
        print(f"Classes: {', '.join(CLASS_NAMES)}")
        print(f"Samples per class: {samples_per_class}")
        print("=" * 60)
        
        for i, class_name in enumerate(CLASS_NAMES):
            print(f"\n[{i+1}/{len(CLASS_NAMES)}] Collecting '{class_name}'...")
            self.collect_class_samples(class_name, samples_per_class)
            
            if i < len(CLASS_NAMES) - 1:
                print(f"\nGet ready for next pose: '{CLASS_NAMES[i+1]}'")
                input("Press ENTER when ready...")
        
        print("\n" + "=" * 60)
        print("✓ DATA COLLECTION COMPLETE!")
        print("=" * 60)
        self.print_dataset_summary()
    
    def print_dataset_summary(self):
        """Print summary of collected dataset"""
        print("\nDataset Summary:")
        total = 0
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(self.data_dir, class_name)
            count = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
            print(f"  {class_name}: {count} samples")
            total += count
        print(f"\nTotal: {total} samples")


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect landmark data')
    parser.add_argument('--class', dest='class_name', type=str,
                       help='Class name or "all"')
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of samples per class')
    
    args = parser.parse_args()
    
    collector = LandmarkDataCollector()
    
    if args.class_name == 'all' or args.class_name is None:
        collector.collect_all_classes(samples_per_class=args.samples)
    else:
        collector.collect_class_samples(args.class_name, num_samples=args.samples)

