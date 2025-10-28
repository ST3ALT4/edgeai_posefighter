"""
data_collector.py - Collect Training Data from Webcam
SYLLABUS COVERAGE: Custom Dataset Creation
"""

import cv2
import os
from datetime import datetime
import numpy as np
from config import CLASS_NAMES


class DataCollector:
    """
    Interactive data collection tool
    Collects images from webcam for each pose class
    """
    
    def __init__(self, data_dir='data/pose_dataset'):
        """
        Args:
            data_dir: Directory to save collected data
        """
        self.data_dir = data_dir
        self.create_directories()
        
    def create_directories(self):
        """Create directories for each class"""
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(self.data_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
        print(f"✓ Created directories in {self.data_dir}")
    
    def collect_class_samples(self, class_name, num_samples=50, countdown=3):
        """
        Collect samples for one class
        
        Args:
            class_name: Name of the pose class
            num_samples: Number of samples to collect
            countdown: Countdown seconds before starting
        """
        if class_name not in CLASS_NAMES:
            print(f"Error: Unknown class '{class_name}'")
            return
        
        class_dir = os.path.join(self.data_dir, class_name)
        
        # Check existing samples
        existing = len([f for f in os.listdir(class_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        print("\n" + "=" * 60)
        print(f"COLLECTING DATA FOR: {class_name.upper()}")
        print("=" * 60)
        print(f"Target samples: {num_samples}")
        print(f"Existing samples: {existing}")
        print(f"\nInstructions:")
        print(f"  1. Position yourself to perform '{class_name}' pose")
        print(f"  2. Press SPACE to capture a sample")
        print(f"  3. Hold the pose and vary slightly for diversity")
        print(f"  4. Press 'Q' to quit early")
        print("=" * 60)
        
        input("\nPress ENTER to start...")
        
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        count = existing
        target = existing + num_samples
        
        while count < target:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display info
            display = frame.copy()
            
            # Progress bar
            progress = (count / target) * 100
            cv2.rectangle(display, (10, 10), (10 + int(progress * 6), 40), 
                         (0, 255, 0), -1)
            cv2.rectangle(display, (10, 10), (610, 40), (255, 255, 255), 2)
            
            # Text info
            info_text = [
                f"Class: {class_name.upper()}",
                f"Samples: {count}/{target}",
                f"Progress: {progress:.1f}%",
                "",
                "SPACE: Capture",
                "Q: Quit"
            ]
            
            y_offset = 70
            for i, text in enumerate(info_text):
                color = (0, 255, 0) if i < 3 else (255, 255, 255)
                cv2.putText(display, text, (10, y_offset + i * 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow('Data Collection', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to capture
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{class_name}_{timestamp}.jpg"
                filepath = os.path.join(class_dir, filename)
                
                cv2.imwrite(filepath, frame)
                count += 1
                print(f"✓ Captured {count}/{target}: {filename}")
                
                # Brief flash effect
                white = np.ones_like(frame) * 255
                cv2.imshow('Data Collection', white)
                cv2.waitKey(100)
                
            elif key == ord('q'):  # Q to quit
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        final_count = len([f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"\n✓ Collection complete for '{class_name}'")
        print(f"  Total samples: {final_count}")
    
    def collect_all_classes(self, samples_per_class=50):
        """
        Collect samples for all classes
        
        Args:
            samples_per_class: Number of samples per class
        """
        print("\n" + "=" * 60)
        print("POSE DATASET COLLECTION")
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
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  {class_name}: {count} samples")
            total += count
        print(f"\nTotal: {total} samples")


# Standalone script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Collect pose training data')
    parser.add_argument('--class', dest='class_name', type=str,
                       help='Class name to collect (or "all" for all classes)')
    parser.add_argument('--samples', type=int, default=50,
                       help='Number of samples per class (default: 50)')
    parser.add_argument('--data-dir', type=str, default='data/pose_dataset',
                       help='Directory to save data')
    
    args = parser.parse_args()
    
    collector = DataCollector(data_dir=args.data_dir)
    
    if args.class_name == 'all' or args.class_name is None:
        collector.collect_all_classes(samples_per_class=args.samples)
    else:
        collector.collect_class_samples(args.class_name, num_samples=args.samples)

