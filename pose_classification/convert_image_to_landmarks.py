"""
convert_images_to_landmarks.py - Convert existing JPEG images to landmark features
Uses MediaPipe to extract landmarks from your collected images
"""

import cv2
import numpy as np
import os
from tqdm import tqdm
import mediapipe as mp

from landmark_features import LandmarkFeatureExtractor
from landmark_config import CLASS_NAMES


class ImageToLandmarkConverter:
    """
    Convert existing JPEG images to landmark .npy files
    Extracts MediaPipe landmarks from images
    """
    
    def __init__(self, image_dir, output_dir):
        """
        Args:
            image_dir: Directory with images (e.g., data/pose_dataset_3class)
            output_dir: Directory to save landmarks (e.g., data/landmark_dataset)
        """
        self.image_dir = image_dir
        self.output_dir = output_dir
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,  # For images (not video)
            model_complexity=1,
            min_detection_confidence=0.5
        )
        
        # Feature extractor
        self.feature_extractor = LandmarkFeatureExtractor()
        
        print("✓ Image to Landmark Converter initialized")
    
    def extract_landmarks_from_image(self, image_path):
        """
        Extract MediaPipe landmarks from one image
        
        Args:
            image_path: Path to JPEG image
        
        Returns:
            Dictionary with landmarks (same format as T1) or None
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Failed to read: {image_path}")
            return None
        
        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb)
        
        if not results.pose_landmarks:
            print(f"⚠️ No landmarks found: {image_path}")
            return None
        
        landmarks = results.pose_landmarks.landmark
        
        # Extract same format as T1 pose detector
        left_hip = landmarks
        right_hip = landmarks
        hip_x = (left_hip.x + right_hip.x) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        
        landmarks_dict = {
            'hip_x': float(hip_x),
            'hip_y': float(hip_y),
            'left_shoulder_x': float(landmarks.x),
            'left_shoulder_y': float(landmarks.y),
            'right_shoulder_x': float(landmarks.x),
            'right_shoulder_y': float(landmarks.y),
            'left_elbow_x': float(landmarks.x),
            'left_elbow_y': float(landmarks.y),
            'right_elbow_x': float(landmarks.x),
            'right_elbow_y': float(landmarks.y),
            'left_wrist_x': float(landmarks.x),
            'left_wrist_y': float(landmarks.y),
            'right_wrist_x': float(landmarks.x),
            'right_wrist_y': float(landmarks.y),
            'left_knee_y': float(landmarks.y),
            'right_knee_y': float(landmarks.y),
        }
        
        return landmarks_dict
    
    def convert_class(self, class_name):
        """
        Convert all images for one class
        
        Args:
            class_name: Name of class (e.g., 'block', 'fireball', 'lightning')
        """
        input_dir = os.path.join(self.image_dir, class_name)
        output_dir = os.path.join(self.output_dir, class_name)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(input_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"\nConverting {class_name}: {len(image_files)} images")
        
        success_count = 0
        fail_count = 0
        
        for img_file in tqdm(image_files, desc=f"  {class_name}"):
            img_path = os.path.join(input_dir, img_file)
            
            # Extract landmarks
            landmarks_dict = self.extract_landmarks_from_image(img_path)
            
            if landmarks_dict is None:
                fail_count += 1
                continue
            
            # Extract features (16 values)
            features = self.feature_extractor.extract_features(landmarks_dict)
            
            if features is None:
                fail_count += 1
                continue
            
            # Save as .npy file
            npy_filename = os.path.splitext(img_file) + '.npy'
            npy_path = os.path.join(output_dir, npy_filename)
            np.save(npy_path, features)
            
            success_count += 1
        
        print(f"  ✓ Success: {success_count}/{len(image_files)}")
        if fail_count > 0:
            print(f"  ⚠️ Failed: {fail_count}/{len(image_files)}")
    
    def convert_all_classes(self):
        """Convert all classes"""
        print("\n" + "=" * 60)
        print("CONVERTING IMAGES TO LANDMARKS")
        print("=" * 60)
        print(f"Input: {self.image_dir}")
        print(f"Output: {self.output_dir}")
        print("=" * 60)
        
        total_success = 0
        total_fail = 0
        
        for class_name in CLASS_NAMES:
            input_class_dir = os.path.join(self.image_dir, class_name)
            
            if not os.path.exists(input_class_dir):
                print(f"\n⚠️ Directory not found: {input_class_dir}")
                continue
            
            self.convert_class(class_name)
        
        print("\n" + "=" * 60)
        print("✓ CONVERSION COMPLETE!")
        print("=" * 60)
        
        # Print summary
        print("\nSummary:")
        for class_name in CLASS_NAMES:
            output_dir = os.path.join(self.output_dir, class_name)
            if os.path.exists(output_dir):
                count = len([f for f in os.listdir(output_dir) if f.endswith('.npy')])
                print(f"  {class_name}: {count} landmark files")
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'pose'):
            self.pose.close()


def main():
    """Main conversion function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert existing JPEG images to landmark features'
    )
    parser.add_argument(
        '--input-dir', 
        type=str, 
        default='data/pose_dataset_3class',
        help='Directory with JPEG images'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/landmark_dataset',
        help='Directory to save landmark .npy files'
    )
    
    args = parser.parse_args()
    
    # Check input directory exists
    if not os.path.exists(args.input_dir):
        print(f"❌ Error: Input directory not found: {args.input_dir}")
        print(f"\nPlease specify the correct directory with --input-dir")
        print(f"Example: python3 convert_images_to_landmarks.py --input-dir data/my_images")
        return
    
    # Convert
    converter = ImageToLandmarkConverter(args.input_dir, args.output_dir)
    converter.convert_all_classes()
    
    print(f"\n✓ Landmark features saved to: {args.output_dir}")
    print(f"✓ Now you can train: python3 -m pose_classification.landmark_train")


if __name__ == "__main__":
    main()

