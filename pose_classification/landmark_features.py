"""
landmark_features.py - Extract features from MediaPipe landmarks
Person-independent geometric features
"""

import numpy as np
import math


class LandmarkFeatureExtractor:
    """
    Extract pose-invariant features from landmarks
    Works for any person regardless of body size, appearance
    """
    
    @staticmethod
    def calculate_angle(p1, p2, p3):
        """
        Calculate angle at p2 between p1-p2-p3
        
        Args:
            p1, p2, p3: (x, y) tuples
        
        Returns:
            Angle in degrees (0-180)
        """
        vector1 = np.array([p1 - p2, p1 - p2])
        vector2 = np.array([p3 - p2, p3 - p2])
        
        dot = np.dot(vector1, vector2)
        mag1 = np.linalg.norm(vector1)
        mag2 = np.linalg.norm(vector2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        cos_angle = np.clip(dot / (mag1 * mag2), -1.0, 1.0)
        angle = math.acos(cos_angle)
        return math.degrees(angle)
    
    @staticmethod
    def distance(p1, p2):
        """Euclidean distance between two points"""
        return math.sqrt((p1 - p2)**2 + (p1 - p2)**2)
    
    @staticmethod
    def extract_features(landmarks_dict):
        """
        Extract 16 person-independent features
        
        Args:
            landmarks_dict: Dictionary from MediaPipe (from T1)
        
        Returns:
            numpy array of 16 features
        """
        if landmarks_dict is None:
            return None
        
        try:
            # Get landmark positions
            hip = (landmarks_dict['hip_x'], landmarks_dict['hip_y'])
            
            l_shoulder = (landmarks_dict['left_shoulder_x'], 
                         landmarks_dict['left_shoulder_y'])
            r_shoulder = (landmarks_dict['right_shoulder_x'],
                         landmarks_dict['right_shoulder_y'])
            
            l_elbow = (landmarks_dict['left_elbow_x'],
                      landmarks_dict['left_elbow_y'])
            r_elbow = (landmarks_dict['right_elbow_x'],
                      landmarks_dict['right_elbow_y'])
            
            l_wrist = (landmarks_dict['left_wrist_x'],
                      landmarks_dict['left_wrist_y'])
            r_wrist = (landmarks_dict['right_wrist_x'],
                      landmarks_dict['right_wrist_y'])
            
            # Feature 1-2: Elbow angles (key for arm extension)
            left_elbow_angle = LandmarkFeatureExtractor.calculate_angle(
                l_shoulder, l_elbow, l_wrist
            )
            right_elbow_angle = LandmarkFeatureExtractor.calculate_angle(
                r_shoulder, r_elbow, r_wrist
            )
            
            # Feature 3-4: Shoulder elevation (arms up/down)
            left_shoulder_elevation = LandmarkFeatureExtractor.calculate_angle(
                hip, l_shoulder, l_elbow
            )
            right_shoulder_elevation = LandmarkFeatureExtractor.calculate_angle(
                hip, r_shoulder, r_elbow
            )
            
            # Feature 5: Arms distance (spread apart or together)
            arms_distance = LandmarkFeatureExtractor.distance(l_wrist, r_wrist)
            
            # Feature 6: Average wrist height (arms up/down)
            avg_wrist_height = (l_wrist + r_wrist) / 2
            
            # Feature 7-8: Individual wrist heights relative to hip
            left_wrist_height = l_wrist - hip
            right_wrist_height = r_wrist - hip
            
            # Feature 9-10: Wrist distance from center
            center_x = (l_shoulder + r_shoulder) / 2
            left_wrist_center_dist = abs(l_wrist - center_x)
            right_wrist_center_dist = abs(r_wrist - center_x)
            
            # Feature 11: Shoulder width (body size normalization)
            shoulder_width = LandmarkFeatureExtractor.distance(
                l_shoulder, r_shoulder
            )
            
            # Feature 12: Arms crossed (wrists crossed over)
            arms_crossed = 1.0 if (l_wrist > r_shoulder and 
                                   r_wrist < l_shoulder) else 0.0
            
            # Feature 13: Arms horizontal spread
            arms_spread = abs(l_wrist - r_wrist)
            
            # Feature 14: Arms vertical (both up)
            arms_up = 1.0 if (l_wrist < l_shoulder and 
                             r_wrist < r_shoulder) else 0.0
            
            # Feature 15-16: Individual arm extension
            left_arm_extended = 1.0 if left_elbow_angle > 150 else 0.0
            right_arm_extended = 1.0 if right_elbow_angle > 150 else 0.0
            
            # Combine all features
            features = np.array([
                left_elbow_angle / 180.0,      # Normalize to [0, 1]
                right_elbow_angle / 180.0,
                left_shoulder_elevation / 180.0,
                right_shoulder_elevation / 180.0,
                arms_distance,
                avg_wrist_height,
                left_wrist_height,
                right_wrist_height,
                left_wrist_center_dist,
                right_wrist_center_dist,
                shoulder_width,
                arms_crossed,
                arms_spread,
                arms_up,
                left_arm_extended,
                right_arm_extended
            ], dtype=np.float32)
            
            return features
            
        except KeyError as e:
            print(f"Missing landmark: {e}")
            return None

