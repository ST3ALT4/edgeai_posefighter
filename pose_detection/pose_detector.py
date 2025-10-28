"""
pose_detector.py - Multi-person pose detection using YOLO + MediaPipe
Team Member 1 Implementation
"""

import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from multiprocessing import Process, Queue
import time
from typing import Dict, Optional
from pose_detection.pose_config import *


class PoseDetectionSystem:
    """
    Complete pose detection system for Pose Fighters
    Runs in separate process and sends data via queue
    """
    
    def __init__(self, pose_queue: Queue):
        """
        Initialize pose detection system
        
        Args:
            pose_queue: Multiprocessing queue to send pose data to game
        """
        self.pose_queue = pose_queue
        self.running = False
        
        # Initialize models
        print("Loading YOLO model...")
        self.yolo = YOLO(YOLO_MODEL)
        
        print("Loading MediaPipe Pose...")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=MEDIAPIPE_MODEL_COMPLEXITY,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        )
        
        print("✓ Pose detection initialized!")
    
    def detect_players(self, frame):
        """
        Detect two players using YOLO
        
        Returns:
            (player1_bbox, player2_bbox) or (None, None)
        """
        results = self.yolo(
            frame,
            classes=[PERSON_CLASS_ID],
            conf=YOLO_CONFIDENCE,
            verbose=False
        )
        
        if not results or not results.boxes or len(results.boxes) == 0:
            return None, None
        
        # Get all person detections
        boxes = results.boxes.xyxy.cpu().numpy()
        
        if len(boxes) == 0:
            return None, None
        
        # Calculate center x for each box
        centers = [(box, (box + box) / 2) for box in boxes]
        
        # Sort by x position (left to right)
        centers.sort(key=lambda x: x)
        
        if len(centers) == 1:
            # Only one person, assign based on position
            box, center_x = centers
            if center_x < SCREEN_CENTER_X:
                return box, None
            else:
                return None, box
        
        # Two or more people detected
        player1_box = centers  # Leftmost
        player2_box = centers  # Second from left
        
        return player1_box, player2_box
    
    def extract_landmarks(self, frame, bbox):
        """
        Extract pose landmarks for one player
        
        Args:
            frame: Full frame
            bbox: [x1, y1, x2, y2]
            
        Returns:
            Dict with normalized landmarks or None
        """
        if bbox is None:
            return None
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Add padding
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape, x2 + padding)
        y2 = min(frame.shape, y2 + padding)
        
        # Crop player region
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
        
        # Convert to RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.pose.process(rgb)
        
        if not results.pose_landmarks:
            return None
        
        landmarks = results.pose_landmarks.landmark
        
        # Extract key points (normalized 0-1)
        # Hip center for player position
        left_hip = landmarks
        right_hip = landmarks
        hip_x = (left_hip.x + right_hip.x) / 2
        hip_y = (left_hip.y + right_hip.y) / 2
        
        # Shoulders
        left_shoulder = landmarks
        right_shoulder = landmarks
        
        # Wrists for attack detection
        left_wrist = landmarks
        right_wrist = landmarks
        
        # Elbows
        left_elbow = landmarks
        right_elbow = landmarks
        
        # Knees for ground pound
        left_knee = landmarks
        right_knee = landmarks
        
        return {
            'hip_x': float(hip_x),
            'hip_y': float(hip_y),
            'left_shoulder_x': float(left_shoulder.x),
            'left_shoulder_y': float(left_shoulder.y),
            'right_shoulder_x': float(right_shoulder.x),
            'right_shoulder_y': float(right_shoulder.y),
            'left_wrist_x': float(left_wrist.x),
            'left_wrist_y': float(left_wrist.y),
            'right_wrist_x': float(right_wrist.x),
            'right_wrist_y': float(right_wrist.y),
            'left_elbow_x': float(left_elbow.x),
            'left_elbow_y': float(left_elbow.y),
            'right_elbow_x': float(right_elbow.x),
            'right_elbow_y': float(right_elbow.y),
            'left_knee_y': float(left_knee.y),
            'right_knee_y': float(right_knee.y),
        }
    
    def run(self):
        """
        Main pose detection loop - runs in separate process
        """
        self.running = True
        
        # Open webcam
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        
        if not cap.isOpened():
            print("ERROR: Could not open webcam!")
            return
        
        print("✓ Webcam opened. Starting pose detection...")
        frame_count = 0
        fps_time = time.time()
        fps = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame")
                break
            
            # Skip frames if configured
            frame_count += 1
            if SKIP_FRAMES > 0 and frame_count % (SKIP_FRAMES + 1) != 0:
                continue
            
            # Detect players
            p1_bbox, p2_bbox = self.detect_players(frame)
            
            # Extract landmarks
            player1_data = self.extract_landmarks(frame, p1_bbox)
            player2_data = self.extract_landmarks(frame, p2_bbox)
            
            # Send data to game
            pose_data = {
                'player1': player1_data,
                'player2': player2_data
            }
            
            try:
                # Non-blocking put
                if not self.pose_queue.full():
                    self.pose_queue.put_nowait(pose_data)
            except:
                pass
            
            # Show debug window
            if SHOW_DEBUG_WINDOW:
                # Draw bounding boxes
                if p1_bbox is not None:
                    x1, y1, x2, y2 = map(int, p1_bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, "P1", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
                if p2_bbox is not None:
                    x1, y1, x2, y2 = map(int, p2_bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "P2", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # Calculate FPS
                if time.time() - fps_time > 1.0:
                    fps = frame_count / (time.time() - fps_time)
                    frame_count = 0
                    fps_time = time.time()
                
                # Display FPS
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Pose Detection (T1)', frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✓ Pose detection stopped")
    
    def stop(self):
        """Stop the pose detection loop"""
        self.running = False


def start_pose_detection_process(pose_queue: Queue):
    """
    Function to run in separate process
    
    Args:
        pose_queue: Queue to send pose data to main game
    """
    detector = PoseDetectionSystem(pose_queue)
    detector.run()

