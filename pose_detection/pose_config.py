"""
pose_config.py - Configuration for T1 pose detection
"""

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

# YOLO settings
YOLO_MODEL = "yolov8n.pt"  # Will auto-download on first run
YOLO_CONFIDENCE = 0.5
PERSON_CLASS_ID = 0

# MediaPipe settings
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5
MEDIAPIPE_MODEL_COMPLEXITY = 1

# Player assignment
SCREEN_CENTER_X = CAMERA_WIDTH // 2

# Performance
SHOW_DEBUG_WINDOW = True  # Set to False to hide webcam window
SKIP_FRAMES = 0  # 0 = process every frame

