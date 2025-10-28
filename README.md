# Pose Fighters - Gesture-Controlled Fighting Game

Real-time fighting game controlled by body gestures using Deep Learning.

**Tech**: PyTorch + YOLO + MediaPipe + Pygame

---

## Quick Start

### 1. Install Dependencies (5 min)

```bash
pip install torch torchvision opencv-python mediapipe ultralytics pygame numpy scikit-learn matplotlib seaborn tqdm pillow
```

### 2. Test Each Component

```bash
# Test T1 (Pose Detection)
python3 test_t1.py

# Test T2 (Game Engine)  
python3 test_t2.py

# Test T3 (ML Classification) - after training
python3 test_t3.py
```

### 3. Train T3 Model

```bash
# Convert images to landmarks (1 min)
python3 -m pose_classification.convert_images_to_landmarks

# Train model (2-3 min)
python3 -m pose_classification.landmark_train --epochs 30
```

### 4. Run Full System

```bash
python3 main_landmark_integration.py
```

---

## Project Structure

```
pose-fighters/
├── test_t1.py                  # Test T1 standalone
├── test_t2.py                  # Test T2 standalone
├── test_t3.py                  # Test T3 standalone
├── main_landmark_integration.py # Full system
│
├── pose_detection/             # T1: YOLO + MediaPipe
│   ├── pose_config.py
│   └── pose_detector.py
│
├── pose_classification/        # T3: Landmark ML
│   ├── landmark_config.py
│   ├── landmark_features.py
│   ├── landmark_model.py
│   ├── landmark_train.py
│   ├── landmark_classifier.py
│   ├── landmark_classification_runner.py
│   └── convert_images_to_landmarks.py
│
├── game/                       # T2: Game Engine
│   ├── game_engine.py
│   └── states.py
│
└── [other game files...]
```

---

## Testing Guide

### T1 (Pose Detection)

```bash
python3 test_t1.py
```

**Expected:**
- ✅ Webcam opens
- ✅ Bounding boxes on people
- ✅ Blue (P1) on left, Red (P2) on right
- ✅ FPS: ~30

### T2 (Game Engine)

```bash
python3 test_t2.py
```

**Expected:**
- ✅ Game window opens
- ✅ Menu appears
- ✅ Press SPACE → Battle starts
- ✅ Keyboard controls work (Q/W/E, U/I/O)

### T3 (ML Classification)

```bash
python3 test_t3.py
```

**Expected:**
- ✅ Landmark data found
- ✅ Model loaded
- ✅ Inference test passes

---

## Training Workflow

### Step 1: Collect Data (if you don't have images)

```bash
# Collect 50 samples per pose (20 min)
python3 -m pose_classification.landmark_data_collector --class all --samples 50
```

### Step 2: Convert Images to Landmarks (if you have images)

```bash
# Convert existing JPEG images (1 min)
python3 -m pose_classification.convert_images_to_landmarks \
    --input-dir data/pose_dataset_3class \
    --output-dir data/landmark_dataset
```

### Step 3: Train

```bash
# Train landmark-based model (2-3 min)
python3 -m pose_classification.landmark_train --epochs 30
```

**Output:**
```
Epoch 30/30: Train: 99% | Val: 96%
Test Accuracy: 95%
✓ Model saved to: models/landmark_classifier.pth
```

---

## The 3 Poses

| Pose | Description |
|------|-------------|
| 🛡️ **Block** | Arms crossed at chest (defensive) |
| 🔥 **Fireball** | Both arms extended forward (Kamehameha) |
| ⚡ **Lightning** | Both arms raised above head (Y-shape) |

---

## Keyboard Controls (Fallback)

**Player 1:**
- Q: Block
- W: Fireball  
- E: Lightning

**Player 2:**
- U: Block
- I: Fireball
- O: Lightning

**Both:**
- ESC: Quit

---

## Troubleshooting

### T1 Issues

**Problem:** Webcam not opening
```python
# In pose_detection/pose_config.py
CAMERA_INDEX = 1  # Try 0, 1, or 2
```

**Problem:** Low FPS
```python
# In pose_detection/pose_config.py
SKIP_FRAMES = 1  # Process every other frame
```

### T2 Issues

**Problem:** Import errors
```bash
# Create __init__.py files
touch game/__init__.py entities/__init__.py combat/__init__.py
```

### T3 Issues

**Problem:** No landmark data
```bash
# Convert images first
python3 -m pose_classification.convert_images_to_landmarks
```

**Problem:** Model not found
```bash
# Train first
python3 -m pose_classification.landmark_train --epochs 30
```

**Problem:** Low accuracy (<80%)
```bash
# Collect more data
python3 -m pose_classification.landmark_data_collector --class all --samples 100
# Train longer
python3 -m pose_classification.landmark_train --epochs 50
```

---

## System Requirements

- **Python**: 3.8+
- **Webcam**: Required
- **GPU**: Optional (CUDA speeds up training)
- **OS**: Linux, macOS, or Windows

---

## Performance

| Metric | Value |
|--------|-------|
| Training Time | 2-3 min |
| Inference Speed | 1-2ms |
| Game FPS | 30 |
| Model Size | 0.5MB |
| Accuracy | 95-97% |

---

## Features

✅ Real-time pose detection (30 FPS)  
✅ General model (works for any person)  
✅ 2-player simultaneous gameplay  
✅ ML-powered attack recognition  
✅ Fast inference (1-2ms)  
✅ Keyboard fallback controls  

---

## Technical Details

**T1 (Pose Detection):**
- YOLOv8 for person detection
- MediaPipe for pose estimation
- 33 landmarks per person

**T2 (Game Engine):**
- Pygame for rendering
- 30 FPS game loop
- Particle effects system

**T3 (ML Classification):**
- PyTorch MLP (3 hidden layers)
- 16 engineered features
- Transfer learning ready
- 95%+ accuracy

---

## Academic Coverage

✅ PyTorch neural networks  
✅ Feature engineering  
✅ Transfer learning  
✅ Batch normalization  
✅ Dropout regularization  
✅ Optimization (Adam, SGD)  
✅ Metrics (Precision, Recall, F1)  
✅ Real-time inference  

---

## Quick Commands

```bash
# Install
pip install torch torchvision opencv-python mediapipe ultralytics pygame numpy scikit-learn matplotlib seaborn tqdm pillow

# Test components
python3 test_t1.py
python3 test_t2.py
python3 test_t3.py

# Train T3
python3 -m pose_classification.convert_images_to_landmarks
python3 -m pose_classification.landmark_train --epochs 30

# Run full system
python3 main_landmark_integration.py
```

---

## Team Members

- **T1**: Pose Detection (YOLO + MediaPipe)
- **T2**: Game Engine (Pygame)
- **T3**: ML Classification (PyTorch)

---

