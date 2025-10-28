"""
test_t3.py - Test T3 (Landmark Classification) Standalone
Run this to test ML model training and inference
"""

import os
import numpy as np
import torch


def test_t3():
    print("\n" + "=" * 60)
    print("TESTING T3: LANDMARK CLASSIFICATION")
    print("=" * 60)
    
    # Test 1: Check if landmark files exist
    print("\n[Test 1] Checking landmark data...")
    data_dir = 'data/landmark_dataset'
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        print("\n   Run this first:")
        print("   python3 -m pose_classification.convert_images_to_landmarks")
        return
    
    # Count samples
    class_names = ['block', 'fireball', 'lightning']
    total = 0
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) if f.endswith('.npy')])
            print(f"  ✓ {class_name}: {count} samples")
            total += count
        else:
            print(f"  ❌ {class_name}: directory not found")
    
    if total == 0:
        print("\n❌ No landmark data found!")
        print("   Run: python3 -m pose_classification.convert_images_to_landmarks")
        return
    
    print(f"\n✅ Total: {total} landmark samples")
    
    # Test 2: Check if model exists
    print("\n[Test 2] Checking trained model...")
    model_path = 'models/landmark_classifier.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("\n   Train the model first:")
        print("   python3 -m pose_classification.landmark_train --epochs 30")
        return
    
    print(f"✅ Model found: {model_path}")
    
    # Test 3: Load model
    print("\n[Test 3] Loading model...")
    try:
        from pose_classification.landmark_model import LandmarkMLP
        from pose_classification.landmark_config import DEVICE
        
        model = LandmarkMLP().to(DEVICE)
        checkpoint = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        best_acc = checkpoint.get('best_val_acc', 0)
        print(f"✅ Model loaded successfully!")
        print(f"   Best validation accuracy: {best_acc:.2f}%")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test 4: Test inference on sample data
    print("\n[Test 4] Testing inference...")
    try:
        # Load one sample
        sample_class = 'block'
        sample_dir = os.path.join(data_dir, sample_class)
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.npy')]
        
        if not sample_files:
            print("❌ No sample files to test")
            return
        
        sample_path = os.path.join(sample_dir, sample_files[0])
        features = np.load(sample_path)
        features_tensor = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            outputs = model(features_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = class_names[predicted.item()]
            confidence_val = confidence.item()
        
        print(f"✅ Inference test passed!")
        print(f"   Input: {sample_class}")
        print(f"   Predicted: {predicted_class}")
        print(f"   Confidence: {confidence_val:.2f}")
        
        if predicted_class == sample_class:
            print("   ✓ Correct prediction!")
        else:
            print("   ⚠️  Incorrect prediction (may need more training)")
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # All tests passed
    print("\n" + "=" * 60)
    print("✅ ALL T3 TESTS PASSED!")
    print("=" * 60)
    print("\nT3 is ready to use in the full system!")
    print("Run: python3 main_landmark_integration.py")


if __name__ == "__main__":
    test_t3()
