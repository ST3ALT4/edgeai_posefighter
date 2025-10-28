"""
landmark_train.py - Training script for landmark-based classifier
Much faster than image-based! Works for any person!
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import os
from tqdm import tqdm

from .landmark_model import LandmarkMLP, get_optimizer
from .landmark_dataset import create_data_loaders
from .metrics import MetricsCalculator, evaluate_model
from .landmark_config import *


class LandmarkTrainer:
    """
    Training pipeline for landmark-based classifier
    Fast and general!
    """
    
    def __init__(self, data_dir=DATA_DIR):
        print("\n" + "=" * 60)
        print("LANDMARK-BASED TRAINING")
        print("=" * 60)
        print(f"Device: {DEVICE}")
        print(f"Features: {FEATURE_DIM}")
        print(f"Classes: {NUM_CLASSES}")
        print("=" * 60)
        
        # Create model
        self.model = LandmarkMLP().to(DEVICE)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = get_optimizer(self.model, OPTIMIZER, LEARNING_RATE)
        
        # Scheduler
        if USE_SCHEDULER:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )
        else:
            self.scheduler = None
        
        # Data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(data_dir)
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        self.best_val_acc = 0.0
    
    def train_one_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        
        for features, labels in pbar:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)
            
            self.optimizer.zero_grad()
            
            # Forward
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            # Stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss / (pbar.n + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, epochs=EPOCHS):
        """Complete training loop"""
        print("\nSTARTING TRAINING")
        print("=" * 60)
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()
            
            if self.scheduler:
                self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(MODEL_SAVE_PATH)
                print(f"  ✓ New best model! (Val Acc: {val_acc:.2f}%)")
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
    
    def save_model(self, path):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }, path)
    
    def evaluate_test_set(self):
        """Evaluate on test set"""
        print("\n" + "=" * 60)
        print("TEST SET EVALUATION")
        print("=" * 60)
        
        metrics = evaluate_model(
            self.model, self.test_loader,
            DEVICE, CLASS_NAMES
        )
        
        print(f"\nTest Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics


def main():
    """Main training function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train landmark classifier')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    
    args = parser.parse_args()
    
    trainer = LandmarkTrainer(data_dir=args.data_dir)
    trainer.train(epochs=args.epochs)
    trainer.evaluate_test_set()
    
    print(f"\n✓ Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()

