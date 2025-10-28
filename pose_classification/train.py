"""
train.py - Training Script with Mixed Precision and All Optimizers
SYLLABUS COVERAGE: All training components
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import os
from tqdm import tqdm
import json

from models import PoseClassifierResNet, get_optimizer, get_scheduler
from dataset import create_data_loaders
from metrics import MetricsCalculator, evaluate_model
from config import *


class Trainer:
    """
    Complete training pipeline with mixed precision
    
    SYLLABUS COVERAGE:
    - Transfer Learning
    - Optimization (SGD, Adam, RMSprop)
    - Mixed Precision Training
    - Metrics (Precision, Recall, F1)
    """
    
    def __init__(self, data_dir, model_name='resnet18', optimizer_name='adam',
                 use_mixed_precision=USE_MIXED_PRECISION):
        """
        Initialize trainer
        
        Args:
            data_dir: Path to training data
            model_name: 'resnet18', 'resnet34', or 'resnet50'
            optimizer_name: 'sgd', 'adam', 'rmsprop', 'adamw'
            use_mixed_precision: Enable FP16 training
        """
        self.data_dir = data_dir
        self.device = DEVICE
        self.use_mixed_precision = use_mixed_precision and torch.cuda.is_available()
        
        print("\n" + "=" * 60)
        print("INITIALIZING TRAINING")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model: {model_name}")
        print(f"Optimizer: {optimizer_name}")
        print(f"Mixed Precision: {self.use_mixed_precision}")
        print("=" * 60)
        
        # Create model
        self.model = PoseClassifierResNet(
            num_classes=NUM_CLASSES,
            model_name=model_name,
            freeze_backbone=FREEZE_BACKBONE,
            dropout_rate=DROPOUT_RATE
        ).to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = get_optimizer(
            self.model, optimizer_name,
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = get_scheduler(self.optimizer) if USE_SCHEDULER else None
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(
            data_dir, batch_size=BATCH_SIZE
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
    
    def train_one_epoch(self, epoch):
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Average loss and accuracy
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.use_mixed_precision:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Scaled backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """
        Validate the model
        
        Returns:
            Average loss and accuracy
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, epochs=EPOCHS):
        """
        Complete training loop
        
        Args:
            epochs: Number of epochs to train
        """
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_one_epoch(epoch)
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups['lr']
                print(f"Learning rate: {current_lr:.6f}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(MODEL_SAVE_PATH)
                print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
    
    def save_model(self, path):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        print(f"✓ Model loaded from {path}")
        print(f"  Best Val Acc: {self.best_val_acc:.2f}%")
    
    def evaluate_test_set(self):
        """Evaluate on test set with full metrics"""
        print("\n" + "=" * 60)
        print("EVALUATING ON TEST SET")
        print("=" * 60)
        
        metrics = evaluate_model(
            self.model, self.test_loader,
            self.device, CLASS_NAMES
        )
        
        print("\nTest Set Metrics:")
        print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def plot_training_curves(self):
        """Plot training history"""
        metrics_calc = MetricsCalculator()
        metrics_calc.plot_training_history(self.history, save_path='logs/training_history.png')


# Main training script
def main():
    """
    Main training function
    
    Usage:
        python train.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train pose classifier')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR,
                       help='Path to training data')
    parser.add_argument('--model', type=str, default=MODEL_NAME,
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='Model architecture')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER,
                       choices=['sgd', 'adam', 'rmsprop', 'adamw'],
                       help='Optimizer')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Update config
    globals()['EPOCHS'] = args.epochs
    globals()['BATCH_SIZE'] = args.batch_size
    globals()['LEARNING_RATE'] = args.lr
    
    # Create trainer
    trainer = Trainer(
        data_dir=args.data_dir,
        model_name=args.model,
        optimizer_name=args.optimizer
    )
    
    # Train
    trainer.train(epochs=args.epochs)
    
    # Evaluate
    trainer.evaluate_test_set()
    
    # Plot curves
    trainer.plot_training_curves()
    
    print("\n✓ Training complete!")
    print(f"Model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()

