"""
metrics.py - Evaluation Metrics for Pose Classification
SYLLABUS COVERAGE: Metrics (Precision, Recall, F1, Confusion Matrix)
"""

import torch
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from config import CLASS_NAMES


class MetricsCalculator:
    """
    Calculate and display classification metrics
    
    SYLLABUS COVERAGE: Precision, Recall, F1-Score, Confusion Matrix
    """
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, average='weighted'):
        """
        Calculate precision, recall, and F1-score
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            average: 'weighted', 'macro', or 'micro'
        
        Returns:
            Dictionary with metrics
        """
        metrics = {
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0),
            'accuracy': np.mean(np.array(y_true) == np.array(y_pred))
        }
        
        return metrics
    
    @staticmethod
    def print_classification_report(y_true, y_pred, class_names=CLASS_NAMES):
        """
        Print detailed classification report
        
        Shows per-class precision, recall, F1
        """
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0
        )
        print(report)
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, class_names=CLASS_NAMES, save_path=None):
        """
        Plot and save confusion matrix
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            class_names: List of class names
            save_path: Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✓ Confusion matrix saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_training_history(history, save_path=None):
        """
        Plot training and validation loss/accuracy curves
        
        Args:
            history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(history['val_loss'], label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Acc', marker='o')
        ax2.plot(history['val_acc'], label='Val Acc', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✓ Training history saved to {save_path}")
        
        plt.show()


def evaluate_model(model, data_loader, device, class_names=CLASS_NAMES):
    """
    Evaluate model on a dataset and print metrics
    
    Args:
        model: PyTorch model
        data_loader: DataLoader
        device: torch.device
        class_names: List of class names
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Calculate metrics
    metrics_calc = MetricsCalculator()
    metrics = metrics_calc.calculate_metrics(y_true, y_pred)
    
    # Print report
    metrics_calc.print_classification_report(y_true, y_pred, class_names)
    
    # Plot confusion matrix
    metrics_calc.plot_confusion_matrix(y_true, y_pred, class_names)
    
    return metrics

