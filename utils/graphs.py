"""
Plotting and Visualization Functions
===================================

This module contains functions for creating training plots, confusion matrices,
and other visualizations for the temple image classification project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix


def plot_training_history(train_losses, val_losses, train_accs, val_accs, f1_scores, save_dir):
    """
    Create comprehensive training history plots.
    
    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        train_accs (list): Training accuracies per epoch
        val_accs (list): Validation accuracies per epoch
        f1_scores (list): F1 scores per epoch
        save_dir (str): Directory to save plots
    """
    # Create comprehensive training plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss plot
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[0, 1].plot(train_accs, label='Train Acc', linewidth=2)
    axes[0, 1].plot(val_accs, label='Val Acc', linewidth=2)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score plot
    axes[0, 2].plot(f1_scores, label='Weighted F1', linewidth=2, color='green')
    axes[0, 2].set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('F1 Score', fontsize=12)
    axes[0, 2].legend(fontsize=10)
    axes[0, 2].grid(True, alpha=0.3)
    
    # Loss convergence (log scale)
    axes[1, 0].semilogy(train_losses, label='Train Loss', linewidth=2)
    axes[1, 0].semilogy(val_losses, label='Val Loss', linewidth=2)
    axes[1, 0].set_title('Loss Convergence (Log Scale)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss (Log Scale)', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy convergence
    axes[1, 1].plot(train_accs, label='Train Acc', linewidth=2)
    axes[1, 1].plot(val_accs, label='Val Acc', linewidth=2)
    axes[1, 1].set_title('Accuracy Convergence', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 100)
    
    # F1 Score convergence
    axes[1, 2].plot(f1_scores, label='Weighted F1', linewidth=2, color='green')
    axes[1, 2].set_title('F1 Score Convergence', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Epoch', fontsize=12)
    axes[1, 2].set_ylabel('F1 Score', fontsize=12)
    axes[1, 2].legend(fontsize=10)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save individual plots
    # Loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Train Acc', linewidth=2)
    plt.plot(val_accs, label='Val Acc', linewidth=2)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # F1 Score plot
    plt.figure(figsize=(10, 6))
    plt.plot(f1_scores, label='Weighted F1', linewidth=2, color='green')
    plt.title('Validation F1 Score', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_score_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """
    Create and save confusion matrix plot.
    
    Args:
        y_true (list): True labels
        y_pred (list): Predicted labels
        class_names (list): Names of classes
        save_dir (str): Directory to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show() 