"""
Image Classification Training Script
==========================================

This script provides a comprehensive training pipeline for temple image classification
using ResNet50 with advanced training strategies including staged unfreezing and different loss functions.

Features:
- ResNet50 model architecture
- Staged unfreezing strategies for transfer learning
- Weighted Cross Entropy and Focal Loss support
- Comprehensive training monitoring and visualization
- Early stopping and model checkpointing
- Detailed logging and experiment tracking

Examples:
  python train.py --model resnet50 --batch_size 32 --epochs 50
  python train.py --model resnet50 --loss focalloss --unfreeze 2
  python train.py --model resnet50 --fc_layers 512 256 --unfreeze 1
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report
import argparse
import json
import warnings
import datetime
import os

from utils import (
    FocalLoss, plot_training_history, plot_confusion_matrix,
    get_model, set_trainable_layers, train_epoch, validate_epoch
)
from utils.dataset import (
    TempleDataset, get_transforms, load_processed_dataset,
    calculate_class_weights, get_weighted_sampler
)

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")


def train_model(args):
    """
    Main training function.
    
    This function orchestrates the entire training process including:
    - Dataset loading and preprocessing
    - Model initialization and configuration
    - Training loop with staged unfreezing
    - Evaluation and result saving
    
    Args:
        args: Parsed command line arguments
    """
    
    # ============================================================================
    # EXPERIMENT SETUP
    # ============================================================================
    print("🚀 Starting Temple Image Classification Training")
    print("=" * 60)
    
    # Create experiment directory
    if args.experiment_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{args.model}_unfreeze{args.unfreeze}_{timestamp}"
    else:
        experiment_name = args.experiment_name
    
    experiment_dir = os.path.join(args.save_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create subdirectories
    models_dir = os.path.join(experiment_dir, 'models')
    plots_dir = os.path.join(experiment_dir, 'plots')
    logs_dir = os.path.join(experiment_dir, 'logs')
    
    for dir_path in [models_dir, plots_dir, logs_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"📁 Experiment directory: {experiment_dir}")
    
    # ============================================================================
    # DATASET LOADING
    # ============================================================================
    print("\n📊 Loading dataset...")
    try:
        train_image_paths, train_labels, class_names, class_mapping = load_processed_dataset(args.data_dir, 'train')
        val_image_paths, val_labels, _, _ = load_processed_dataset(args.data_dir, 'valid')
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    num_classes = len(class_names)
    print(f"✅ Dataset loaded successfully!")
    print(f"   📈 Training images: {len(train_image_paths)}")
    print(f"   📊 Validation images: {len(val_image_paths)}")
    print(f"   🏷️  Number of classes: {num_classes}")
    
    # Display class distribution
    print("\n📋 Class distribution:")
    unique, counts = np.unique(train_labels, return_counts=True)
    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        print(f"   {class_names[class_idx]}: {count} images")
    
    # ============================================================================
    # DATA PREPROCESSING
    # ============================================================================
    print("\n🔄 Setting up data preprocessing...")
    
    # Calculate class weights for imbalanced dataset
    class_weights = calculate_class_weights(train_labels, num_classes).to(device)
    print(f"   ⚖️  Class weights: {class_weights.cpu().numpy()}")
    
    # Create datasets and dataloaders
    transform = get_transforms()
    train_dataset = TempleDataset(train_image_paths, train_labels, transform)
    val_dataset = TempleDataset(val_image_paths, val_labels, transform)
    
    # Create weighted sampler for balanced training
    weighted_sampler = get_weighted_sampler(train_labels, class_weights)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        sampler=weighted_sampler, 
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True
    )
    
    # ============================================================================
    # MODEL INITIALIZATION
    # ============================================================================
    print(f"\n�� Initializing {args.model} model...")
    try:
        model = get_model(args.model, num_classes, fc_layers=args.fc_layers, device=device)
        # Add device attribute to model for training functions
        model.device = device
        print(f"   ✅ Model initialized successfully!")
        if args.fc_layers:
            print(f"   🧠 Custom FC layers: {args.fc_layers}")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return
    
    # ============================================================================
    # LOSS FUNCTION AND OPTIMIZER
    # ============================================================================
    print("\n⚙️  Setting up loss function and optimizer...")
    
    # Set loss function
    if args.loss == 'focalloss':
        criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
        print(f"   🎯 Using Focal Loss (gamma={args.gamma})")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("   🎯 Using Weighted Cross Entropy Loss")
    
    # Set optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, verbose=True
    )
    
    print(f"   📈 Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"   📉 Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)")
    
    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================
    print(f"\n🎯 Training configuration:")
    print(f"   🔄 Unfreeze mode: {args.unfreeze}")
    print("   📚 Unfreeze modes explanation:")
    print("      0 - Only classifier layers trainable")
    print("      1 - Unfreezes deepest backbone block at stage 2 (after 15 epochs)")
    print("      2 - Unfreezes stage 2 after 15 and stage 3 after 30 epochs")
    print("      3 - All layers trainable from start")
    
    # Initialize tracking variables
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    f1_scores = []
    best_f1 = 0.0
    patience = 10
    patience_counter = 0
    
    # ============================================================================
    # TRAINING LOOP
    # ============================================================================
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(args.epochs):
        # Determine current stage based on unfreeze mode
        if args.unfreeze == 0:
            stage = 1  # Always stage 1 (only classifier)
        elif args.unfreeze == 1:
            stage = 2 if epoch >= 15 else 1
        elif args.unfreeze == 2:
            if epoch < 15:
                stage = 1
            elif epoch < 30:
                stage = 2
            else:
                stage = 3
        else:  # unfreeze == 3
            stage = 3  # Always stage 3 (all layers)
        
        # Set trainable layers for current stage
        set_trainable_layers(model, stage, args.unfreeze)
        
        # Adjust learning rate based on stage changes
        if args.unfreeze in [1, 2] and epoch in [15, 30]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
            print(f"   📉 Learning rate lowered to {optimizer.param_groups[0]['lr']:.2e}")
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Training and validation
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, f1_weighted, f1_macro, predictions, true_labels = validate_epoch(
            model, val_loader, criterion)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        f1_scores.append(f1_weighted)
        
        # Print epoch results
        print(f"\n📊 Epoch {epoch+1}/{args.epochs}")
        print(f"   🎯 Stage: {stage}, 📈 LR: {current_lr:.2e}")
        print(f"   🏋️  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   ✅ Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"   🎯 F1 Weighted: {f1_weighted:.4f}, F1 Macro: {f1_macro:.4f}")
        print("-" * 60)
        
        # Learning rate scheduling
        scheduler.step(f1_weighted)
        
        # Model checkpointing
        if f1_weighted > best_f1:
            best_f1 = f1_weighted
            patience_counter = 0
            # Save model state dict in device-agnostic way
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_f1': best_f1,
                'class_names': class_names,
                'class_mapping': class_mapping,
                'unfreeze_mode': args.unfreeze,
                'model_name': args.model
            }, os.path.join(models_dir, f'best_model_{args.model}_unfreeze{args.unfreeze}.pth'))
            print(f"   💾 New best model saved with F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"   ⏹️  Early stopping triggered after {patience} epochs without improvement")
            break
    
    # ============================================================================
    # FINAL EVALUATION
    # ============================================================================
    print("\n" + "="*60)
    print("🏁 FINAL RESULTS")
    print("="*60)
    
    # Load best model and evaluate
    checkpoint = torch.load(os.path.join(models_dir, f'best_model_{args.model}_unfreeze{args.unfreeze}.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    val_loss, val_acc, f1_weighted, f1_macro, predictions, true_labels = validate_epoch(
        model, val_loader, criterion)
    
    print(f"🤖 Model: {args.model}")
    print(f"🔄 Unfreeze Mode: {args.unfreeze}")
    print(f"🎯 Loss Function: {args.loss}")
    print(f"🏆 Best Validation F1 (Weighted): {best_f1:.4f}")
    print(f"📊 Final Validation Accuracy: {val_acc:.2f}%")
    print(f"🎯 Final F1 Macro: {f1_macro:.4f}")
    
    # Print classification report
    print("\n📋 Classification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names))
    
    # ============================================================================
    # SAVE RESULTS
    # ============================================================================
    print("\n💾 Saving results...")
    
    # Save classification report
    with open(os.path.join(logs_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(true_labels, predictions, target_names=class_names))
    
    # Create and save plots
    plot_training_history(train_losses, val_losses, train_accs, val_accs, f1_scores, plots_dir)
    plot_confusion_matrix(true_labels, predictions, class_names, plots_dir)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'f1_scores': f1_scores,
        'final_metrics': {
            'best_f1_weighted': best_f1,
            'final_accuracy': val_acc,
            'final_f1_macro': f1_macro
        },
        'config': {
            'model': args.model,
            'unfreeze_mode': args.unfreeze,
            'loss_function': args.loss,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'fc_layers': args.fc_layers
        }
    }
    
    with open(os.path.join(logs_dir, f'training_history_{args.model}_unfreeze{args.unfreeze}.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save training configuration
    config = {
        'model': args.model,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'loss_function': args.loss,
        'unfreeze_mode': args.unfreeze,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'gamma': args.gamma,
        'fc_layers': args.fc_layers,
        'data_dir': args.data_dir,
        'experiment_name': experiment_name,
        'num_classes': num_classes,
        'class_names': class_names,
        'final_metrics': {
            'best_f1_weighted': best_f1,
            'final_accuracy': val_acc,
            'final_f1_macro': f1_macro
        }
    }
    
    with open(os.path.join(logs_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print(f"\n🎉 Training completed successfully!")
    print(f"📁 Experiment directory: {experiment_dir}")
    print(f"🤖 Models saved to: {models_dir}")
    print(f"📊 Plots saved to: {plots_dir}")
    print(f"📝 Logs saved to: {logs_dir}")
    print("=" * 60)


def main():
    """
    Main function that handles argument parsing and calls the training function.
    """
    # ============================================================================
    # ARGUMENT PARSING
    # ============================================================================
    parser = argparse.ArgumentParser(
        description='🏛️ Temple Image Classifier Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='processed_dataset', 
                       help='Path to processed dataset directory')
    
    # Model arguments
    parser.add_argument('--model', type=str, 
                       choices=['resnet50'], 
                       default='resnet50', help='Model architecture (only ResNet50 supported)')
    parser.add_argument('--fc_layers', type=int, nargs='+', default=None,
                       help='Custom FC layer sizes (e.g., 512 256 for 2 hidden layers)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    # Loss and optimization arguments
    parser.add_argument('--loss', type=str, choices=['weightedce', 'focalloss'], 
                       default='weightedce', help='Loss function')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma parameter')
    
    # Transfer learning arguments
    parser.add_argument('--unfreeze', type=int, choices=[0, 1, 2, 3], 
                       default=0, help='Unfreezing strategy (0-3)')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./models', 
                       help='Directory to save models and results')
    parser.add_argument('--experiment_name', type=str, default=None, 
                       help='Name for this training experiment')
    
    args = parser.parse_args()
    
    # Call the training function
    train_model(args)


if __name__ == "__main__":
    main() 