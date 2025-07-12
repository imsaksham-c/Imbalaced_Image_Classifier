import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b3, resnext50_32x4d
import timm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import argparse
import os
import cv2
from PIL import Image
import json
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TempleDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (512, 512), color='white')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_transforms():
    return transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_model(model_name, num_classes, pretrained=True):
    if model_name == 'resnet50':
        model = resnet50(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == 'resnext50':
        model = resnext50_32x4d(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(model.fc.in_features, num_classes)
        )
    elif model_name == 'efficientnet_b3':
        model = efficientnet_b3(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
    return model.to(device)

def load_processed_dataset(processed_dir, split):
    image_paths = []
    labels = []
    class_names = []
    class_mapping = {}
    split_dir = os.path.join(processed_dir, split)
    if not os.path.exists(split_dir):
        raise ValueError(f"Split directory {split_dir} does not exist.")
    for idx, class_name in enumerate(sorted(os.listdir(split_dir))):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        class_mapping[class_name] = idx
        class_names.append(class_name)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_path = os.path.join(class_dir, img_file)
                image_paths.append(image_path)
                labels.append(idx)
    return image_paths, labels, class_names, class_mapping

def calculate_class_weights(labels, num_classes):
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)
    return torch.FloatTensor(class_weights).to(device)

def get_weighted_sampler(labels, class_weights):
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(labels))

def set_trainable_layers(model, stage, unfreeze_mode):
    """
    Set trainable layers based on unfreeze mode and stage
    
    unfreeze_mode:
    0 - Only classifier layers trainable
    1 - Unfreezes deepest backbone block at stage 2 after 15 epochs
    2 - Unfreezes stage 2 after 15 and stage 3 after 30 All layers trainable
    3 - All layers trainable from start
    """
    
    if unfreeze_mode == 0:
        # Only classifier layers trainable
        for name, param in model.named_parameters():
            if 'fc' not in name and 'classifier' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    elif unfreeze_mode == 1:
        if stage == 1:
            # Only classifier layers trainable
            for name, param in model.named_parameters():
                if 'fc' not in name and 'classifier' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif stage == 2:
            # Unfreeze deepest backbone block + classifier
            for name, param in model.named_parameters():
                if 'resnet' in str(type(model)).lower() or 'resnext' in str(type(model)).lower():
                    # Unfreeze layer4 (deepest) and fc
                    if 'layer4' in name or 'fc' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                elif 'efficientnet' in str(type(model)).lower():
                    # Unfreeze features.7 (deepest) and classifier
                    if 'features.7' in name or 'classifier' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        else:  # stage 3
            # All layers trainable
            for param in model.parameters():
                param.requires_grad = True
    
    elif unfreeze_mode == 2:
        if stage == 1:
            # Only classifier layers trainable
            for name, param in model.named_parameters():
                if 'fc' not in name and 'classifier' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        elif stage == 2:
            # Unfreeze layer4 (for ResNet/ResNeXt) or features.7 (for EfficientNet) + classifier
            for name, param in model.named_parameters():
                if 'resnet' in str(type(model)).lower() or 'resnext' in str(type(model)).lower():
                    if 'layer4' in name or 'fc' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                elif 'efficientnet' in str(type(model)).lower():
                    if 'features.7' in name or 'classifier' in name:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
        else:  # stage 3
            # All layers trainable
            for param in model.parameters():
                param.requires_grad = True
    
    elif unfreeze_mode == 3:
        # All layers trainable from start
        for param in model.parameters():
            param.requires_grad = True

def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc='Training', leave=False)):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 10 == 0:
            tqdm.write(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    
    return epoch_loss, epoch_acc, f1_weighted, f1_macro, all_predictions, all_labels

def plot_training_history(train_losses, val_losses, train_accs, val_accs, f1_scores, save_dir):
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

def main():
    parser = argparse.ArgumentParser(description='Temple Image Classifier Training')
    parser.add_argument('--data_dir', type=str, default='processed_dataset', 
                       help='Path to processed dataset directory')
    parser.add_argument('--model', type=str, choices=['resnet50', 'resnext50', 'efficientnet_b3'], 
                       default='resnet50', help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--loss', type=str, choices=['weightedce', 'focalloss'], 
                       default='weightedce', help='Loss function')
    parser.add_argument('--unfreeze', type=int, choices=[0, 1, 2, 3], 
                       default=0, help='Unfreezing strategy')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--save_dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--gamma', type=float, default=2.0, help='Focal loss gamma parameter')
    parser.add_argument('--experiment_name', type=str, default=None, help='Name for this training experiment')
    
    args = parser.parse_args()
    
    # Create experiment directory
    if args.experiment_name is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"train_{timestamp}"
    else:
        experiment_name = args.experiment_name
    
    # Create experiment directory
    experiment_dir = os.path.join(args.save_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"Experiment directory created: {experiment_dir}")
    
    # Create subdirectories for different types of files
    models_dir = os.path.join(experiment_dir, 'models')
    plots_dir = os.path.join(experiment_dir, 'plots')
    logs_dir = os.path.join(experiment_dir, 'logs')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    print("Loading processed dataset...")
    train_image_paths, train_labels, class_names, class_mapping = load_processed_dataset(args.data_dir, 'train')
    val_image_paths, val_labels, _, _ = load_processed_dataset(args.data_dir, 'valid')
    
    num_classes = len(class_names)
    print(f"Total train images: {len(train_image_paths)}")
    print(f"Total val images: {len(val_image_paths)}")
    print(f"Number of classes: {num_classes}")
    
    unique, counts = np.unique(train_labels, return_counts=True)
    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        print(f"{class_names[class_idx]}: {count} images")
    
    class_weights = calculate_class_weights(train_labels, num_classes)
    print(f"Class weights: {class_weights}")
    
    transform = get_transforms()
    train_dataset = TempleDataset(train_image_paths, train_labels, transform)
    val_dataset = TempleDataset(val_image_paths, val_labels, transform)
    
    weighted_sampler = get_weighted_sampler(train_labels, class_weights)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             sampler=weighted_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"Initializing {args.model} model...")
    model = get_model(args.model, num_classes)
    
    # Set loss function
    if args.loss == 'focalloss':
        criterion = FocalLoss(alpha=class_weights, gamma=args.gamma)
        print(f"Using Focal Loss with gamma={args.gamma}")
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using Weighted Cross Entropy Loss")
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5, verbose=True)
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    f1_scores = []
    best_f1 = 0.0
    patience = 10
    patience_counter = 0
    
    print(f"Starting training with unfreeze mode: {args.unfreeze}")
    print("Unfreeze modes:")
    print("0 - Only classifier layers trainable")
    print("1 - Unfreezes deepest backbone block at stage 2 after 15 epochs")
    print("2 - Unfreezes stage 2 after 15 and stage 3 after 30 All layers trainable")
    print("3 - All layers trainable from start")
    
    for epoch in range(args.epochs):
        # Determine current stage based on unfreeze mode
        if args.unfreeze == 0:
            stage = 1  # Always stage 1 (only classifier)
        elif args.unfreeze == 1:
            if epoch < 15:
                stage = 1
            else:
                stage = 2
        elif args.unfreeze == 2:
            if epoch < 15:
                stage = 1
            elif epoch < 30:
                stage = 2
            else:
                stage = 3
        else:  # unfreeze == 3
            stage = 3  # Always stage 3 (all layers)
        
        # Set trainable layers
        set_trainable_layers(model, stage, args.unfreeze)
        
        # Adjust learning rate based on stage changes
        if args.unfreeze in [1, 2] and epoch in [15, 30]:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.1
            print(f"Learning rate lowered to {optimizer.param_groups[0]['lr']}")
        
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, f1_weighted, f1_macro, predictions, true_labels = validate_epoch(
            model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        f1_scores.append(f1_weighted)
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Stage: {stage}, LR: {current_lr:.2e}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"F1 Weighted: {f1_weighted:.4f}, F1 Macro: {f1_macro:.4f}")
        print("-" * 50)
        
        scheduler.step(f1_weighted)
        
        if f1_weighted > best_f1:
            best_f1 = f1_weighted
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'class_names': class_names,
                'class_mapping': class_mapping,
                'unfreeze_mode': args.unfreeze,
                'model_name': args.model
            }, os.path.join(models_dir, f'best_model_{args.model}_unfreeze{args.unfreeze}.pth'))
            print(f"New best model saved with F1: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Load best model and evaluate
    checkpoint = torch.load(os.path.join(models_dir, f'best_model_{args.model}_unfreeze{args.unfreeze}.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    val_loss, val_acc, f1_weighted, f1_macro, predictions, true_labels = validate_epoch(
        model, val_loader, criterion)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Unfreeze Mode: {args.unfreeze}")
    print(f"Loss Function: {args.loss}")
    print(f"Best Validation F1 (Weighted): {best_f1:.4f}")
    print(f"Final Validation Accuracy: {val_acc:.2f}%")
    print(f"Final F1 Macro: {f1_macro:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, target_names=class_names))
    
    # Save classification report to file
    with open(os.path.join(logs_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(true_labels, predictions, target_names=class_names))
    
    plot_training_history(train_losses, val_losses, train_accs, val_accs, f1_scores, plots_dir)
    plot_confusion_matrix(true_labels, predictions, class_names, plots_dir)
    
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
            'epochs': args.epochs
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
    
    print(f"\nTraining completed!")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Models saved to: {models_dir}")
    print(f"Plots saved to: {plots_dir}")
    print(f"Logs saved to: {logs_dir}")

if __name__ == "__main__":
    main() 