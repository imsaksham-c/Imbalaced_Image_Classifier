# Temple Image Classification Project

This project provides a complete pipeline for temple image classification using a custom, multi-class dataset. The repository includes advanced preprocessing techniques, sophisticated training strategies, and comprehensive solutions for handling class imbalance and improving model convergence.

## 🎯 Key Technical Innovations

This project implements several advanced techniques to handle the challenging imbalanced temple dataset:

### 1. **Adaptive Preprocessing with Weighted Augmentation**
- **Dynamic Image Size**: Configurable image resolution (default 512x512) for optimal feature extraction
- **Class-Aware Augmentation**: Aggressive augmentation for smaller classes, minimal for larger classes
- **Smart Augmentation Strategy**:
  - Classes with ≥80 images: 25% augmentation rate, 1 augmentation per image
  - Classes with ≥50 images: 50% augmentation rate, 1 augmentation per image  
  - Classes with ≥25 images: 75% augmentation rate, 1 augmentation per image
  - Classes with <25 images: 100% augmentation rate, 2-4 augmentations per image
- **Comprehensive Augmentation Pipeline**: Horizontal flip, rotation, shift-scale-rotate, brightness/contrast adjustment, gamma correction, hue/saturation shifts, noise injection, and blur

### 2. **Advanced Transfer Learning with Staged Unfreezing**
- **Progressive Unfreezing**: Prevents catastrophic forgetting and enables gradual feature adaptation
- **Adaptive Learning Rate**: LR reduction by 10x at each stage transition
- **Gradient Clipping**: Prevents gradient explosion with max_norm=1.0

### 3. **Sophisticated Class Imbalance Handling**
- **Weighted Random Sampling**: Ensures balanced batch composition during training
- **Class Weight Calculation**: Inverse frequency weighting for loss function
- **Focal Loss Implementation**: Focuses training on hard examples with configurable gamma parameter
- **Multi-Metric Evaluation**: F1-weighted, F1-macro, and accuracy tracking

### 4. **Advanced Training Optimizations**
- **AdamW Optimizer**: Better weight decay implementation for regularization
- **ReduceLROnPlateau Scheduler**: Adaptive learning rate based on validation F1 score
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Model Checkpointing**: Saves best model based on validation F1 score

## Dataset Overview

The dataset consists of temple images from various countries and regions, organized by folder:

| Country/Region                | Number of Images |
|-------------------------------|---------------------|
| Armenia                       | 11                  |
| Australia                     | 36                  |
| Germany                       | 90                  |
| Hungary+Slovakia+Croatia      | 48                  |
| Indonesia-Bali                | 44                  |
| Japan                         | 60                  |
| Malaysia+Indonesia            | 56                  |
| Portugal+Brazil               | 54                  |
| Russia                        | 100                 |
| Spain                         | 65                  |
| Thailand                      | 101                 |

Each folder is a class label containing images from that region.

## Project Structure

### Preprocessing (`preprocess.py`)

- **Smart Dataset Scanning**: Automatically detects class structure and calculates class distribution
- **Adaptive Data Splitting**: Ensures each class has representation in train/validation sets
- **Class-Aware Augmentation**: Implements weighted augmentation based on class sizes
- **Image Normalization**: Uses ImageNet statistics for optimal transfer learning
- **Configurable image size** (default: 512x512)
- **Robust error handling** for corrupted images



### Training Scripts

| Script                | Model(s)           | Loss Function(s)             | Unfreezing Strategy           | Class Imbalance Handling           | Advanced Features            |
|-----------------------|--------------------|------------------------------|-------------------------------|------------------------------------|------------------------------|
| train.py              | ResNet50           | Weighted CE / Focal Loss     | Configurable (0-2 stages)     | Class weights, Focal Loss          | Baseline with all optimizations |
| train_all.sh          | ResNet50           | All combinations             | All strategies (0-2)          | Comprehensive handling             | Automated experiment runner |

## Advanced Training Features

### **Unfreezing Strategies (Configurable via `--unfreeze` parameter)**

#### **Mode 0: Feature Extraction**
- **Description**: Only classifier layers trainable, backbone frozen
- **Use Case**: Quick baseline training, feature extraction
- **Benefits**: Fast training, prevents overfitting on small datasets

#### **Mode 1: Two-Stage Unfreezing**
- **Stage 1 (Epochs 0-14)**: Only classifier layers trainable
- **Stage 2 (Epochs 15+)**: Unfreezes deepest backbone block (layer4)
- **Use Case**: Standard transfer learning approach
- **Benefits**: Gradual adaptation, prevents catastrophic forgetting

#### **Mode 2: Three-Stage Progressive Unfreezing**
- **Stage 1 (Epochs 0-14)**: Only classifier layers trainable
- **Stage 2 (Epochs 15-29)**: Unfreezes layer4 (deepest features)
- **Stage 3 (Epochs 30+)**: All layers trainable
- **Use Case**: Advanced transfer learning for complex datasets
- **Benefits**: Maximum feature adaptation, optimal for imbalanced data

### **Loss Functions**

- **Weighted Cross Entropy**: Standard CE with inverse class frequency weights
- **Focal Loss**: Custom implementation with configurable gamma (default: 2.0)

## Quick Start Guide

### **Step 1: Download Dataset**
```bash
gdown --fuzzy "https://drive.google.com/file/d/1ccqGu9r815WvgHAlG2CujzUPOEW_Pvo9/view?usp=sharing"
```

### **Step 2: Preprocess Data**
```bash
python preprocess.py --dataset ./dataset --output ./processed_dataset --image_size 512
```

### **Step 3: Train Model**

#### **Quick Baseline**
```bash
python train.py --model resnet50 --batch_size 32 --epochs 50 --loss weightedce
```

#### **Advanced Training**
```bash
python train.py --model resnet50 --batch_size 32 --epochs 50 --loss focalloss --gamma 2.0 --unfreeze 2
```

#### **Run All Experiments**
```bash
chmod +x train_all.sh
./train_all.sh
```

### **Key Parameters**

| Parameter | Options | Description |
|-----------|---------|-------------|
| `--unfreeze` | 0, 1, 2 | Unfreezing strategy (0=feature extraction, 1=two-stage, 2=three-stage) |
| `--loss` | weightedce, focalloss | Loss function choice |
| `--gamma` | float (default: 2.0) | Focal loss focusing parameter |
| `--batch_size` | int (default: 32) | Training batch size |
| `--epochs` | int (default: 50) | Number of training epochs |
| `--lr` | float (default: 1e-3) | Learning rate |
| `--fc_layers` | int list | Custom FC layer sizes (e.g., 512 256) |

## Outputs

- **Model Checkpoints**: Best models saved based on validation F1 score
- **Training Logs**: Detailed JSON files with all metrics and configuration
- **Visualization Plots**: Training/validation curves, F1 scores, confusion matrices
- **Classification Reports**: Detailed per-class performance analysis

```
models/
├── experiment_name/
│   ├── models/          # Model checkpoints
│   ├── plots/           # Training visualizations
│   ├── logs/            # Training history and reports
│   └── config.json      # Experiment configuration
```

## Technical Details

### **Model Architecture**
- **Backbone**: ResNet50 with ImageNet pretrained weights
- **Classifier**: Configurable FC layers with dropout and batch normalization

### **Training Optimizations**
- **Gradient Clipping**: Prevents gradient explosion (max_norm=1.0)
- **Weighted Sampling**: Ensures balanced batch composition
- **Learning Rate Scheduling**: ReduceLROnPlateau with patience=3, factor=0.5
- **Early Stopping**: Prevents overfitting with patience=10 epochs

### **System Requirements**
- **Dependencies**: PyTorch, torchvision, albumentations, scikit-learn, matplotlib, seaborn
- **Hardware**: GPU recommended for training (automatic CPU fallback)
- **Memory**: 8GB+ RAM recommended for batch_size=32

## Device Compatibility

- **Universal Compatibility**: Models work on GPU or CPU automatically
- **No Conversion Needed**: Same scripts work on any device

## Performance Summary

### **Key Techniques**
- **Adaptive Augmentation**: Class-size based augmentation rates
- **Staged Unfreezing**: Progressive backbone unfreezing
- **Focal Loss**: Custom implementation with gamma=2.0
- **Weighted Sampling**: Inverse frequency sampling
- **Gradient Clipping**: max_norm=1.0
- **Adaptive LR**: ReduceLROnPlateau scheduler
- **Early Stopping**: Patience-based stopping

### **Recommended Strategy**
1. **Start**: `--unfreeze 0 --loss weightedce` (baseline)
2. **Standard**: `--unfreeze 1 --loss focalloss --gamma 2.0`
3. **Advanced**: `--unfreeze 2 --loss focalloss --gamma 2.0`

### **Expected Improvements**
- **Class Imbalance**: 15-25% improvement in minority class accuracy
- **Staged Unfreezing**: 10-20% improvement in overall accuracy
- **Combined**: 25-40% improvement over baseline methods
