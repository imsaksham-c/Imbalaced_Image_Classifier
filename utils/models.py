"""
Model Initialization and Configuration
=====================================

This module contains functions for initializing and configuring deep learning
models for temple image classification.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, efficientnet_b3, resnext50_32x4d


def get_model(model_name, num_classes, pretrained=True, fc_layers=None, device=None):
    """
    Initialize and configure a deep learning model.
    
    Args:
        model_name (str): Name of the model architecture
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        fc_layers (list, optional): List of FC layer sizes for custom classifier
        device (torch.device, optional): Device to place model on
        
    Returns:
        torch.nn.Module: Configured model
        
    Raises:
        ValueError: If model_name is not supported
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name == 'resnet50':
        model = resnet50(pretrained=pretrained)
        if fc_layers:
            # Build custom classifier with specified layer sizes
            layers = []
            in_features = model.fc.in_features
            
            for i, layer_size in enumerate(fc_layers):
                layers.extend([
                    nn.Linear(in_features if i == 0 else fc_layers[i-1], layer_size),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(layer_size),
                    nn.Dropout(0.5)
                ])
                in_features = layer_size
            
            # Final classification layer
            layers.append(nn.Linear(fc_layers[-1], num_classes))
            model.fc = nn.Sequential(*layers)
        else:
            # Default classifier
            model.fc = nn.Sequential(
                nn.Dropout(0.6),
                nn.Linear(model.fc.in_features, num_classes)
            )
            
    elif model_name == 'resnext50':
        model = resnext50_32x4d(pretrained=pretrained)
        if fc_layers:
            # Build custom classifier with specified layer sizes
            layers = []
            in_features = model.fc.in_features
            
            for i, layer_size in enumerate(fc_layers):
                layers.extend([
                    nn.Linear(in_features if i == 0 else fc_layers[i-1], layer_size),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(layer_size),
                    nn.Dropout(0.5)
                ])
                in_features = layer_size
            
            # Final classification layer
            layers.append(nn.Linear(fc_layers[-1], num_classes))
            model.fc = nn.Sequential(*layers)
        else:
            # Default classifier
            model.fc = nn.Sequential(
                nn.Dropout(0.6),
                nn.Linear(model.fc.in_features, num_classes)
            )
            
    elif model_name == 'efficientnet_b3':
        model = efficientnet_b3(pretrained=pretrained)
        if fc_layers:
            # Build custom classifier with specified layer sizes
            layers = []
            in_features = model.classifier[1].in_features
            
            for i, layer_size in enumerate(fc_layers):
                layers.extend([
                    nn.Linear(in_features if i == 0 else fc_layers[i-1], layer_size),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(layer_size),
                    nn.Dropout(0.5)
                ])
                in_features = layer_size
            
            # Final classification layer
            layers.append(nn.Linear(fc_layers[-1], num_classes))
            model.classifier = nn.Sequential(*layers)
        else:
            # Default classifier
            model.classifier = nn.Sequential(
                nn.Dropout(0.6),
                nn.Linear(model.classifier[1].in_features, num_classes)
            )
    else:
        raise ValueError(f"❌ Model {model_name} is not supported. "
                        f"Supported models: resnet50, resnext50, efficientnet_b3")
    
    return model.to(device)


def set_trainable_layers(model, stage, unfreeze_mode):
    """
    Configure which layers are trainable based on unfreeze mode and training stage.
    
    This function implements staged unfreezing strategies for transfer learning:
    
    Unfreeze Modes:
    0 - Only classifier layers trainable (feature extraction)
    1 - Unfreezes deepest backbone block at stage 2 (after 15 epochs)
    2 - Unfreezes stage 2 after 15 epochs and stage 3 after 30 epochs
    3 - All layers trainable from start (full fine-tuning)
    
    Args:
        model (torch.nn.Module): The model to configure
        stage (int): Current training stage (1, 2, or 3)
        unfreeze_mode (int): Unfreezing strategy (0, 1, 2, or 3)
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