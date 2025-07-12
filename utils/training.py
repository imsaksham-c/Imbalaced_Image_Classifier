"""
Training Functions
=================

This module contains the core training and validation functions for the
temple image classification model.
"""

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer):
    """
    Train the model for one epoch.
    
    Args:
        model (torch.nn.Module): The model to train
        dataloader (DataLoader): Training data loader
        criterion (callable): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        
    Returns:
        tuple: (epoch_loss, epoch_accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Training loop with progress bar
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc='Training', leave=False)):
        images, labels = images.to(model.device), labels.to(model.device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Progress updates
        if batch_idx % 10 == 0:
            tqdm.write(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion):
    """
    Validate the model for one epoch.
    
    Args:
        model (torch.nn.Module): The model to validate
        dataloader (DataLoader): Validation data loader
        criterion (callable): Loss function
        
    Returns:
        tuple: (epoch_loss, epoch_accuracy, f1_weighted, f1_macro, predictions, true_labels)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validating', leave=False):
            images, labels = images.to(model.device), labels.to(model.device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect predictions for metrics
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    
    return epoch_loss, epoch_acc, f1_weighted, f1_macro, all_predictions, all_labels 