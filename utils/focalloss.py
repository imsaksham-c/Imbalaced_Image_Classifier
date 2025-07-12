"""
Focal Loss Implementation
========================

This module contains the Focal Loss implementation for handling class imbalance
in image classification tasks.

Reference: "Focal Loss for Dense Object Detection" by Lin et al.
"""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    
    Focal Loss down-weights easy examples and focuses training on hard examples,
    which is particularly useful for imbalanced datasets.
    
    Reference: "Focal Loss for Dense Object Detection" by Lin et al.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha (torch.Tensor, optional): Class weights for weighted loss
            gamma (float): Focusing parameter (default: 2.0)
            reduction (str): Reduction method ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Compute focal loss.
        
        Args:
            inputs (torch.Tensor): Model predictions (logits)
            targets (torch.Tensor): Ground truth labels
            
        Returns:
            torch.Tensor: Computed focal loss
        """
        # Compute cross entropy loss
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        
        # Compute probability
        pt = torch.exp(-ce_loss)
        
        # Apply focal loss formula: (1 - pt)^gamma * ce_loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss 