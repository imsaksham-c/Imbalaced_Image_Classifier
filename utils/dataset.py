"""
Dataset Loading and Preprocessing
================================

This module contains functions for loading, preprocessing, and managing
the temple image dataset.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image


class TempleDataset(Dataset):
    """
    Custom Dataset class for temple images.
    
    This dataset handles loading and preprocessing of temple images with proper
    error handling for corrupted or missing files.
    
    Attributes:
        image_paths (list): List of paths to image files
        labels (list): List of corresponding labels
        transform (callable): Optional transform to apply to images
    """
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_paths (list): List of image file paths
            labels (list): List of corresponding labels
            transform (callable, optional): Transform to apply to images
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, label) where image is a PIL Image or tensor
        """
        image_path = self.image_paths[idx]
        try:
            # Load and convert image to RGB format
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"⚠️  Error loading image {image_path}: {e}")
            # Create a white image as fallback
            image = Image.new('RGB', (512, 512), color='white')
        
        label = self.labels[idx]
        
        # Apply transforms if specified
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms():
    """
    Get image transformations for training and validation.
    
    Returns:
        transforms.Compose: Composition of image transformations
    """
    return transforms.Compose([
        transforms.Resize((512, 512)),  # Resize to consistent dimensions
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize(           # Normalize with ImageNet statistics
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_processed_dataset(processed_dir, split):
    """
    Load processed dataset from directory structure.
    
    Expected directory structure:
    processed_dir/
    ├── train/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── valid/
        ├── class1/
        ├── class2/
        └── ...
    
    Args:
        processed_dir (str): Path to processed dataset directory
        split (str): Dataset split ('train' or 'valid')
        
    Returns:
        tuple: (image_paths, labels, class_names, class_mapping)
        
    Raises:
        ValueError: If split directory doesn't exist
    """
    image_paths = []
    labels = []
    class_names = []
    class_mapping = {}
    
    split_dir = os.path.join(processed_dir, split)
    if not os.path.exists(split_dir):
        raise ValueError(f"❌ Split directory {split_dir} does not exist.")
    
    # Iterate through class directories
    for idx, class_name in enumerate(sorted(os.listdir(split_dir))):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        class_mapping[class_name] = idx
        class_names.append(class_name)
        
        # Collect all image files in the class directory
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_path = os.path.join(class_dir, img_file)
                image_paths.append(image_path)
                labels.append(idx)
    
    return image_paths, labels, class_names, class_mapping


def calculate_class_weights(labels, num_classes):
    """
    Calculate class weights for handling imbalanced datasets.
    
    Args:
        labels (list): List of class labels
        num_classes (int): Total number of classes
        
    Returns:
        torch.Tensor: Class weights tensor
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    # Inverse frequency weighting
    class_weights = total_samples / (num_classes * class_counts)
    return torch.FloatTensor(class_weights)


def get_weighted_sampler(labels, class_weights):
    """
    Create weighted random sampler for balanced training.
    
    Args:
        labels (list): List of class labels
        class_weights (torch.Tensor): Class weights
        
    Returns:
        WeightedRandomSampler: Configured sampler
    """
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(labels)) 