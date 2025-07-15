"""
Dataset Loading and Preprocessing
================================

This module contains functions for loading, preprocessing, and managing
the temple image dataset.
"""

import torch
import torch.utils.data as data
from torch.utils.data import Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import numpy as np
import os
import albumentations as A
import cv2
from collections import defaultdict
from pathlib import Path
from PIL import Image


class TempleDataset(Dataset):
    """Custom dataset for temple images."""
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


def get_transforms(image_size=512):
    """Get basic transforms for training/validation."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def load_processed_dataset(processed_dir, split):
    """Load processed dataset from train/valid splits."""
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
    """Calculate class weights for imbalanced dataset."""
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    class_weights = total_samples / (num_classes * class_counts)
    return torch.FloatTensor(class_weights)


def get_weighted_sampler(labels, class_weights):
    """Create weighted sampler for balanced training."""
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(labels))


def scan_dataset(dataset_path):
    """Scan dataset and collect image paths and class counts."""
    dataset_path = Path(dataset_path)
    image_paths = []
    class_counts = defaultdict(int)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    for class_folder in dataset_path.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            class_images = []
            for img_path in class_folder.iterdir():
                if img_path.suffix.lower() in image_extensions:
                    class_images.append(img_path)
                    class_counts[class_name] += 1
            image_paths.extend([(img_path, class_name) for img_path in class_images])
    return image_paths, dict(class_counts)


def split_dataset(image_paths, test_size=0.2, test_split=False, test_size_ratio=0.1, random_state=42):
    """Split dataset into train/validation/test sets by class."""
    from random import Random
    class_images = defaultdict(list)
    for img_path, class_name in image_paths:
        class_images[class_name].append(img_path)
    
    train_images = []
    valid_images = []
    test_images = []
    
    for class_name, images in class_images.items():
        if len(images) == 1:
            train_images.extend([(img, class_name) for img in images])
        else:
            if len(images) < 5:
                if test_split:
                    valid_count = 1
                    test_count = 1
                    train_count = len(images) - 2
                else:
                    valid_count = 1
                    train_count = len(images) - 1
            else:
                if test_split:
                    valid_count = max(1, int(len(images) * test_size))
                    test_count = max(1, int(len(images) * test_size_ratio))
                    train_count = len(images) - valid_count - test_count
                else:
                    valid_count = max(1, int(len(images) * test_size))
                    train_count = len(images) - valid_count
            
            Random(random_state).shuffle(images)
            
            if test_split:
                train_imgs = images[:train_count]
                valid_imgs = images[train_count:train_count + valid_count]
                test_imgs = images[train_count + valid_count:train_count + valid_count + test_count]
                
                train_images.extend([(img, class_name) for img in train_imgs])
                valid_images.extend([(img, class_name) for img in valid_imgs])
                test_images.extend([(img, class_name) for img in test_imgs])
            else:
                train_imgs = images[:train_count]
                valid_imgs = images[train_count:train_count + valid_count]
                
                train_images.extend([(img, class_name) for img in train_imgs])
                valid_images.extend([(img, class_name) for img in valid_imgs])
    
    if test_split:
        return train_images, valid_images, test_images
    else:
        return train_images, valid_images


def get_augmentation_strategy(class_counts, class_name):
    """
    Enhanced augmentation strategy for temple classification.
    More augmentations for weak (small) classes.
    """
    count = class_counts[class_name]
    if count >= 80:
        # Large classes: Minimal augmentation
        return 0.2, 1
    elif count >= 50:
        # Medium-large classes: Light augmentation
        return 0.4, 1
    elif count >= 25:
        # Medium classes: Moderate augmentation
        return 0.6, 2
    else:
        # Small classes: Heavy and diverse augmentation
        return 1.0, 4  # Always augment, 4 times per image


def save_image(image, path):
    """Save image to disk, denormalizing if needed."""
    if image.dtype == np.float32:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def get_albumentations_transforms(image_size=512):
    """Get a diverse albumentations augmentation pipeline for temple images, compatible with Albumentations 2.0.8."""
    return A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.CenterCrop(height=image_size, width=image_size, p=0.3),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
        A.ElasticTransform(alpha=1, sigma=50, p=0.15),  # removed alpha_affine
        # Lighting and color
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.RandomGamma(gamma_limit=(70, 130), p=0.5),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=18, val_shift_limit=18, p=0.5),
        # A.Solarize and A.Posterize are not available in 2.0.8, so remove them
        # Weather/atmospheric effects
        A.GaussNoise(var_limit=(8.0, 40.0), p=0.3),
        A.Blur(blur_limit=(3, 5), p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=0.1),  # updated argument
        A.RandomRain(blur_value=2, brightness_coefficient=0.9, drop_length=10, drop_width=1, p=0.1),
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.1),  # removed num_shadows_lower/upper
        A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.1),  # removed angle_lower
        # Dropout
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, min_holes=2, min_height=8, min_width=8, fill_value=0, p=0.2),
        # Resize and normalize
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_basic_transform(image_size=512):
    """Get albumentations basic transform (resize + normalize)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 