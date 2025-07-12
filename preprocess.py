"""
Dataset Preprocessing Script
===========================

Preprocesses image datasets for deep learning training by:
- Scanning dataset structure
- Splitting into train/validation sets
- Applying data augmentation based on class sizes
- Normalizing and resizing images
"""

import os
import shutil
import random
import argparse
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from collections import defaultdict
import json
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class DatasetPreprocessor:
    """Dataset preprocessor for image classification."""
    
    def __init__(self, dataset_path="./dataset", output_path="./processed_dataset", image_size=512):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.image_size = image_size
        self.train_path = self.output_path / "train"
        self.valid_path = self.output_path / "valid"
        
        # Validate input dataset path
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
        
        # Create output directories
        self.train_path.mkdir(parents=True, exist_ok=True)
        self.valid_path.mkdir(parents=True, exist_ok=True)
        
        # Setup augmentations
        self.setup_augmentations(image_size)
        
        # Store dataset statistics
        self.dataset_stats = {}
        
        print(f"Initialized DatasetPreprocessor")
        print(f"Input dataset: {self.dataset_path}")
        print(f"Output directory: {self.output_path}")
        print(f"Image size: {image_size}x{image_size}")
        
    def setup_augmentations(self, image_size=512):
        """Setup augmentation pipelines."""
        
        # Full augmentation pipeline for training
        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=0.3),
            A.GaussNoise(var_limit=(10, 30), p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Basic preprocessing (no augmentation)
        self.basic_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def scan_dataset(self):
        """Scan dataset and collect image paths."""
        image_paths = []
        class_counts = defaultdict(int)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        print(f"Scanning dataset at: {self.dataset_path}")
        
        for class_folder in self.dataset_path.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                class_images = []
                
                for img_path in class_folder.iterdir():
                    if img_path.suffix.lower() in image_extensions:
                        class_images.append(img_path)
                        class_counts[class_name] += 1
                
                image_paths.extend([(img_path, class_name) for img_path in class_images])
                
                if class_counts[class_name] == 0:
                    print(f"Warning: No valid images found in class '{class_name}'")
        
        # Store statistics
        self.dataset_stats['class_counts'] = dict(class_counts)
        self.dataset_stats['total_images'] = len(image_paths)
        self.dataset_stats['num_classes'] = len(class_counts)
        
        # Validate dataset
        if len(image_paths) == 0:
            raise ValueError("No valid images found in the dataset")
        
        if len(class_counts) == 0:
            raise ValueError("No valid class directories found in the dataset")
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Total images: {self.dataset_stats['total_images']}")
        print(f"Number of classes: {self.dataset_stats['num_classes']}")
        print(f"Class distribution:")
        
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} images")
        
        return image_paths
    
    def split_dataset(self, image_paths, test_size=0.2, random_state=42):
        """Split dataset into train/validation sets."""
        
        class_images = defaultdict(list)
        for img_path, class_name in image_paths:
            class_images[class_name].append(img_path)
        
        train_images = []
        valid_images = []
        
        print(f"\nSplitting dataset (train: {1-test_size:.0%}, validation: {test_size:.0%})...")
        
        for class_name, images in class_images.items():
            if len(images) == 1:
                train_images.extend([(img, class_name) for img in images])
                print(f"Class '{class_name}' has only 1 image → training set")
            else:
                if len(images) < 5:
                    valid_count = 1
                    train_count = len(images) - 1
                else:
                    valid_count = max(1, int(len(images) * test_size))
                    train_count = len(images) - valid_count
                
                random.Random(random_state).shuffle(images)
                train_imgs = images[:train_count]
                valid_imgs = images[train_count:train_count + valid_count]
                
                train_images.extend([(img, class_name) for img in train_imgs])
                valid_images.extend([(img, class_name) for img in valid_imgs])
                
                print(f"Class '{class_name}': {len(train_imgs)} train, {len(valid_imgs)} valid")
        
        print(f"\nSplit Summary:")
        print(f"Training images: {len(train_images)}")
        print(f"Validation images: {len(valid_images)}")
        print(f"Total: {len(train_images) + len(valid_images)}")
        
        return train_images, valid_images
    
    def get_augmentation_strategy(self, class_name):
        """Get augmentation strategy based on class size."""
        count = self.dataset_stats['class_counts'][class_name]
        
        if count >= 80:
            return 0.25, 1  # 25% of images, 1 augmentation per selected image
        elif count >= 50:
            return 0.5, 1   # 50% of images, 1 augmentation per selected image
        elif count >= 25:
            return 0.75, 1  # 75% of images, 1 augmentation per selected image
        else:
            if count < 15:
                return 1.0, 4  # 100% of images, 4 augmentations each
            else:
                return 1.0, 2  # 100% of images, 2 augmentations each
    
    def apply_augmentation(self, image):
        """Apply augmentation pipeline to an image."""
        return self.augmentation_pipeline(image=image)['image']
    
    def save_image(self, image, path):
        """Save image to disk."""
        if image.dtype == np.float32:
            # Denormalize using ImageNet statistics
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), image_bgr)
    
    def process_split(self, image_list, split_name):
        """Process a dataset split (train or valid)."""
        split_path = self.train_path if split_name == 'train' else self.valid_path
        
        # Create class directories
        for class_name in self.dataset_stats['class_counts'].keys():
            (split_path / class_name).mkdir(exist_ok=True)
        
        processed_count = 0
        augmented_count = 0
        
        print(f"\nProcessing {split_name} split...")
        
        for img_path, class_name in tqdm(image_list, desc=f"Processing {split_name}"):
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get augmentation strategy
            aug_ratio, aug_per_image = self.get_augmentation_strategy(class_name)
            
            # Save original image
            original_processed = self.basic_transform(image=image)['image']
            original_name = f"{img_path.stem}_original{img_path.suffix}"
            self.save_image(original_processed, split_path / class_name / original_name)
            processed_count += 1
            
            # Apply augmentation (only for training)
            if split_name == 'train' and random.random() < aug_ratio:
                for aug_idx in range(aug_per_image):
                    augmented = self.apply_augmentation(image)
                    aug_name = f"{img_path.stem}_aug{aug_idx + 1}{img_path.suffix}"
                    self.save_image(augmented, split_path / class_name / aug_name)
                    augmented_count += 1
        
        print(f"{split_name.capitalize()} split processed: {processed_count} images")
        if split_name == 'train':
            print(f"Augmented images: {augmented_count}")
    
    def save_dataset_info(self):
        """Save dataset information."""
        info = {
            'dataset_stats': self.dataset_stats,
            'augmentation_strategy': {
                'very_large_classes': '80+ images: 25% augmentation',
                'large_classes': '50-80 images: 50% augmentation',
                'medium_classes': '25-50 images: 75% augmentation', 
                'small_classes': '15-25 images: 100% augmentation (2x)',
                'very_small_classes': '<15 images: 100% augmentation (4x)'
            },
            'preprocessing': {
                'image_size': f'{self.image_size}x{self.image_size}',
                'normalization': 'ImageNet normalization',
                'train_valid_split': '80/20'
            }
        }
        
        info_path = self.output_path / 'dataset_info.json'
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Dataset information saved to: {info_path}")
    
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline."""
        print("Starting dataset preprocessing...")
        print("=" * 50)
        
        # Step 1: Scan dataset
        print("\nStep 1: Scanning dataset...")
        image_paths = self.scan_dataset()
        
        # Step 2: Split dataset
        print("\nStep 2: Splitting dataset...")
        train_images, valid_images = self.split_dataset(image_paths)
        
        # Step 3: Process training set
        print("\nStep 3: Processing training set...")
        self.process_split(train_images, 'train')
        
        # Step 4: Process validation set
        print("\nStep 4: Processing validation set...")
        self.process_split(valid_images, 'valid')
        
        # Step 5: Save dataset information
        print("\nStep 5: Saving dataset information...")
        self.save_dataset_info()
        
        print(f"\nPreprocessing completed!")
        print(f"Output directory: {self.output_path}")


def get_final_statistics(preprocessor):
    """Calculate final statistics."""
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    
    train_count = 0
    valid_count = 0
    
    for class_dir in preprocessor.train_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            class_train_count = len(list(class_dir.glob('*')))
            train_count += class_train_count
            
            valid_class_dir = preprocessor.valid_path / class_dir.name
            if valid_class_dir.exists():
                class_valid_count = len(list(valid_class_dir.glob('*')))
                valid_count += class_valid_count
                print(f"{class_name}: {class_train_count} train, {class_valid_count} valid")
    
    total_processed = train_count + valid_count
    original_total = preprocessor.dataset_stats['total_images']
    augmentation_gain = total_processed - original_total
    
    print(f"\nFinal Statistics:")
    print(f"Training images: {train_count}")
    print(f"Validation images: {valid_count}")
    print(f"Total processed: {total_processed}")
    print(f"Original images: {original_total}")
    print(f"Augmented images: {augmentation_gain}")
    
    return {
        'train_count': train_count,
        'valid_count': valid_count,
        'total_processed': total_processed,
        'original_count': original_total,
        'augmentation_gain': augmentation_gain
    }


def main():
    """Main function with command-line arguments."""
    
    parser = argparse.ArgumentParser(
        description='Dataset Preprocessing Script',
        epilog="""
Examples:
  python preprocess.py --dataset ./my_dataset
  python preprocess.py --dataset ./dataset --output ./processed_data
  python preprocess.py --dataset ./dataset --image_size 224
        """
    )
    
    parser.add_argument('--dataset', type=str, default='./dataset',
                       help='Path to input dataset directory (default: ./dataset)')
    parser.add_argument('--output', type=str, default='./processed_dataset',
                       help='Path to output directory (default: ./processed_dataset)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size for resizing (default: 512)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output directory')
    
    args = parser.parse_args()
    
    print("Dataset Preprocessing Script")
    print("=" * 50)
    
    # Validate input dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return
    
    # Check output directory
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"Error: Output directory already exists: {output_path}")
        print(f"Use --overwrite flag to overwrite")
        return
    
    if output_path.exists() and args.overwrite:
        print(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    
    try:
        # Initialize and run preprocessor
        preprocessor = DatasetPreprocessor(
            dataset_path=args.dataset,
            output_path=args.output,
            image_size=args.image_size
        )
        
        preprocessor.run_preprocessing()
        final_stats = get_final_statistics(preprocessor)
        
        print(f"\nPreprocessing completed successfully!")
        print(f"Output directory: {output_path}")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return


if __name__ == "__main__":
    main()