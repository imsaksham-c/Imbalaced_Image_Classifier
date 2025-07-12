import os
import shutil
import random
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from collections import defaultdict
import json

class TempleDatasetPreprocessor:
    def __init__(self, dataset_path="./dataset", output_path="./processed_dataset"):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.train_path = self.output_path / "train"
        self.valid_path = self.output_path / "valid"
        
        # Create output directories
        self.train_path.mkdir(parents=True, exist_ok=True)
        self.valid_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize augmentation pipelines
        self.setup_augmentations()
        
        # Store dataset statistics
        self.dataset_stats = {}
        
    def setup_augmentations(self):
        """Setup single augmentation pipeline for all classes"""
        
        # Single augmentation pipeline for all classes
        self.augmentation_pipeline = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=15, p=0.3),
            A.GaussNoise(var_limit=(10, 30), p=0.2),
            A.Blur(blur_limit=3, p=0.1),
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Basic preprocessing (no augmentation)
        self.basic_transform = A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def scan_dataset(self):
        """Scan the dataset and collect image paths and class information"""
        image_paths = []
        class_counts = defaultdict(int)
        
        # Supported image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for class_folder in self.dataset_path.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                class_images = []
                
                for img_path in class_folder.iterdir():
                    if img_path.suffix.lower() in image_extensions:
                        class_images.append(img_path)
                        class_counts[class_name] += 1
                
                image_paths.extend([(img_path, class_name) for img_path in class_images])
        
        self.dataset_stats['class_counts'] = dict(class_counts)
        self.dataset_stats['total_images'] = len(image_paths)
        
        print(f"Dataset Statistics:")
        print(f"Total images: {self.dataset_stats['total_images']}")
        print(f"Classes found: {len(class_counts)}")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} images")
        
        return image_paths
    
    def split_dataset(self, image_paths, test_size=0.2, random_state=42):
        """Split dataset into train/validation ensuring all classes are represented"""
        
        # Group images by class
        class_images = defaultdict(list)
        for img_path, class_name in image_paths:
            class_images[class_name].append(img_path)
        
        train_images = []
        valid_images = []
        
        # Split each class separately to ensure representation
        for class_name, images in class_images.items():
            if len(images) == 1:
                # If only one image, put it in training
                train_images.extend([(img, class_name) for img in images])
                print(f"Warning: Class '{class_name}' has only 1 image, putting in training set")
            else:
                # Ensure at least one image in validation
                if len(images) < 5:
                    # For very small classes, manually split
                    valid_count = 1
                    train_count = len(images) - 1
                else:
                    valid_count = max(1, int(len(images) * test_size))
                    train_count = len(images) - valid_count
                
                # Randomly shuffle and split
                random.Random(random_state).shuffle(images)
                train_imgs = images[:train_count]
                valid_imgs = images[train_count:train_count + valid_count]
                
                train_images.extend([(img, class_name) for img in train_imgs])
                valid_images.extend([(img, class_name) for img in valid_imgs])
                
                print(f"Class '{class_name}': {len(train_imgs)} train, {len(valid_imgs)} valid")
        
        return train_images, valid_images
    
    def get_augmentation_strategy(self, class_name):
        """Determine augmentation strategy based on class size (4 categories)"""
        count = self.dataset_stats['class_counts'][class_name]
        
        if count >= 80:
            return 0.25, 1  # 25% of images, 1 augmentation per selected image
        elif count >= 50:
            return 0.5, 1   # 50% of images, 1 augmentation per selected image
        elif count >= 25:
            return 0.75, 1  # 75% of images, 1 augmentation per selected image
        else:
            # For very small classes, augment all images multiple times
            if count < 15:
                return 1.0, 4  # 100% of images, 4 augmentations each
            else:
                return 1.0, 2  # 100% of images, 2 augmentations each
    
    def apply_augmentation(self, image):
        """Apply the same augmentation pipeline to all classes"""
        return self.augmentation_pipeline(image=image)['image']
    
    def save_image(self, image, path):
        """Save image to disk"""
        # Convert from normalized float to uint8
        if image.dtype == np.float32:
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = image * std + mean
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), image_bgr)
    
    def process_split(self, image_list, split_name):
        """Process a split (train or valid) with appropriate augmentations"""
        split_path = self.train_path if split_name == 'train' else self.valid_path
        
        # Create class directories
        for class_name in self.dataset_stats['class_counts'].keys():
            (split_path / class_name).mkdir(exist_ok=True)
        
        processed_count = 0
        augmented_count = 0
        
        for img_path, class_name in image_list:
            # Read image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                continue
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get augmentation strategy for this class
            aug_ratio, aug_per_image = self.get_augmentation_strategy(class_name)
            
            # Save original image with basic preprocessing
            original_processed = self.basic_transform(image=image)['image']
            original_name = f"{img_path.stem}_original{img_path.suffix}"
            self.save_image(original_processed, split_path / class_name / original_name)
            processed_count += 1
            
            # Apply augmentation based on strategy (only for training)
            if split_name == 'train' and random.random() < aug_ratio:
                # Create multiple augmented versions for small classes
                for aug_idx in range(aug_per_image):
                    augmented = self.apply_augmentation(image)
                    aug_name = f"{img_path.stem}_aug{aug_idx + 1}{img_path.suffix}"
                    self.save_image(augmented, split_path / class_name / aug_name)
                    augmented_count += 1
        
        print(f"{split_name.capitalize()} split processed: {processed_count} images")
        if split_name == 'train':
            print(f"  Augmented images: {augmented_count}")
            
            # Print augmentation statistics per class
            print("  Per-class augmentation breakdown:")
            for class_name in set([class_name for _, class_name in image_list]):
                total_count = self.dataset_stats['class_counts'][class_name]
                aug_ratio, aug_per_image = self.get_augmentation_strategy(class_name)
                
                # Calculate expected augmented images
                train_count = len([1 for _, cn in image_list if cn == class_name])
                expected_aug = int(train_count * aug_ratio * aug_per_image)
                
                # Determine class category
                if total_count >= 80:
                    category = "Very Large (80+)"
                elif total_count >= 50:
                    category = "Large (50-80)"
                elif total_count >= 25:
                    category = "Medium (25-50)"
                else:
                    category = "Small (<25)"
                
                print(f"    {class_name}: {train_count} original → ~{expected_aug} augmented ({int(aug_ratio*100)}% × {aug_per_image} each) - {category}")
            
    
    def save_dataset_info(self):
        """Save dataset information and processing statistics"""
        info = {
            'dataset_stats': self.dataset_stats,
            'augmentation_strategy': {
                'same_augmentations_for_all_classes': True,
                'augmentation_ratios': {
                    'very_large_classes': '80+ images: 25% augmentation (1x per selected image)',
                    'large_classes': '50-80 images: 50% augmentation (1x per selected image)',
                    'medium_classes': '25-50 images: 75% augmentation (1x per selected image)', 
                    'small_classes': '15-25 images: 100% augmentation (2x per image)',
                    'very_small_classes': '<15 images: 100% augmentation (4x per image)'
                },
                'augmentation_techniques': [
                    'HorizontalFlip (p=0.5)',
                    'RandomBrightnessContrast (p=0.6)',
                    'RandomRotate90 (p=0.3)',
                    'ShiftScaleRotate (p=0.5)',
                    'RandomGamma (p=0.4)',
                    'HueSaturationValue (p=0.3)',
                    'GaussNoise (p=0.2)',
                    'Blur (p=0.1)'
                ]
            },
            'preprocessing': {
                'image_size': '512x512',
                'normalization': 'ImageNet normalization',
                'train_valid_split': '80/20'
            }
        }
        
        with open(self.output_path / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)
    
    def run_preprocessing(self):
        """Run the complete preprocessing pipeline"""
        print("Starting temple dataset preprocessing...")
        
        # Step 1: Scan dataset
        print("\n1. Scanning dataset...")
        image_paths = self.scan_dataset()
        
        # Step 2: Split dataset
        print("\n2. Splitting dataset (80/20)...")
        train_images, valid_images = self.split_dataset(image_paths)
        
        print(f"Train set: {len(train_images)} images")
        print(f"Validation set: {len(valid_images)} images")
        
        # Step 3: Process training set
        print("\n3. Processing training set with augmentations...")
        self.process_split(train_images, 'train')
        
        # Step 4: Process validation set
        print("\n4. Processing validation set...")
        self.process_split(valid_images, 'valid')
        
        # Step 5: Save dataset information
        print("\n5. Saving dataset information...")
        self.save_dataset_info()
        
        print(f"\nPreprocessing complete!")
        print(f"Processed dataset saved to: {self.output_path}")
        print(f"Structure:")
        print(f"  {self.output_path}/train/")
        print(f"  {self.output_path}/valid/")
        print(f"  {self.output_path}/dataset_info.json")

def main():
    """Main function to run the preprocessing"""
    
    # Initialize preprocessor
    preprocessor = TempleDatasetPreprocessor(
        dataset_path="./dataset",
        output_path="./processed_dataset"
    )
    
    # Run preprocessing
    preprocessor.run_preprocessing()
    
    # Print final statistics
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    
    # Count final images
    train_count = 0
    valid_count = 0
    
    for class_dir in preprocessor.train_path.iterdir():
        if class_dir.is_dir():
            class_train_count = len(list(class_dir.glob('*')))
            train_count += class_train_count
            
            valid_class_dir = preprocessor.valid_path / class_dir.name
            if valid_class_dir.exists():
                class_valid_count = len(list(valid_class_dir.glob('*')))
                valid_count += class_valid_count
                print(f"{class_dir.name}: {class_train_count} train, {class_valid_count} valid")
    
    print(f"\nTotal processed images:")
    print(f"  Training: {train_count}")
    print(f"  Validation: {valid_count}")
    print(f"  Total: {train_count + valid_count}")

if __name__ == "__main__":
    main()