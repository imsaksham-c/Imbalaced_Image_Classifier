"""
Dataset Preprocessing Script
===========================

Preprocesses image datasets for deep learning training by:
- Scanning dataset structure
- Splitting into train/validation sets (or train/validation/test sets)
- Applying data augmentation based on class sizes
- Normalizing and resizing images

Examples:
  python preprocess.py --dataset ./my_dataset
  python preprocess.py --dataset ./dataset --output ./processed_data
  python preprocess.py --dataset ./dataset --image_size 512
  python preprocess.py --dataset ./dataset --test_split --test_size_ratio 0.15
"""

import argparse
from pathlib import Path
import cv2
import json
from tqdm import tqdm
import shutil

from utils.dataset import (
    scan_dataset, split_dataset, get_augmentation_strategy, save_image,
    get_albumentations_transforms, get_basic_transform
)

class DatasetPreprocessor:
    """Dataset preprocessor for image classification."""
    def __init__(self, dataset_path="./dataset", output_path="./processed_dataset", image_size=512, test_split=False, test_size_ratio=0.1):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.image_size = image_size
        self.test_split = test_split
        self.test_size_ratio = test_size_ratio
        self.train_path = self.output_path / "train"
        self.valid_path = self.output_path / "valid"
        self.test_path = self.output_path / "test" if test_split else None
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
        self.train_path.mkdir(parents=True, exist_ok=True)
        self.valid_path.mkdir(parents=True, exist_ok=True)
        if self.test_split:
            self.test_path.mkdir(parents=True, exist_ok=True)
        self.augmentation_pipeline = get_albumentations_transforms(image_size)
        self.basic_transform = get_basic_transform(image_size)
        self.dataset_stats = {}
        print(f"Initialized DatasetPreprocessor")
        print(f"Input dataset: {self.dataset_path}")
        print(f"Output directory: {self.output_path}")
        print(f"Image size: {image_size}x{image_size}")
        print(f"Test split: {test_split}")
        if test_split:
            print(f"Test size ratio: {test_size_ratio}")

    def run_preprocessing(self):
        image_paths, class_counts = scan_dataset(self.dataset_path)
        self.dataset_stats['class_counts'] = class_counts
        self.dataset_stats['total_images'] = len(image_paths)
        self.dataset_stats['num_classes'] = len(class_counts)
        print(f"\nDataset Statistics:")
        print(f"Total images: {self.dataset_stats['total_images']}")
        print(f"Number of classes: {self.dataset_stats['num_classes']}")
        print(f"Class distribution:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} images")
        
        if self.test_split:
            train_images, valid_images, test_images = split_dataset(
                image_paths, test_split=True, test_size_ratio=self.test_size_ratio
            )
            print(f"\nSplit Summary:")
            print(f"Training images: {len(train_images)}")
            print(f"Validation images: {len(valid_images)}")
            print(f"Test images: {len(test_images)}")
            print(f"Total: {len(train_images) + len(valid_images) + len(test_images)}")
            
            # Store split information for comparison
            self.dataset_stats['train_images'] = train_images
            self.dataset_stats['valid_images'] = valid_images
            self.dataset_stats['test_images'] = test_images
            
            self.process_split(train_images, 'train', class_counts)
            self.process_split(valid_images, 'valid', class_counts, augment=False)
            self.process_split(test_images, 'test', class_counts, augment=False)
        else:
            train_images, valid_images = split_dataset(image_paths)
            print(f"\nSplit Summary:")
            print(f"Training images: {len(train_images)}")
            print(f"Validation images: {len(valid_images)}")
            print(f"Total: {len(train_images) + len(valid_images)}")
            
            # Store split information for comparison
            self.dataset_stats['train_images'] = train_images
            self.dataset_stats['valid_images'] = valid_images
            
            self.process_split(train_images, 'train', class_counts)
            self.process_split(valid_images, 'valid', class_counts, augment=False)
        
        self.save_dataset_info()
        self.print_before_after_comparison()

    def process_split(self, image_list, split_name, class_counts, augment=True):
        if split_name == 'train':
            split_dir = self.train_path
        elif split_name == 'valid':
            split_dir = self.valid_path
        elif split_name == 'test':
            split_dir = self.test_path
        else:
            raise ValueError(f"Invalid split name: {split_name}")
        
        # Track processed images for this split
        processed_counts = {}
        
        for img_path, class_name in tqdm(image_list, desc=f'Processing {split_name} set'):
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            out_path = class_dir / img_path.name
            save_image(self.basic_transform(image=img)['image'], out_path)
            
            # Count original image
            processed_counts[class_name] = processed_counts.get(class_name, 0) + 1
            
            if augment and split_name == 'train':
                frac, n_aug = get_augmentation_strategy(class_counts, class_name)
                # Only augment a fraction of images, not all images
                import random
                if random.random() < frac:
                    for i in range(n_aug):
                        aug_img = self.augmentation_pipeline(image=img)['image']
                        aug_path = class_dir / f"aug_{i}_{img_path.name}"
                        save_image(aug_img, aug_path)
                        # Count augmented image
                        processed_counts[class_name] += 1
        
        # Store processed counts for this split
        if not hasattr(self, 'processed_stats'):
            self.processed_stats = {}
        self.processed_stats[split_name] = processed_counts

    def save_dataset_info(self):
        info = {
            'class_counts': self.dataset_stats['class_counts'],
            'total_images': self.dataset_stats['total_images'],
            'num_classes': self.dataset_stats['num_classes'],
            'preprocessing': {
                'image_size': f"{self.image_size}x{self.image_size}",
                'augmentation': True,
                'test_split': self.test_split,
                'test_size_ratio': self.test_size_ratio if self.test_split else None
            }
        }
        with open(self.output_path / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)

    def print_before_after_comparison(self):
        """Print detailed before and after preprocessing comparison."""
        print("\n" + "="*80)
        print("📊 BEFORE vs AFTER PREPROCESSING COMPARISON")
        print("="*80)
        
        # Calculate training set counts before augmentation (after splitting)
        train_before_counts = {}
        for img_path, class_name in self.dataset_stats['train_images']:
            train_before_counts[class_name] = train_before_counts.get(class_name, 0) + 1
        
        # Calculate total processed images
        total_processed = 0
        train_processed = 0
        valid_processed = 0
        test_processed = 0
        
        if hasattr(self, 'processed_stats'):
            if 'train' in self.processed_stats:
                train_processed = sum(self.processed_stats['train'].values())
            if 'valid' in self.processed_stats:
                valid_processed = sum(self.processed_stats['valid'].values())
            if 'test' in self.processed_stats:
                test_processed = sum(self.processed_stats['test'].values())
            total_processed = train_processed + valid_processed + test_processed
        
        # Print summary
        print(f"\n📈 SUMMARY:")
        print(f"   Training set (before augmentation): {sum(train_before_counts.values())} images")
        print(f"   Training set (after augmentation): {train_processed} images")
        print(f"   Training increase: {train_processed - sum(train_before_counts.values())} images ({((train_processed - sum(train_before_counts.values())) / sum(train_before_counts.values()) * 100):.1f}%)")
        print(f"   Validation set: {valid_processed} images (no augmentation)")
        if self.test_split:
            print(f"   Test set: {test_processed} images (no augmentation)")
        print(f"   Total processed: {total_processed} images")
        
        # Print detailed class comparison (training set only)
        print(f"\n📋 DETAILED TRAINING SET COMPARISON:")
        print(f"{'Class':<25} {'Before Aug':<10} {'After Aug':<10} {'Increase':<10} {'Rate':<8}")
        print("-" * 65)
        
        for class_name in sorted(train_before_counts.keys()):
            before = train_before_counts[class_name]
            after = self.processed_stats['train'].get(class_name, 0) if hasattr(self, 'processed_stats') else before
            
            increase = after - before
            rate = (increase / before * 100) if before > 0 else 0
            
            print(f"{class_name:<25} {before:<10} {after:<10} {increase:<10} {rate:>6.1f}%")
        
        # Calculate class imbalance ratios (training set only)
        print(f"\n⚖️  TRAINING SET IMBALANCE ANALYSIS:")
        before_max = max(train_before_counts.values())
        before_min = min(train_before_counts.values())
        before_ratio = before_max / before_min if before_min > 0 else float('inf')
        
        if hasattr(self, 'processed_stats') and 'train' in self.processed_stats:
            after_max = max(self.processed_stats['train'].values())
            after_min = min(self.processed_stats['train'].values())
            after_ratio = after_max / after_min if after_min > 0 else float('inf')
            
            print(f"   Before augmentation: {before_ratio:.1f}x ({before_max}/{before_min})")
            print(f"   After augmentation: {after_ratio:.1f}x ({after_max}/{after_min})")
            print(f"   Improvement: {((before_ratio - after_ratio) / before_ratio * 100):.1f}% reduction in imbalance")
        
        print("="*80)


def run_preprocessing(args):
    """
    Main preprocessing function that handles the preprocessing logic.
    
    Args:
        args: Parsed command line arguments
    """
    print("Dataset Preprocessing Script")
    print("=" * 50)
    
    # Validate input dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return False
    
    # Check output directory
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        print(f"Error: Output directory already exists: {output_path}")
        print(f"Use --overwrite flag to overwrite")
        return False
    
    if output_path.exists() and args.overwrite:
        print(f"Removing existing output directory: {output_path}")
        shutil.rmtree(output_path)
    
    try:
        # Initialize and run preprocessor
        preprocessor = DatasetPreprocessor(
            dataset_path=args.dataset,
            output_path=args.output,
            image_size=args.image_size,
            test_split=args.test_split,
            test_size_ratio=args.test_size_ratio
        )
        
        preprocessor.run_preprocessing()
        
        print(f"\nPreprocessing completed successfully!")
        print(f"Output directory: {output_path}")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return False


def main():
    """
    Main function that handles argument parsing and calls the preprocessing function.
    """
    # ============================================================================
    # ARGUMENT PARSING
    # ============================================================================
    parser = argparse.ArgumentParser(
        description='Dataset Preprocessing Script'
    )
    
    parser.add_argument('--dataset', type=str, default='./train',
                       help='Path to input dataset directory (default: ./dataset)')
    parser.add_argument('--output', type=str, default='./processed_dataset',
                       help='Path to output directory (default: ./processed_dataset)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    parser.add_argument('--test_split', action='store_true',
                       help='Enable test split (default: False)')
    parser.add_argument('--test_size_ratio', type=float, default=0.1,
                       help='Test split ratio (default: 0.1, only used when --test_split is enabled)')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Image size for resizing (default: 224)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing output directory')
    
    args = parser.parse_args()
    
    # Call the preprocessing function
    run_preprocessing(args)


if __name__ == "__main__":
    main()