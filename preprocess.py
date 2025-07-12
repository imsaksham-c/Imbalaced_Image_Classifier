"""
Dataset Preprocessing Script
===========================

Preprocesses image datasets for deep learning training by:
- Scanning dataset structure
- Splitting into train/validation sets
- Applying data augmentation based on class sizes
- Normalizing and resizing images

Examples:
  python preprocess.py --dataset ./my_dataset
  python preprocess.py --dataset ./dataset --output ./processed_data
  python preprocess.py --dataset ./dataset --image_size 224
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
    def __init__(self, dataset_path="./dataset", output_path="./processed_dataset", image_size=512):
        self.dataset_path = Path(dataset_path)
        self.output_path = Path(output_path)
        self.image_size = image_size
        self.train_path = self.output_path / "train"
        self.valid_path = self.output_path / "valid"
        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")
        self.train_path.mkdir(parents=True, exist_ok=True)
        self.valid_path.mkdir(parents=True, exist_ok=True)
        self.augmentation_pipeline = get_albumentations_transforms(image_size)
        self.basic_transform = get_basic_transform(image_size)
        self.dataset_stats = {}
        print(f"Initialized DatasetPreprocessor")
        print(f"Input dataset: {self.dataset_path}")
        print(f"Output directory: {self.output_path}")
        print(f"Image size: {image_size}x{image_size}")

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
        train_images, valid_images = split_dataset(image_paths)
        print(f"\nSplit Summary:")
        print(f"Training images: {len(train_images)}")
        print(f"Validation images: {len(valid_images)}")
        print(f"Total: {len(train_images) + len(valid_images)}")
        self.process_split(train_images, 'train', class_counts)
        self.process_split(valid_images, 'valid', class_counts, augment=False)
        self.save_dataset_info()

    def process_split(self, image_list, split_name, class_counts, augment=True):
        split_dir = self.train_path if split_name == 'train' else self.valid_path
        for img_path, class_name in tqdm(image_list, desc=f'Processing {split_name} set'):
            class_dir = split_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            out_path = class_dir / img_path.name
            save_image(self.basic_transform(image=img)['image'], out_path)
            if augment and split_name == 'train':
                frac, n_aug = get_augmentation_strategy(class_counts, class_name)
                n_to_aug = int(len(image_list) * frac)
                if n_to_aug > 0:
                    for i in range(n_aug):
                        aug_img = self.augmentation_pipeline(image=img)['image']
                        aug_path = class_dir / f"aug_{i}_{img_path.name}"
                        save_image(aug_img, aug_path)

    def save_dataset_info(self):
        info = {
            'class_counts': self.dataset_stats['class_counts'],
            'total_images': self.dataset_stats['total_images'],
            'num_classes': self.dataset_stats['num_classes'],
            'preprocessing': {
                'image_size': f"{self.image_size}x{self.image_size}",
                'augmentation': True
            }
        }
        with open(self.output_path / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=2)


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
            image_size=args.image_size
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
    
    # Call the preprocessing function
    run_preprocessing(args)


if __name__ == "__main__":
    main()