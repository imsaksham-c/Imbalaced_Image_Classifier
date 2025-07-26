#!/usr/bin/env python3
"""
Test script for preprocessing with test split functionality
"""

import os
import shutil
from pathlib import Path
from preprocess import run_preprocessing
import argparse

def create_test_dataset():
    """Create a small test dataset for testing."""
    test_dataset = Path("./test_dataset")
    if test_dataset.exists():
        shutil.rmtree(test_dataset)
    
    test_dataset.mkdir()
    
    # Create some test classes
    classes = ["class1", "class2", "class3"]
    for class_name in classes:
        class_dir = test_dataset / class_name
        class_dir.mkdir()
        
        # Create some dummy image files
        for i in range(10):  # 10 images per class
            img_file = class_dir / f"image_{i}.jpg"
            img_file.touch()  # Create empty file for testing
    
    return test_dataset

def test_preprocessing():
    """Test preprocessing with and without test split."""
    print("Testing preprocessing functionality...")
    
    # Create test dataset
    test_dataset = create_test_dataset()
    
    # Test 1: Without test split (default behavior)
    print("\n" + "="*50)
    print("TEST 1: Without test split")
    print("="*50)
    
    args = argparse.Namespace()
    args.dataset = str(test_dataset)
    args.output = "./test_output_no_split"
    args.test_size = 0.2
    args.test_split = False
    args.test_size_ratio = 0.1
    args.image_size = 512
    args.overwrite = True
    
    success = run_preprocessing(args)
    print(f"Test 1 result: {'PASSED' if success else 'FAILED'}")
    
    # Check output structure
    output_dir = Path(args.output)
    if output_dir.exists():
        print("Output structure:")
        for item in output_dir.iterdir():
            if item.is_dir():
                print(f"  {item.name}/")
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        print(f"    {subitem.name}/ ({len(list(subitem.iterdir()))} files)")
    
    # Test 2: With test split
    print("\n" + "="*50)
    print("TEST 2: With test split")
    print("="*50)
    
    args.output = "./test_output_with_split"
    args.test_split = True
    args.test_size_ratio = 0.1
    
    success = run_preprocessing(args)
    print(f"Test 2 result: {'PASSED' if success else 'FAILED'}")
    
    # Check output structure
    output_dir = Path(args.output)
    if output_dir.exists():
        print("Output structure:")
        for item in output_dir.iterdir():
            if item.is_dir():
                print(f"  {item.name}/")
                for subitem in item.iterdir():
                    if subitem.is_dir():
                        print(f"    {subitem.name}/ ({len(list(subitem.iterdir()))} files)")
    
    # Cleanup
    print("\nCleaning up test files...")
    if test_dataset.exists():
        shutil.rmtree(test_dataset)
    if Path("./test_output_no_split").exists():
        shutil.rmtree("./test_output_no_split")
    if Path("./test_output_with_split").exists():
        shutil.rmtree("./test_output_with_split")
    
    print("Test completed!")

if __name__ == "__main__":
    test_preprocessing() 