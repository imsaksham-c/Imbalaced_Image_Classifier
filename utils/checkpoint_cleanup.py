"""
Checkpoint Cleanup Utilities
============================

This module provides utilities for cleaning model checkpoints by removing
optimizer state dict to reduce file size from ~205MB to ~98MB.

Functions:
    cleanup_checkpoint: Remove optimizer state dict from a single checkpoint
    cleanup_all_models: Clean all .pth files in a directory
"""

import torch
import os
import glob
from pathlib import Path


def cleanup_checkpoint(model_path, output_path=None):
    """
    Remove optimizer state dict from a checkpoint to reduce file size.
    
    Args:
        model_path (str): Path to the original checkpoint
        output_path (str, optional): Path for the cleaned checkpoint. 
                                   If None, overwrites original with '_clean' suffix
    
    Returns:
        tuple: (original_size_mb, new_size_mb, size_reduction_mb, output_path)
    """
    print(f"🧹 Cleaning checkpoint: {model_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get original file size
    original_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    
    # Remove optimizer state dict if it exists
    if 'optimizer_state_dict' in checkpoint:
        del checkpoint['optimizer_state_dict']
        print(f"   ✅ Removed optimizer state dict")
    else:
        print(f"   ℹ️  No optimizer state dict found")
    
    # Determine output path
    if output_path is None:
        base_path = Path(model_path)
        output_path = base_path.parent / f"{base_path.stem}_clean{base_path.suffix}"
    
    # Save cleaned checkpoint
    torch.save(checkpoint, output_path)
    
    # Get new file size
    new_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    size_reduction = original_size - new_size
    
    print(f"   📊 Original size: {original_size:.1f} MB")
    print(f"   📊 New size: {new_size:.1f} MB")
    print(f"   📉 Size reduction: {size_reduction:.1f} MB ({size_reduction/original_size*100:.1f}%)")
    print(f"   💾 Saved to: {output_path}")
    print()
    
    return original_size, new_size, size_reduction, str(output_path)


def cleanup_all_models(models_dir):
    """
    Clean all .pth files in a directory.
    
    Args:
        models_dir (str): Directory containing model checkpoints
    
    Returns:
        dict: Summary statistics of the cleanup operation
    """
    print(f"🔍 Scanning directory: {models_dir}")
    
    # Find all .pth files
    pattern = os.path.join(models_dir, "**/*.pth")
    model_files = glob.glob(pattern, recursive=True)
    
    if not model_files:
        print("❌ No .pth files found in the directory")
        return {
            'files_processed': 0,
            'total_original_size': 0,
            'total_new_size': 0,
            'total_space_saved': 0
        }
    
    print(f"📁 Found {len(model_files)} model files")
    print()
    
    total_original_size = 0
    total_new_size = 0
    cleaned_files = []
    
    for model_path in model_files:
        # Create output path with '_clean' suffix
        base_path = Path(model_path)
        output_path = base_path.parent / f"{base_path.stem}_clean{base_path.suffix}"
        
        # Clean the checkpoint
        orig_size, new_size, reduction, output_path_str = cleanup_checkpoint(model_path, output_path)
        
        # Track totals
        total_original_size += orig_size
        total_new_size += new_size
        cleaned_files.append({
            'original_path': model_path,
            'cleaned_path': output_path_str,
            'original_size': orig_size,
            'new_size': new_size,
            'reduction': reduction
        })
    
    # Print summary
    print("=" * 60)
    print("📊 CLEANUP SUMMARY")
    print("=" * 60)
    print(f"📁 Total files processed: {len(model_files)}")
    print(f"📊 Total original size: {total_original_size:.1f} MB")
    print(f"📊 Total new size: {total_new_size:.1f} MB")
    print(f"📉 Total space saved: {total_original_size - total_new_size:.1f} MB")
    print(f"💾 Clean models saved with '_clean' suffix")
    
    return {
        'files_processed': len(model_files),
        'total_original_size': total_original_size,
        'total_new_size': total_new_size,
        'total_space_saved': total_original_size - total_new_size,
        'cleaned_files': cleaned_files
    }


def cleanup_experiment_models(experiment_dir):
    """
    Clean all model checkpoints in an experiment directory.
    
    Args:
        experiment_dir (str): Path to the experiment directory
    
    Returns:
        dict: Summary statistics of the cleanup operation
    """
    models_dir = os.path.join(experiment_dir, 'models')
    
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return None
    
    print(f"🧹 Cleaning checkpoints for experiment: {experiment_dir}")
    print("=" * 60)
    
    return cleanup_all_models(models_dir) 