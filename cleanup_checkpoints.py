#!/usr/bin/env python3
"""
Checkpoint Cleanup Script
=========================

This script removes optimizer state dict from existing model checkpoints
to reduce file size from ~205MB to ~98MB.

Examples:
  python cleanup_checkpoints.py --model_path models/best_model_resnet50_unfreeze0.pth
  python cleanup_checkpoints.py --models_dir models/ --cleanup_all
  python cleanup_checkpoints.py --model_path model.pth --output_path clean_model.pth
"""

import torch
import argparse
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

def cleanup_all_models(models_dir):
    """
    Clean all .pth files in a directory.
    
    Args:
        models_dir (str): Directory containing model checkpoints
    """
    print(f"🔍 Scanning directory: {models_dir}")
    
    # Find all .pth files
    pattern = os.path.join(models_dir, "**/*.pth")
    model_files = glob.glob(pattern, recursive=True)
    
    if not model_files:
        print("❌ No .pth files found in the directory")
        return
    
    print(f"📁 Found {len(model_files)} model files")
    print()
    
    total_original_size = 0
    total_new_size = 0
    
    for model_path in model_files:
        # Create output path with '_clean' suffix
        base_path = Path(model_path)
        output_path = base_path.parent / f"{base_path.stem}_clean{base_path.suffix}"
        
        # Clean the checkpoint
        cleanup_checkpoint(model_path, output_path)
        
        # Track total sizes
        total_original_size += os.path.getsize(model_path) / (1024 * 1024)
        total_new_size += os.path.getsize(output_path) / (1024 * 1024)
    
    # Print summary
    print("=" * 60)
    print("📊 CLEANUP SUMMARY")
    print("=" * 60)
    print(f"📁 Total files processed: {len(model_files)}")
    print(f"📊 Total original size: {total_original_size:.1f} MB")
    print(f"📊 Total new size: {total_new_size:.1f} MB")
    print(f"📉 Total space saved: {total_original_size - total_new_size:.1f} MB")
    print(f"💾 Clean models saved with '_clean' suffix")

def cleanup_models(args):
    """
    Main cleanup function that handles the cleanup logic.
    
    Args:
        args: Parsed command line arguments
    """
    if args.model_path:
        # Clean single model
        if not os.path.exists(args.model_path):
            print(f"❌ Model file not found: {args.model_path}")
            return
        
        cleanup_checkpoint(args.model_path, args.output_path)
        
    elif args.models_dir and args.cleanup_all:
        # Clean all models in directory
        if not os.path.exists(args.models_dir):
            print(f"❌ Directory not found: {args.models_dir}")
            return
        
        cleanup_all_models(args.models_dir)
        
    else:
        print("❌ Please specify either --model_path or --models_dir with --cleanup_all")
        return False
    
    return True

def main():
    """
    Main function that handles argument parsing and calls the cleanup function.
    """
    # ============================================================================
    # ARGUMENT PARSING
    # ============================================================================
    parser = argparse.ArgumentParser(
        description='🧹 Clean model checkpoints by removing optimizer state dict',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model_path', type=str, 
                       help='Path to single model checkpoint to clean')
    parser.add_argument('--output_path', type=str, 
                       help='Output path for cleaned model (optional)')
    parser.add_argument('--models_dir', type=str, 
                       help='Directory containing multiple model checkpoints')
    parser.add_argument('--cleanup_all', action='store_true',
                       help='Clean all .pth files in the specified directory')
    
    args = parser.parse_args()
    
    # Call the cleanup function
    cleanup_models(args)

if __name__ == "__main__":
    main() 