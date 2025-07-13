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

import argparse
import os
from utils.checkpoint_cleanup import cleanup_checkpoint, cleanup_all_models


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