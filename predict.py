"""
Model Prediction Script
======================

This script loads trained models and makes predictions on images.
Supports both single image and folder predictions.

Examples:
  python predict.py --model_path ./models/best_model_resnet50_unfreeze0.pth --image_path ./test_image.jpg
  python predict.py --model_path ./models/best_model_resnet50_unfreeze0.pth --folder_path ./test_images/
  python predict.py --model_path ./models/best_model_resnet50_unfreeze0.pth --folder_path ./test_images/ --output_path ./my_predictions/
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
from PIL import Image
import argparse
import json
from pathlib import Path
import cv2
from tqdm import tqdm
import datetime
import warnings
import matplotlib.pyplot as plt
import pandas as pd

from utils.dataset import get_transforms

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def get_model(model_name, num_classes, fc_layers=None):
    """Initialize model architecture."""
    if model_name == 'resnet50':
        model = resnet50(pretrained=False)
        if fc_layers:
            layers = []
            in_features = model.fc.in_features
            
            for i, layer_size in enumerate(fc_layers):
                layers.extend([
                    nn.Linear(in_features if i == 0 else fc_layers[i-1], layer_size),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm1d(layer_size),
                    nn.Dropout(0.5)
                ])
                in_features = layer_size
            
            layers.append(nn.Linear(fc_layers[-1], num_classes))
            model.fc = nn.Sequential(*layers)
        else:
            model.fc = nn.Sequential(
                nn.Dropout(0.6),
                nn.Linear(model.fc.in_features, num_classes)
            )
            
    else:
        raise ValueError(f"Model {model_name} not supported (only resnet50 supported)")
    
    return model.to(device)





def load_model(model_path):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {model_path}")
    
    # Load checkpoint with CPU compatibility
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extract model information
    model_name = checkpoint.get('model_name', 'resnet50')
    class_names = checkpoint.get('class_names', [])
    unfreeze_mode = checkpoint.get('unfreeze_mode', 0)
    
    # Get model configuration from training config if available
    config_path = Path(model_path).parent.parent / 'logs' / 'training_config.json'
    fc_layers = None
    image_size = 512
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            fc_layers = config.get('fc_layers')
            # Try to get image size from dataset info
            dataset_info_path = Path(config.get('data_dir', 'processed_dataset')) / 'dataset_info.json'
            if dataset_info_path.exists():
                with open(dataset_info_path, 'r') as f:
                    dataset_info = json.load(f)
                    image_size_str = dataset_info['preprocessing']['image_size']
                    image_size = int(image_size_str.split('x')[0])
    
    print(f"Model: {model_name}")
    print(f"Classes: {len(class_names)}")
    print(f"Image size: {image_size}x{image_size}")
    if fc_layers:
        print(f"FC layers: {fc_layers}")
    
    # Initialize model
    model = get_model(model_name, len(class_names), fc_layers)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, class_names, image_size


def predict_single_image(model, image_path, transform, class_names):
    """Make prediction on a single image."""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        
        results = {
            'predicted_class': class_names[predicted_class],
            'confidence': confidence,
            'top3_predictions': [
                {
                    'class': class_names[idx.item()],
                    'probability': prob.item()
                }
                for prob, idx in zip(top3_probs, top3_indices)
            ]
        }
        
        return results, image
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None


def predict_folder(model, folder_path, transform, class_names, output_path):
    """Make predictions on all images in a folder."""
    results = []
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f'*{ext}'))
        image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return results
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for image_path in tqdm(image_files, desc="Processing images"):
        result, image = predict_single_image(model, image_path, transform, class_names)
        
        if result:
            result['image_path'] = str(image_path)
            result['image_name'] = image_path.name
            results.append(result)
            
            # Save prediction visualization
            if image:
                save_prediction_visualization(image, result, output_path / f"pred_{image_path.stem}.png")
    
    return results


def save_prediction_visualization(image, result, output_path):
    """Save prediction visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title(f"Predicted: {result['predicted_class']}\nConfidence: {result['confidence']:.3f}")
    ax1.axis('off')
    
    # Top 3 predictions bar chart
    classes = [pred['class'] for pred in result['top3_predictions']]
    probabilities = [pred['probability'] for pred in result['top3_predictions']]
    
    bars = ax2.bar(range(len(classes)), probabilities, color=['red', 'orange', 'blue'])
    ax2.set_xlabel('Classes')
    ax2.set_ylabel('Probability')
    ax2.set_title('Top 3 Predictions')
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    
    # Add probability values on bars
    for bar, prob in zip(bars, probabilities):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results(results, output_path):
    """Save prediction results to CSV."""
    if not results:
        print("No results to save")
        return
    
    # Create simple CSV with filename and detected class
    csv_data = []
    for result in results:
        csv_data.append({
            'filename': result['image_name'],
            'detected_country': result['predicted_class']
        })
    
    df = pd.DataFrame(csv_data)
    
    # Save CSV
    csv_path = output_path / 'result.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Print summary
    print(f"\nPrediction Summary:")
    print(f"Total images processed: {len(results)}")
    
    # Class distribution
    class_counts = df['detected_country'].value_counts()
    print(f"\nPredicted class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")


def run_prediction(args):
    """
    Main prediction function that handles the prediction logic.
    
    Args:
        args: Parsed command line arguments
    """
    # Validate arguments
    if not args.image_path and not args.folder_path:
        print("Error: Please provide either --image_path or --folder_path")
        return False
    
    if args.image_path and args.folder_path:
        print("Error: Please provide either --image_path or --folder_path, not both")
        return False
    
    # Set output path
    if args.output_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"runs/{timestamp}")
    else:
        output_path = Path(args.output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("Model Prediction Script")
    print("=" * 50)
    print(f"Output directory: {output_path}")
    
    try:
        # Load model
        model, class_names, image_size = load_model(args.model_path)
        
        # Setup transforms
        transform = get_transforms(image_size)
        
        # Make predictions
        if args.image_path:
            print(f"\nProcessing single image: {args.image_path}")
            result, image = predict_single_image(model, args.image_path, transform, class_names)
            
            if result:
                print(f"\nPrediction Results:")
                print(f"Image: {args.image_path}")
                print(f"Predicted Class: {result['predicted_class']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"\nTop 3 Predictions:")
                for i, pred in enumerate(result['top3_predictions']):
                    print(f"  {i+1}. {pred['class']}: {pred['probability']:.3f}")
                
                # Save visualization
                if image:
                    save_prediction_visualization(image, result, output_path / "prediction.png")
                
                # Save results
                result['image_path'] = args.image_path
                result['image_name'] = Path(args.image_path).name
                save_results([result], output_path)
        
        elif args.folder_path:
            print(f"\nProcessing folder: {args.folder_path}")
            results = predict_folder(model, args.folder_path, transform, class_names, output_path)
            save_results(results, output_path)
        
        print(f"\nPrediction completed successfully!")
        print(f"Results saved to: {output_path}")
        print(f"CSV file: {output_path}/result.csv")
        
        return True
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return False


def main():
    """
    Main function that handles argument parsing and calls the prediction function.
    """
    # ============================================================================
    # ARGUMENT PARSING
    # ============================================================================
    parser = argparse.ArgumentParser(
        description='Model Prediction Script'
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint')
    parser.add_argument('--image_path', type=str, default=None,
                       help='Path to a single image for prediction')
    parser.add_argument('--folder_path', type=str, default=None,
                       help='Path to folder containing images for prediction')
    parser.add_argument('--output_path', type=str, default=None,
                       help='Path to save prediction results (default: runs/<timestamp>)')
    
    args = parser.parse_args()
    
    # Call the prediction function
    run_prediction(args)


if __name__ == "__main__":
    main() 