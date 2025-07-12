# Temple Image Classification Project

This project provides a complete pipeline for temple image classification using a custom, multi-class dataset. The repository includes scripts for preprocessing and several advanced training strategies with detailed control over transfer learning and class imbalance.

## Dataset Overview

The dataset consists of temple images from various countries and regions, organized by folder:

| Country/Region                | Number of Images[1] |
|-------------------------------|---------------------|
| Armenia                       | 11                  |
| Australia                     | 36                  |
| Germany                       | 90                  |
| Hungary+Slovakia+Croatia      | 48                  |
| Indonesia-Bali                | 44                  |
| Japan                         | 60                  |
| Malaysia+Indonesia            | 56                  |
| Portugal+Brazil               | 54                  |
| Russia                        | 100                 |
| Spain                         | 65                  |
| Thailand                      | 101                 |

Each folder is a class label containing images from that region.

## Project Structure

### Preprocessing

**Script:** `preprocess.py`

- Scans the raw dataset, applies augmentations, and splits into training and validation sets.
- Augmentation is adaptive: more aggressive for smaller classes.
- Images are resized to 512x512 and normalized to ImageNet stats.
- Output: `processed_dataset/train/` and `processed_dataset/valid/` folders, plus a `dataset_info.json` summary.

**Run:**
```bash
python preprocess.py
```

### Training Scripts

| Script                | Model(s)           | Loss Function(s)             | Unfreezing Strategy           | Class Imbalance Handling           | Notes                        |
|-----------------------|--------------------|------------------------------|-------------------------------|------------------------------------|------------------------------|
| train.py              | 1 (choice)         | Weighted CE / Focal Loss     | None (all layers trainable)   | Class weights, Focal Loss          | Simple baseline              |
| train_unfreeze.py     | 1 (choice)         | Weighted CE / Focal Loss     | 3-stage: layer4/features.7    | Class weights, Focal Loss          | Staged unfreezing            |
| train_unfreeze_2.py   | 1 (choice)         | Class-Balanced Focal Loss    | 3-stage: layer4/features.7    | Class-Balanced Focal Loss          | For imbalanced data          |
| train_unfreeze_3.py   | 1 (choice)         | Class-Balanced Focal Loss    | 3-stage: layer3/features.6    | Class-Balanced Focal Loss          | Deeper staged unfreezing     |
| train_ensamble.py     | 2 (ensemble)       | Class-Balanced Focal Loss    | 3-stage: layer4/features.7    | Class-Balanced Focal Loss          | Ensemble of two models       |

## Detailed Training Script Differences

### `train_unfreeze.py`
- **Loss Function:** Weighted Cross Entropy or Focal Loss (configurable).
- **Unfreezing:**  
  - **Epochs 0–9:** Only classifier layers trainable; backbone frozen.
  - **Epochs 10–24:** Unfreezes deepest backbone block (`layer4` for ResNet50).
  - **Epoch 25+:** All layers trainable.
- **Use:** Standard staged transfer learning.

### `train_unfreeze_2.py`
- **Loss Function:** Class-Balanced Focal Loss only.
- **Unfreezing:** Same staged schedule as above.
- **Use:** When class imbalance is significant; loss function improves minority class performance.

### `train_unfreeze_3.py`
- **Loss Function:** Class-Balanced Focal Loss only.
- **Unfreezing:**  
  - **Epochs 0–9:** Only classifier layers trainable.
  - **Epochs 10–24:** Unfreezes a *deeper* block (`layer3` for ResNet50).
  - **Epoch 25+:** All layers trainable.
- **Use:** For advanced transfer learning, allowing more mid-level features to adapt earlier.

## How to Run

All scripts expect the processed dataset in `./processed_dataset`.

### Preprocessing
```bash
python preprocess.py
```

### Basic Training
```bash
python train.py --model resnet50 --batch_size 32 --epochs 50 --focal_loss --gamma 2.0
```

### Staged Unfreezing
```bash
python train_unfreeze.py --model resnet50 --batch_size 32 --epochs 50 --focal_loss --gamma 2.0
```

### Class-Balanced Focal Loss
```bash
python train_unfreeze_2.py --model resnet50 --batch_size 32 --epochs 50 --gamma 2.0 --beta 0.9999
```

### Deeper Staged Unfreezing
```bash
python train_unfreeze_3.py --model resnet50 --batch_size 32 --epochs 50 --gamma 2.0 --beta 0.9999
```

### Ensemble Training
```bash
python train_ensamble.py --batch_size 32 --epochs 50 --gamma 2.0 --beta 0.9999
```

**Common arguments:**
- `--model`: `resnet50` (only ResNet50 supported)
- `--focal_loss` / `--gamma`: for Focal Loss
- `--beta`: for Class-Balanced Focal Loss
- `--save_dir`: output directory for models/logs

## Outputs

- **Models:** Saved to `--save_dir`
- **Logs:** Training history and metrics as JSON
- **Plots:** Training curves and confusion matrix as PNGs

## Notes

- Install dependencies: PyTorch, torchvision, albumentations, etc.
- GPU recommended for training.
- Adjust hyperparameters for your hardware and dataset size.

## Device Compatibility

All models are trained to work on any device (GPU or CPU):

- **Model Loading**: All checkpoints are loaded with `map_location='cpu'` for universal compatibility
- **Device Handling**: Models automatically adapt to available hardware (GPU/CPU)
- **Inference**: Prediction scripts work seamlessly on both GPU and CPU

### Deployment

For deployment on any system:
1. Train models on GPU (recommended for speed)
2. Use the same prediction scripts - they automatically handle device inference
3. No additional conversion needed

## Summary Table

| Script                | Loss Function(s)           | Stage 2 Unfreezing         | Class Imbalance Handling     | Recommended Use                  |
|-----------------------|---------------------------|----------------------------|-----------------------------|----------------------------------|
| train_unfreeze.py     | Weighted CE / Focal Loss  | layer4                     | Class weights, Focal Loss   | Standard staged unfreezing       |
| train_unfreeze_2.py   | Class-Balanced Focal Loss | layer4                     | Class-Balanced Focal Loss   | Imbalanced data                  |
| train_unfreeze_3.py   | Class-Balanced Focal Loss | layer3 (deeper)            | Class-Balanced Focal Loss   | Advanced staged unfreezing       |
