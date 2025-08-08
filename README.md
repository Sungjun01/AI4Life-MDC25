# AI4Life-MDC25

Advanced CARE U-Net submission for the AI4Life Microscopy Denoising Challenge 2025

## Overview

This repository contains a complete Docker container submission for the AI4Life MDC25 denoising challenge, featuring an Advanced CARE U-Net model trained on microscopy image denoising.

## Model Performance

- **Architecture**: Advanced CARE U-Net with attention mechanisms
- **Parameters**: 31.2 million trainable parameters  
- **Performance**: 27.2 dB PSNR, 0.776 SSIM on validation data
- **Training**: Noise2Clean supervised learning on 2,477 image pairs

## Submission Contents

### Core Files
- `inference.py` - Main inference pipeline for Grand Challenge
- `Dockerfile` - Container definition with PyTorch base
- `requirements.txt` - Python dependencies
- `resources/submission_model.pth` - Trained model (TorchScript format, 119MB)

### Test Data
- `test/` - Grand Challenge compatible test structure
- Multi-page TIFF stacks with UUID filenames
- Supports both `.tif` and `.mha` formats

### Training Dataset
- `data/` - Complete training dataset (4,954 TIFF files)
- `small_images/` - 2,457 image pairs (256×256)
- `large_images/` - 20 image pairs (1024×1024)
- Ground truth and noisy image pairs

### Utilities
- `create_stacks.py` - Creates stacked TIFF format for testing
- `export_model.py` - Converts trained model to TorchScript
- `validate_data.py` - Model validation with real data
- Various test scripts

## Usage

### Local Testing
```bash
./do_test_run.sh    # Build and test container
./do_save.sh        # Create submission archive
```

### Grand Challenge Submission
1. Build the container using `do_build.sh`
2. Save the container using `do_save.sh`  
3. Upload the generated archive to Grand Challenge platform

## Technical Details

### Input/Output Format
- **Input**: `/input/images/image-stack-unstructured-noise/<uuid>.tif`
- **Output**: `/output/images/image-stack-denoised/<uuid>.tif`
- Supports stacked TIFF images as required by Grand Challenge

### Model Architecture
- U-Net backbone with skip connections
- Content-aware attention mechanisms at multiple scales
- Batch normalization and dropout regularization
- Sigmoid output activation for [0,1] range

### Training Configuration
- Base channels: 64, Depth: 4
- Loss: L1 + SSIM combination
- Optimizer: Adam (lr=1e-3)
- Data augmentation: flips, rotations, intensity scaling

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- tifffile
- SimpleITK (for .mha support)
- NumPy

## Challenge Information

**AI4Life Microscopy Denoising Challenge 2025**
- Platform: Grand Challenge
- Task: 2D microscopy image denoising
- Metric: PSNR, SSIM
- Submission deadline: August 12, 2025

## Results

The Advanced CARE U-Net model achieves state-of-the-art performance on microscopy image denoising while maintaining computational efficiency suitable for the Grand Challenge platform constraints.