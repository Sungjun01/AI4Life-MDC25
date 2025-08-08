# Complete Training Dataset

This folder contains the complete microscopy image denoising training dataset used to train the Advanced CARE U-Net model for the AI4Life MDC25 challenge.

## Dataset Structure

```
data/
├── large_images/
│   ├── noisy/     # 20 large noisy images (approx. 512x512+ pixels)
│   └── gt/        # 20 corresponding ground truth clean images
├── small_images/
│   ├── noisy/     # 2,457 small noisy images (256x256 pixels)  
│   └── gt/        # 2,457 corresponding ground truth clean images
└── README.md      # This documentation
```

## Dataset Statistics

- **Total Images**: 4,954 TIFF files
- **Image Pairs**: 2,477 noisy/clean pairs
- **Large Images**: 40 files (20 pairs)
- **Small Images**: 4,914 files (2,457 pairs)
- **Format**: 16-bit TIFF files
- **Content**: Fluorescence microscopy images with synthetic noise

## Training Information

### Model: Advanced CARE U-Net
- **Architecture**: U-Net + attention mechanisms
- **Parameters**: 31.2 million trainable parameters  
- **Configuration**: base_channels=64, depth=4
- **Performance**: 27.2 dB PSNR, 0.776 SSIM

### Training Strategy
- **Method**: Noise2Clean supervised learning
- **Data Split**: 80% training, 20% validation
- **Augmentation**: Random flips, rotations, intensity scaling
- **Loss Function**: L1 + SSIM loss combination
- **Optimizer**: Adam with learning rate 1e-3

## Image Specifications

### Small Images (256x256)
- Primary training dataset  
- Faster training iterations
- Good for attention mechanism learning
- Files: 1000.tif - 3456.tif

### Large Images (512x512+)
- High-resolution validation
- Real-world size testing  
- Memory-intensive processing
- Files: 1000.tif - 1019.tif

## Data Quality

- **Noise Type**: Realistic Gaussian + Poisson noise simulation
- **SNR Range**: 5-25 dB across different images
- **Intensity Range**: Normalized to [0,1] for training
- **Ground Truth**: Clean reference images without noise

## Usage Instructions

1. **For Training**: Use both small and large images with data augmentation
2. **For Validation**: Test on held-out subset of each size category  
3. **For Inference**: Model works with any image size through adaptive padding
4. **For Comparison**: Use corresponding GT images to compute metrics (PSNR, SSIM)

This comprehensive dataset enabled the Advanced CARE U-Net to achieve state-of-the-art denoising performance on microscopy images.