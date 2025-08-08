#!/usr/bin/env python3
"""
Validate that the model works with actual microscopy data
"""

import torch
import tifffile
import numpy as np
from pathlib import Path

def test_with_real_data():
    """Test model with actual microscopy images"""
    
    # Load the exported model
    model_path = Path("submission_model.pth")
    print(f"Loading model: {model_path}")
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()
    
    # Test with actual microscopy data
    test_image_path = Path("AI4Life-MDC25-submission/test/input/interface_0/images/image-stack-unstructured-noise/1000.tif")
    
    if not test_image_path.exists():
        print(f"Test image not found: {test_image_path}")
        return False
    
    # Load and preprocess image
    print(f"Loading test image: {test_image_path}")
    image = tifffile.imread(test_image_path)
    image = image.astype(np.float32)
    
    # Normalize to [0,1]
    if image.max() > 1.0:
        image = image / image.max()
    
    print(f"Image shape: {image.shape}, range: [{image.min():.3f}, {image.max():.3f}]")
    
    # Convert to tensor with proper dimensions
    if image.ndim == 2:
        input_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    else:
        input_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, C, H, W)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        
        # Remove batch dimension
        if output.dim() == 4:
            output = output.squeeze(0)
        if output.dim() == 3 and output.shape[0] == 1:
            output = output.squeeze(0)
        
        output_array = output.numpy()
    
    print(f"Output shape: {output_array.shape}, range: [{output_array.min():.3f}, {output_array.max():.3f}]")
    
    # Save test output
    output_path = Path("test_denoised_1000.tif")
    tifffile.imwrite(output_path, output_array.astype(np.float32))
    print(f"Test output saved: {output_path}")
    
    # Validate output
    assert output_array.shape == image.shape, f"Shape mismatch: {output_array.shape} vs {image.shape}"
    assert 0 <= output_array.min() and output_array.max() <= 1, f"Output range invalid: [{output_array.min():.3f}, {output_array.max():.3f}]"
    
    print("✓ Real data test passed!")
    return True

if __name__ == "__main__":
    try:
        success = test_with_real_data()
        if success:
            print("\n🎉 Model validation with real microscopy data successful!")
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()