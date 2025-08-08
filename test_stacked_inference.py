#!/usr/bin/env python3
"""
Test the stacked image inference pipeline
"""

import torch
import tifffile
import numpy as np
from pathlib import Path
import sys

# Add submission path
sys.path.append("AI4Life-MDC25-submission")

def test_stacked_inference():
    """Test inference with stacked TIFF images"""
    
    # Load model
    model_path = Path("submission_model.pth")
    print(f"Loading model: {model_path}")
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()
    
    # Find stacked test images
    test_dir = Path("AI4Life-MDC25-submission/test/input/interface_0/images/image-stack-unstructured-noise")
    stack_files = list(test_dir.glob("*.tif"))
    
    print(f"Found {len(stack_files)} stacked image files:")
    for f in stack_files:
        print(f"  - {f.name}")
    
    if not stack_files:
        print("No stacked test files found!")
        return False
    
    # Test with first stack
    test_stack_path = stack_files[0]
    print(f"\nTesting with: {test_stack_path}")
    
    # Load stack
    image_stack = tifffile.imread(test_stack_path).astype(np.float32)
    
    # Normalize
    if image_stack.max() > 1.0:
        image_stack = image_stack / image_stack.max()
    
    print(f"Stack shape: {image_stack.shape}")
    print(f"Stack range: [{image_stack.min():.3f}, {image_stack.max():.3f}]")
    
    # Process first image from stack
    if image_stack.ndim == 3:
        test_image = image_stack[0]  # First image from stack
        print(f"Testing with single image shape: {test_image.shape}")
        
        # Convert to tensor
        if test_image.ndim == 2:
            input_tensor = torch.from_numpy(test_image).unsqueeze(0).unsqueeze(0)
        
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Run inference
        with torch.no_grad():
            output = model(input_tensor)
            
            if output.dim() == 4:
                output = output.squeeze(0)
            if output.dim() == 3 and output.shape[0] == 1:
                output = output.squeeze(0)
            
            output_array = output.numpy()
        
        print(f"Output shape: {output_array.shape}")
        print(f"Output range: [{output_array.min():.3f}, {output_array.max():.3f}]")
        
        # Save test result
        output_path = Path("test_stack_output.tif")
        tifffile.imwrite(output_path, output_array)
        print(f"Test output saved: {output_path}")
        
        print("✅ Stacked inference test passed!")
        return True
    
    else:
        print("❌ Expected 3D stack but got different shape")
        return False

if __name__ == "__main__":
    try:
        success = test_stacked_inference()
        if success:
            print("\n🎉 Stacked image inference working correctly!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()