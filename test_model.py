#!/usr/bin/env python3
"""
Test script to validate the exported model can be loaded and run.
"""

import torch
import numpy as np
from pathlib import Path

def test_model_loading():
    """Test that the exported model can be loaded and produces valid outputs."""
    
    model_path = Path("submission_model.pth")
    
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return False
    
    print(f"Loading model from: {model_path}")
    print(f"Model size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    try:
        # Load model
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()
        print("✓ Model loaded successfully")
        
        # Create test input (microscopy image size)
        test_input = torch.randn(1, 1, 256, 256)
        print(f"Test input shape: {test_input.shape}")
        
        # Run inference
        with torch.no_grad():
            output = model(test_input)
        
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # Validate output
        assert output.shape == test_input.shape, f"Shape mismatch: {output.shape} vs {test_input.shape}"
        assert 0 <= output.min() and output.max() <= 1, f"Output not in [0,1]: [{output.min():.3f}, {output.max():.3f}]"
        
        print("✓ Model inference test passed")
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 Model is ready for submission!")
    else:
        print("\n❌ Model validation failed")