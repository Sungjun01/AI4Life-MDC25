#!/usr/bin/env python3
"""
Export advanced_care_unet model to TorchScript for Grand Challenge submission.
"""

import torch
import sys
from pathlib import Path

# Add denoising module to path
sys.path.append(str(Path(__file__).parent / "denoising"))

from models.cnn.care import AdvancedCAREUNet

def export_model():
    """Export the trained advanced_care_unet model to TorchScript format."""
    
    # Model parameters (match training configuration: base_channels=64, depth=4)
    model = AdvancedCAREUNet(
        in_channels=1,
        out_channels=1, 
        base_channels=64,
        depth=4
    )
    
    # Load trained weights
    model_path = Path("denoising/results/advanced_care_unet_best_model.pth")
    print(f"Loading model from: {model_path}")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load state dict with weights_only=False for compatibility
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create example input (typical microscopy image size)
    example_input = torch.randn(1, 1, 512, 512)
    
    # Trace the model
    print("Tracing model with TorchScript...")
    with torch.no_grad():
        traced_model = torch.jit.trace(model, example_input)
    
    # Save the traced model
    output_path = Path("submission_model.pth")
    torch.jit.save(traced_model, output_path)
    
    print(f"Model exported successfully to: {output_path}")
    print(f"Model size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Test the exported model
    print("Testing exported model...")
    loaded_model = torch.jit.load(output_path)
    
    with torch.no_grad():
        original_output = model(example_input)
        traced_output = loaded_model(example_input)
        
    # Check if outputs are close
    if torch.allclose(original_output, traced_output, atol=1e-6):
        print("✓ Model export verification successful!")
    else:
        print("✗ Model export verification failed!")
        return False
    
    return True

if __name__ == "__main__":
    success = export_model()
    if not success:
        sys.exit(1)