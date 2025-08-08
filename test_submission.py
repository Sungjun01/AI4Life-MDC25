#!/usr/bin/env python3
"""
Test script to simulate the Grand Challenge container environment
"""

import os
import sys
import shutil
from pathlib import Path

# Add submission path to run inference
sys.path.append("AI4Life-MDC25-submission")

def setup_container_environment():
    """Simulate the container's file system structure"""
    
    # Create input/output directories
    input_path = Path("/tmp/test_container/input/images/image-stack-unstructured-noise")
    output_path = Path("/tmp/test_container/output/images/image-stack-denoised")
    input_json_path = Path("/tmp/test_container/input")
    
    input_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy test images
    test_images = Path("AI4Life-MDC25-submission/test/input/interface_0/images/image-stack-unstructured-noise")
    for image_file in test_images.glob("*.tif"):
        shutil.copy2(image_file, input_path)
    
    # Copy inputs.json
    inputs_json = Path("AI4Life-MDC25-submission/test/input/interface_0/inputs.json")
    shutil.copy2(inputs_json, input_json_path)
    
    # Copy model to expected location
    model_src = Path("submission_model.pth")
    model_dst = Path("/tmp/test_container/opt/app/resources/submission_model.pth")
    model_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_src, model_dst)
    
    print(f"✓ Test environment set up")
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")
    print(f"Model path: {model_dst}")
    
    return input_path, output_path

def test_inference():
    """Test the inference script in simulated environment"""
    
    # Set up environment variables
    old_path = os.getcwd()
    
    try:
        # Set up container environment
        input_path, output_path = setup_container_environment()
        
        # Mock the container paths
        import AI4Life-MDC25-submission.inference as inference_module
        
        # Override the paths temporarily
        original_input = inference_module.INPUT_PATH
        original_output = inference_module.OUTPUT_PATH
        
        inference_module.INPUT_PATH = input_path
        inference_module.OUTPUT_PATH = output_path
        
        # Update model path in load_model function
        original_load_model = inference_module.load_model
        
        def mock_load_model():
            model_path = Path("/tmp/test_container/opt/app/resources/submission_model.pth")
            print(f"Loading model from: {model_path}")
            import torch
            model = torch.jit.load(model_path, map_location='cpu')
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
                print("Model moved to GPU")
            else:
                print("Using CPU inference")
            return model
        
        inference_module.load_model = mock_load_model
        
        # Run inference
        print("Running inference test...")
        result = inference_module.inference_handler()
        
        # Restore original functions
        inference_module.INPUT_PATH = original_input
        inference_module.OUTPUT_PATH = original_output
        inference_module.load_model = original_load_model
        
        # Check results
        output_files = list(output_path.glob("*.tif"))
        print(f"Generated {len(output_files)} output files:")
        for f in output_files:
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
        
        return result == 0 and len(output_files) > 0
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    finally:
        os.chdir(old_path)
        # Clean up
        import shutil
        if Path("/tmp/test_container").exists():
            shutil.rmtree("/tmp/test_container")

if __name__ == "__main__":
    success = test_inference()
    if success:
        print("🎉 Container test passed!")
    else:
        print("❌ Container test failed")