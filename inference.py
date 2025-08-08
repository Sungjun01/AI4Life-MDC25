"""
AI4Life MDC25 Denoising Challenge Submission

This submission uses an advanced CARE U-Net model (AdvancedCAREUNet) 
trained for microscopy image denoising. The model achieves 27.2 dB PSNR
and 0.776 SSIM on validation data.

Model: Advanced CARE U-Net with attention mechanisms
Training: 31.2M parameters, trained on noise2clean pairs

The following is the inference pipeline for the AI4Life MDC25 challenge.

It is meant to run within a container.

To run the container locally, you can call the following bash script:

  ./do_test_run.sh

This will start the inference and reads from ./test/input and writes to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./do_save.sh

Any container that shows the same behaviour will do, this is purely an example of how one COULD do it.

Reference the documentation to get details on the runtime environment on the platform:
https://grand-challenge.org/documentation/runtime-environment/

Happy programming!
"""

from pathlib import Path
import json
import tifffile 
import numpy as np
import torch
import SimpleITK as sitk


# Constants for the location of the input and output files, please do not modify! 
INPUT_PATH = Path("/input/images/image-stack-unstructured-noise")
OUTPUT_PATH = Path("/output/images/image-stack-denoised")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================================
# USER-CUSTOMIZABLE FUNCTIONS
# Modify these functions to implement your own inference pipeline
# ============================================================================

def load_model():
    """
    Load the advanced_care_unet denoising model.
    Model is saved as TorchScript in the resources directory.
    """
    # Advanced CARE U-Net model saved in resources/
    model_path = Path("/opt/app/resources/submission_model.pth")
    
    print(f"Loading advanced_care_unet model from: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to GPU")
    else:
        print("Using CPU inference")
    
    return model


def read_image(image_path: Path) -> np.ndarray:
    """
    Read and preprocess input image stack for microscopy denoising.
    Supports both .tif and .mha file formats.
    Handles multi-page TIFF stacks as required by Grand Challenge.
    Normalizes image to [0,1] range for optimal model performance.
    """
    print(f"Reading image stack: {image_path}")
    
    # Handle different file formats
    if image_path.suffix.lower() == '.mha':
        # Use SimpleITK for .mha files
        sitk_image = sitk.ReadImage(str(image_path))
        input_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)
    else:
        # Use tifffile for .tif files (handles multi-page TIFF stacks)
        input_array = tifffile.imread(image_path).astype(np.float32)
    
    # Normalize to [0,1] range for consistent model input
    if input_array.max() > 1.0:
        input_array = input_array / input_array.max()
    
    print(f"Loaded image stack shape: {input_array.shape}, range: [{input_array.min():.3f}, {input_array.max():.3f}]")
    
    return input_array


def run_inference(model, input_array):
    """
    Run inference on the input array using advanced CARE U-Net.
    Handles proper tensor formatting and GPU/CPU inference.
    """
    print("Running denoising inference...")
    print(f"Input shape: {input_array.shape}")
    
    # Convert to tensor and add batch dimension if needed
    if input_array.ndim == 2:
        # Add channel and batch dimensions (H, W) -> (1, 1, H, W)
        input_tensor = torch.from_numpy(input_array).unsqueeze(0).unsqueeze(0)
    elif input_array.ndim == 3:
        # Add batch dimension (C, H, W) -> (1, C, H, W)
        input_tensor = torch.from_numpy(input_array).unsqueeze(0)
    else:
        input_tensor = torch.from_numpy(input_array)
    
    # Move to GPU if model is on GPU
    if next(model.parameters()).is_cuda:
        input_tensor = input_tensor.cuda()
    
    print(f"Tensor shape for inference: {input_tensor.shape}")
    
    with torch.no_grad():
        output = model(input_tensor)
        
        # Remove batch dimension and convert back to numpy
        if output.dim() == 4:
            output = output.squeeze(0)  # Remove batch dim
        if output.dim() == 3 and output.shape[0] == 1:
            output = output.squeeze(0)  # Remove channel dim if single channel
            
        output = output.cpu().numpy()
    
    print(f"Output shape: {output.shape}, range: [{output.min():.3f}, {output.max():.3f}]")
    return output


def save_output(array, output_path):
    """
    Save the processed array or image stack.
    Supports both .tif and .mha output formats.
    Handles multi-page TIFF stacks for Grand Challenge format.
    """
    print(f"Saving output stack to: {output_path} (shape: {array.shape})")
    
    if output_path.suffix.lower() == '.mha':
        # Use SimpleITK for .mha files
        sitk_image = sitk.GetImageFromArray(array)
        sitk.WriteImage(sitk_image, str(output_path))
    else:
        # Use tifffile for .tif files
        with tifffile.TiffWriter(output_path) as out:
            out.write(
                array,
                resolutionunit=2 # This flag is important for the GC to process the output correctly! 
            )


def inference_handler():
    """
    Main handler for processing images with unstructured noise.
    This is where you should implement your main inference pipeline.
    """
    # Show torch cuda info
    _show_torch_cuda_info()

    # Load your model
    model = load_model()

    # Load and prepare input (support both .tif and .mha files)
    input_files = sorted(list(INPUT_PATH.glob("*.tif")) + list(INPUT_PATH.glob("*.mha")))
    print(f"Reading input files: {input_files}")

    for input_file in input_files:
        input_stack = read_image(input_file)

        # Process image stack - handle 3D stacks (Z, Y, X)
        if input_stack.ndim == 3:
            print(f"Processing {input_stack.shape[0]} images in stack...")
            processed_stack = []
            
            for i, single_image in enumerate(input_stack):
                print(f"Processing image {i+1}/{input_stack.shape[0]}")
                
                # Run inference on single image
                result = run_inference(model, single_image)
                processed_stack.append(result)
            
            # Stack results back together
            result_stack = np.stack(processed_stack, axis=0)
            print(f"Final output stack shape: {result_stack.shape}")
            
        else:
            # Single 2D image
            result_stack = run_inference(model, input_stack)

        # Save output stack
        output_path = OUTPUT_PATH / input_file.name
        save_output(result_stack, output_path)

    return 0


# ============================================================================
# UTILITY FUNCTIONS
# These functions handle the interface with the Grand Challenge platform
# ============================================================================

def run():
    # The key is a tuple of the slugs of the input sockets
    interface_key = get_interface_key()

    print(f"Interface key: {interface_key}")

    handler = inference_handler

    # Call the handler
    return handler()


def get_interface_key():
    # The inputs.json is a system generated file that contains information about
    # the inputs that interface with the algorithm
    inputs = load_json_file(INPUT_PATH.parent.parent / "inputs.json")
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(location):
    # Reads a json file
    with open(location, "r") as f:
        return json.loads(f.read())


def _show_torch_cuda_info():
    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
