#!/usr/bin/env python3
"""
Create stacked TIFF images for Grand Challenge submission format.
Bundles individual TIFF files into multi-page TIFF stacks.
"""

import tifffile
import numpy as np
from pathlib import Path
import uuid

def create_image_stacks():
    """Create stacked TIFF images from individual files"""
    
    # Source directories
    small_noisy = Path("denoising/data/small_images/noisy")
    small_clean = Path("denoising/data/small_images/gt")
    large_noisy = Path("denoising/data/large_images/noisy") 
    large_clean = Path("denoising/data/large_images/gt")
    
    # Target directories following Grand Challenge structure
    input_dir = Path("AI4Life-MDC25-submission/test/input/interface_0/images/image-stack-unstructured-noise")
    output_dir = Path("AI4Life-MDC25-submission/test/output/interface_0/images/image-stack-denoised")
    
    # Create target directories
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating stacked image files...")
    
    # Create small image stack (first 50 images for testing)
    small_files = sorted(list(small_noisy.glob("*.tif")))[:50]
    if small_files:
        print(f"Creating small image stack from {len(small_files)} images...")
        
        # Load all small images
        small_stack = []
        for img_file in small_files:
            img = tifffile.imread(img_file)
            small_stack.append(img)
        
        # Stack into 3D array (Z, Y, X)
        small_stack = np.stack(small_stack, axis=0)
        print(f"Small stack shape: {small_stack.shape}")
        
        # Save as multi-page TIFF with UUID filename
        stack_uuid = str(uuid.uuid4())
        stack_path = input_dir / f"{stack_uuid}.tif"
        tifffile.imwrite(stack_path, small_stack)
        print(f"Saved small stack: {stack_path}")
        
        # Create corresponding clean stack for reference
        clean_files = [small_clean / f.name for f in small_files]
        clean_stack = []
        for img_file in clean_files:
            if img_file.exists():
                img = tifffile.imread(img_file)
                clean_stack.append(img)
        
        if clean_stack:
            clean_stack = np.stack(clean_stack, axis=0)
            clean_path = output_dir / f"{stack_uuid}.tif"
            tifffile.imwrite(clean_path, clean_stack)
            print(f"Saved clean reference: {clean_path}")
    
    # Create large image stack (all 20 images)
    large_files = sorted(list(large_noisy.glob("*.tif")))
    if large_files:
        print(f"Creating large image stack from {len(large_files)} images...")
        
        # Load all large images
        large_stack = []
        for img_file in large_files:
            img = tifffile.imread(img_file)
            large_stack.append(img)
        
        # Stack into 3D array
        large_stack = np.stack(large_stack, axis=0)
        print(f"Large stack shape: {large_stack.shape}")
        
        # Save as multi-page TIFF
        stack_uuid = str(uuid.uuid4())
        stack_path = input_dir / f"{stack_uuid}.tif"
        tifffile.imwrite(stack_path, large_stack)
        print(f"Saved large stack: {stack_path}")
        
        # Create corresponding clean stack
        clean_files = [large_clean / f.name for f in large_files]
        clean_stack = []
        for img_file in clean_files:
            if img_file.exists():
                img = tifffile.imread(img_file)
                clean_stack.append(img)
        
        if clean_stack:
            clean_stack = np.stack(clean_stack, axis=0)
            clean_path = output_dir / f"{stack_uuid}.tif"
            tifffile.imwrite(clean_path, clean_stack)
            print(f"Saved large clean reference: {clean_path}")
    
    print("\n✅ Image stacks created successfully!")
    print(f"Input stacks: {len(list(input_dir.glob('*.tif')))} files")
    print(f"Reference stacks: {len(list(output_dir.glob('*.tif')))} files")
    
    return True

if __name__ == "__main__":
    create_image_stacks()