"""
From a dataset, extract images from a specified episode and save them as PNG files.
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import cv2
from PIL import Image
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer
from auv_track_launcher.dataset.auv_tracking_dataset import AUVTrackingDataset
import argparse
from pathlib import Path

def extract_episode_images(dataset_path, output_dir, episode_idx=0, key='camera_image'):
    """
    Extract camera images from a specified episode in the dataset and save them as PNG files.

    Args:
        dataset_path (str): Path to the dataset (.zarr file)
        output_dir (str): Output directory
        episode_idx (int): Episode index (default 0, i.e., the first episode)
        key (str): Key name for the image data (default 'camera_image')
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Extracting images from episode {episode_idx} in {dataset_path}...")
    print(f"Output directory: {output_dir}")

    try:
        # Directly read zarr file
        # zarr_data = zarr.open(dataset_path, mode='r')
        zarr_data = ReplayBuffer.copy_from_path(dataset_path, keys=[key])
        print(f"Dataset structure:")
        print(f"  Keys: {list(zarr_data.keys())}")
        
        if key not in zarr_data:
            print(f"Error: Key '{key}' not found in dataset")
            print(f"Available keys: {list(zarr_data.keys())}")
            return False

        # Get image data
        images = zarr_data.get_episode(episode_idx)[key]
        print(f"Image data shape: {images.shape}")
        print(f"Image data type: {images.dtype}")

        # Select the first column of each frame (shape changes from (717, 5, 3, 224, 224) to (717, 3, 224, 224))
        episode_images = images[:, 0, :, :, :]
        print(f"Shape after selecting first column: {episode_images.shape}")

        # Save each frame
        saved_count = 0
        for step_idx, img in enumerate(episode_images):

            # Process image data
            # img shape is (3, 224, 224), need to convert to (224, 224, 3)
            if len(img.shape) == 3 and img.shape[0] == 3:
                # Convert from (C, H, W) to (H, W, C)
                img = np.transpose(img, (1, 2, 0))
            
            if img.dtype == np.float32 or img.dtype == np.float64:
                # If float, assume range is [0,1], convert to [0,255]
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)

            # Ensure image format is correct (H, W, C)
            if len(img.shape) == 3 and img.shape[2] == 3:
                # RGB image, convert to BGR for OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                # Single-channel image
                img_bgr = img.squeeze()
            else:
                img_bgr = img

            # Generate filename (keep numeric sorting)
            filename = f"frame_{step_idx:04d}.png"
            filepath = output_path / filename

            # Save image
            success = cv2.imwrite(str(filepath), img_bgr)
            if success:
                saved_count += 1
                if step_idx % 50 == 0:  # Print progress every 50 frames
                    print(f"  Saved {step_idx + 1}/{episode_images.shape[0]} frames")
            else:
                print(f"  Warning: Failed to save frame {step_idx}")

        print(f"✅ Successfully saved {saved_count}/{episode_images.shape[0]} frames to {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error occurred while extracting images: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():  
    success = extract_episode_images(
        dataset_path="log/sample/simple/auv_data_partial_20.zarr",
        output_dir="log/sample/test",
        episode_idx=10,
        key='camera_image'
    )

if __name__ == "__main__":
    main()
