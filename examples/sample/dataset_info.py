"""
Info the information of sample dataset
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

def test_simple_collection():

    print("\nTesting dataset loading...")
    from auv_track_launcher.dataset.auv_tracking_dataset import AUVTrackingDataset
    
    dataset = AUVTrackingDataset(
        data_path='log/sample/simple/auv_data_partial_20.zarr',
        key=['action', 'camera_image', 'state'],
        horizon=4,
        pad_before=1,
        val_ratio=0.2
    )

    print(f"✅ Dataset loading successful!")
    print(f"Dataset size: {len(dataset)}")

    # Test getting a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample data structure:")
        print(f"Observation data:")
        for key, value in sample['obs'].items():
            print(f"  {key}: {value.shape}")
        print(f"Action data: {sample['action'].shape}")
    


if __name__ == '__main__':
    test_simple_collection()
