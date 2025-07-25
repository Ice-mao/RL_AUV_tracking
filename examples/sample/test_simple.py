"""
测试简单数据采集和数据集加载
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

def test_simple_collection():

    print("\n2. 测试数据集加载...")
    from auv_track_launcher.dataset.auv_tracking_dataset import AUVTrackingDataset
    
    dataset = AUVTrackingDataset(
        data_path='/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/auv_data/auv_data_final.zarr',
        key=['action', 'camera_image', 'state'],
        horizon=4,
        pad_before=1,
        val_ratio=0.2
    )
    
    print(f"✅ 数据集加载成功!")
    print(f"数据集大小: {len(dataset)}")
    
    # 测试获取一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n样本数据结构:")
        print(f"观测数据:")
        for key, value in sample['obs'].items():
            print(f"  {key}: {value.shape}")
        print(f"动作数据: {sample['action'].shape}")
    


if __name__ == '__main__':
    test_simple_collection()
