"""
测试简单数据采集和数据集加载
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

def test_simple_collection():
    """测试简单数据采集"""
    
    print("=== 测试简单数据采集 ===")
    
    # 1. 采集数据
    print("1. 开始采集数据...")
    from simple_collect import collect_auv_data
    collect_auv_data(n_episodes=5)  # 只采集5个episodes测试
    
    # 2. 加载数据集
    print("\n2. 测试数据集加载...")
    try:
        from auv_track_launcher.dataset.auv_tracking_dataset import AUVTrackingDataset
        
        dataset = AUVTrackingDataset(
            data_path='./simple_auv_data/auv_data_final.zarr',
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
        
        print(f"\n✅ 测试成功! 数据采集和加载都正常工作")
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return False
    
    return True


if __name__ == '__main__':
    success = test_simple_collection()
    
    if success:
        print(f"""
🎉 简单数据采集测试成功!

使用方法:
1. 采集更多数据: python simple_collect.py
2. 创建数据集: 
   dataset = AUVTrackingDataset('./simple_auv_data/auv_data_final.zarr')
3. 开始训练!
        """)
    else:
        print("❌ 测试失败，请检查环境配置")
