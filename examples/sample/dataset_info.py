"""
Info the information of sample dataset
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer

def print_dataset_info(data_path):
    """
    打印数据集详细信息
    """
    print("=" * 60)
    print(f"数据集信息: {data_path}")
    print("=" * 60)
    
    try:
        # 加载数据集
        print("\n正在加载数据集...")
        replay_buffer = ReplayBuffer.copy_from_path(data_path)
        
        # 基本信息
        print("\n【基本信息】")
        print(f"  Episode 数量: {replay_buffer.n_episodes}")
        print(f"  总步数: {replay_buffer.n_steps}")
        print(f"  后端类型: {replay_buffer.backend}")
        
        # Episode 长度统计
        episode_lengths = replay_buffer.episode_lengths
        print(f"\n【Episode 长度统计】")
        print(f"  最小长度: {episode_lengths.min()}")
        print(f"  最大长度: {episode_lengths.max()}")
        print(f"  平均长度: {episode_lengths.mean():.2f}")
        print(f"  中位数长度: {np.median(episode_lengths):.2f}")
        print(f"  标准差: {episode_lengths.std():.2f}")
        
        # 数据键和形状
        print(f"\n【数据键信息】")
        data_keys = list(replay_buffer.keys())
        print(f"  数据键数量: {len(data_keys)}")
        print(f"  数据键列表: {data_keys}")
        
        print(f"\n【各数据键详细信息】")
        for key in data_keys:
            arr = replay_buffer[key]
            print(f"\n  Key: {key}")
            print(f"    形状 (shape): {arr.shape}")
            print(f"    数据类型 (dtype): {arr.dtype}")
            
            # 如果是 zarr 数组，显示更多信息
            if replay_buffer.backend == 'zarr':
                import zarr
                if isinstance(arr, zarr.Array):
                    print(f"    块大小 (chunks): {arr.chunks}")
                    print(f"    压缩器 (compressor): {arr.compressor}")
                    # 计算数据大小（近似）
                    itemsize = np.dtype(arr.dtype).itemsize
                    total_size = np.prod(arr.shape) * itemsize
                    print(f"    总大小 (近似): {total_size / (1024**3):.2f} GB")
            
            # 显示数据范围（仅对数值类型）
            if np.issubdtype(arr.dtype, np.number):
                try:
                    # 只读取一小部分数据来获取统计信息（避免加载整个数组）
                    if arr.shape[0] > 0:
                        sample_size = min(1000, arr.shape[0])
                        if len(arr.shape) == 1:
                            sample = arr[:sample_size]
                        else:
                            # 对于多维数组，只取第一个维度的一部分
                            indices = tuple([slice(0, sample_size)] + [0] * (len(arr.shape) - 1))
                            sample = arr[indices]
                        
                        print(f"    数据范围 (样本): min={sample.min():.4f}, max={sample.max():.4f}, mean={sample.mean():.4f}")
                except Exception as e:
                    print(f"    无法计算数据范围: {e}")
        
        # 显示第一个 episode 的示例数据
        if replay_buffer.n_episodes > 0:
            print(f"\n【第一个 Episode 示例】")
            episode_0 = replay_buffer.get_episode(0, copy=False)
            print(f"  Episode 0 长度: {len(next(iter(episode_0.values())))}")
            print(f"  Episode 0 包含的键: {list(episode_0.keys())}")
            for key, value in episode_0.items():
                print(f"    {key}: shape={value.shape}, dtype={value.dtype}")
        
        print("\n" + "=" * 60)
        print("✅ 数据集信息查看完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 加载数据集时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 查看 v1_episodes 数据集
    data_path = 'log/sample/v1_episodes/auv_data_final.zarr'
    print_dataset_info(data_path)
