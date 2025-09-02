"""
从AUV数据集中过滤掉长度小于指定值的episode
并保存为新的数据集
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
from pathlib import Path
import argparse

def filter_episodes_by_length(input_path, output_path, min_length, keys=None):
    """
    过滤数据集中episode长度小于指定值的episode
    
    Args:
        input_path (str): 输入数据集路径 (.zarr文件)
        output_path (str): 输出数据集路径 (.zarr文件)
        min_length (int): 最小episode长度阈值
        keys (list): 要保留的数据键名列表，None表示保留所有键
    """
    
    print(f"正在从 {input_path} 加载数据集...")
    
    try:
        # 加载原始数据集
        if keys is None:
            input_buffer = ReplayBuffer.copy_from_path(input_path)
        else:
            input_buffer = ReplayBuffer.copy_from_path(input_path, keys=keys)
        
        print(f"原始数据集信息:")
        print(f"  总episodes: {input_buffer.n_episodes}")
        print(f"  总步数: {input_buffer.n_steps}")
        print(f"  数据键: {list(input_buffer.keys())}")
        
        # 获取所有episode长度
        episode_lengths = input_buffer.episode_lengths
        print(f"  Episode长度: min={episode_lengths.min()}, max={episode_lengths.max()}, mean={episode_lengths.mean():.1f}")
        
        # 找到满足长度要求的episode
        valid_episodes = []
        for i, length in enumerate(episode_lengths):
            if length >= min_length:
                valid_episodes.append(i)
        
        print(f"\n过滤条件: episode长度 >= {min_length}")
        print(f"满足条件的episodes: {len(valid_episodes)}/{len(episode_lengths)}")
        
        if len(valid_episodes) == 0:
            print("❌ 没有episode满足长度要求!")
            return False
        
        # 创建新的数据集
        print(f"正在创建新数据集...")
        output_buffer = ReplayBuffer.create_empty_numpy()
        
        # 复制满足条件的episodes
        total_steps = 0
        for i, episode_idx in enumerate(valid_episodes):
            episode_data = input_buffer.get_episode(episode_idx, copy=True)
            episode_length = len(next(iter(episode_data.values())))
            
            output_buffer.add_episode(episode_data)
            total_steps += episode_length
            
            if (i + 1) % 10 == 0 or (i + 1) == len(valid_episodes):
                print(f"  已处理 {i + 1}/{len(valid_episodes)} episodes")
        
        print(f"\n新数据集信息:")
        print(f"  总episodes: {output_buffer.n_episodes}")
        print(f"  总步数: {output_buffer.n_steps}")
        print(f"  平均episode长度: {output_buffer.n_steps / output_buffer.n_episodes:.1f}")
        
        # 保存新数据集
        print(f"正在保存到 {output_path}...")
        
        # 确保输出目录存在
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存为zarr格式
        output_buffer.save_to_path(
            output_path,
            chunks={'camera_image': (1, 3, 224, 224), 'state': (1, -1)},  # 可根据需要调整
            compressors='default'
        )
        
        print(f"✅ 成功保存过滤后的数据集到 {output_path}")
        print(f"过滤前: {len(episode_lengths)} episodes, {input_buffer.n_steps} steps")
        print(f"过滤后: {output_buffer.n_episodes} episodes, {output_buffer.n_steps} steps")
        print(f"保留比例: episodes {output_buffer.n_episodes/len(episode_lengths)*100:.1f}%, steps {output_buffer.n_steps/input_buffer.n_steps*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ 处理数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='过滤数据集中长度不足的episode')
    parser.add_argument('--input', type=str, required=True,
                       help='输入数据集路径 (.zarr)')
    parser.add_argument('--output', type=str, required=True,
                       help='输出数据集路径 (.zarr)')
    parser.add_argument('--min_length', type=int, default=500,
                       help='最小episode长度阈值 (默认: 500)')
    parser.add_argument('--keys', nargs='+', default=None,
                       help='要保留的数据键名列表 (默认: 保留所有键)')
    
    args = parser.parse_args()
    
    print("=== Episode长度过滤工具 ===")
    print(f"输入数据集: {args.input}")
    print(f"输出数据集: {args.output}")
    print(f"最小长度阈值: {args.min_length}")
    print(f"保留的键: {args.keys if args.keys else '所有键'}")
    print()
    
    success = filter_episodes_by_length(
        input_path=args.input,
        output_path=args.output,
        min_length=args.min_length,
        keys=args.keys
    )
    
    if success:
        print(f"\n🎉 数据集过滤完成!")
    else:
        print(f"\n💥 数据集过滤失败!")

if __name__ == "__main__":
    # 示例用法（如果没有命令行参数）
    if len(sys.argv) == 1:
        print("示例用法:")
        print("python examples/sample/filter_episodes_by_length.py --input log/sample/simple/auv_data_partial_20.zarr --output log/sample/simple/auv_data_filtered.zarr --min_length 500")
        print()
        print("使用默认参数进行测试...")
        
        success = filter_episodes_by_length(
            input_path="log/sample/simple/auv_data_partial_20.zarr",
            output_path="log/sample/simple/auv_data_filtered.zarr", 
            min_length=500,
            keys=None
        )
        
        if success:
            print(f"\n🎉 默认测试完成!")
    else:
        main()
