"""
从AUV数据集中提取episode的camera_image并保存为PNG格式
用于后续的视频制作
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
    从数据集中提取指定episode的相机图像并保存为PNG
    
    Args:
        dataset_path (str): 数据集路径 (.zarr文件)
        output_dir (str): 输出目录
        episode_idx (int): episode索引 (默认0，即第一个episode)
        key (str): 图像数据的键名 (默认'camera_image')
    """
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"正在从 {dataset_path} 提取第 {episode_idx} 个episode的图像...")
    print(f"输出目录: {output_dir}")
    
    try:
        # 直接读取zarr文件
        # zarr_data = zarr.open(dataset_path, mode='r')
        zarr_data = ReplayBuffer.copy_from_path(dataset_path, keys=[key])
        print(f"数据集结构:")
        print(f"  Keys: {list(zarr_data.keys())}")
        
        if key not in zarr_data:
            print(f"错误: 数据集中未找到键 '{key}'")
            print(f"可用的键: {list(zarr_data.keys())}")
            return False
            
        # 获取图像数据
        images = zarr_data.get_episode(episode_idx)[key]
        print(f"图像数据形状: {images.shape}")
        print(f"图像数据类型: {images.dtype}")
        
        # 选择每一帧的第0列图像 (形状从 (717, 5, 3, 224, 224) 变为 (717, 3, 224, 224))
        episode_images = images[:, 0, :, :, :]
        print(f"选择第0列后的图像形状: {episode_images.shape}")
        
        # 保存每一帧
        saved_count = 0
        for step_idx, img in enumerate(episode_images):
            
            # 处理图像数据
            # img形状为 (3, 224, 224)，需要转换为 (224, 224, 3)
            if len(img.shape) == 3 and img.shape[0] == 3:
                # 从 (C, H, W) 转换为 (H, W, C)
                img = np.transpose(img, (1, 2, 0))
            
            if img.dtype == np.float32 or img.dtype == np.float64:
                # 如果是浮点数，假设范围在[0,1]，转换为[0,255]
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)
            
            # 确保图像格式正确 (H, W, C)
            if len(img.shape) == 3 and img.shape[2] == 3:
                # RGB图像，转换为BGR用于OpenCV
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                # 单通道图像
                img_bgr = img.squeeze()
            else:
                img_bgr = img
            
            # 生成文件名 (保持数字排序)
            filename = f"frame_{step_idx:04d}.png"
            filepath = output_path / filename
            
            # 保存图像
            success = cv2.imwrite(str(filepath), img_bgr)
            if success:
                saved_count += 1
                if step_idx % 50 == 0:  # 每50帧打印一次进度
                    print(f"  已保存 {step_idx + 1}/{episode_images.shape[0]} 帧")
            else:
                print(f"  警告: 无法保存帧 {step_idx}")
        
        print(f"✅ 成功保存 {saved_count}/{episode_images.shape[0]} 帧图像到 {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 提取图像时出错: {e}")
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
    
    if success:
        print(f"\n🎬 现在你可以运行以下命令制作视频:")
        print(f"python examples/sample/make_video.py")

if __name__ == "__main__":
    main()
