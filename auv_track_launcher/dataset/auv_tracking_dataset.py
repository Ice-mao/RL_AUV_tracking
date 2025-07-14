"""
AUV跟踪任务的数据集实现示例

这个文件展示了如何为AUV跟踪任务创建专门的数据集类
包含了传感器数据（声纳、相机）和控制动作的处理
"""

from typing import Dict, List, Optional
import torch
import numpy as np
import copy
import os
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class AUVTrackingDataset(BaseImageDataset):
    """
    AUV跟踪任务的数据集类
    
    支持多种传感器数据：
    - 相机图像
    - 声纳数据
    - IMU数据
    - GPS位置
    - 目标位置信息
    """
    
    def __init__(self,
            data_path: str,
            horizon: int = 8,           # 预测序列长度
            pad_before: int = 1,        # 历史观测步数
            pad_after: int = 0,         # 未来观测步数
            seed: int = 42,
            val_ratio: float = 0.1,
            max_train_episodes: Optional[int] = None,
            image_size: tuple = (224, 224),
            sonar_range: float = 50.0,  # 声纳最大探测距离
            use_image: bool = True,
            use_sonar: bool = True,
            use_imu: bool = True,
            control_freq: float = 10.0, # 控制频率
            ):
        
        super().__init__()
        
        # 保存配置
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.image_size = image_size
        self.sonar_range = sonar_range
        self.use_image = use_image
        self.use_sonar = use_sonar
        self.use_imu = use_imu
        self.control_freq = control_freq
        
        # 定义数据键名
        self.data_keys = ['action']  # 动作始终需要
        if use_image:
            self.data_keys.append('camera_image')
        if use_sonar:
            self.data_keys.append('sonar_data')
        self.data_keys.extend(['auv_state', 'target_state'])
        
        # 加载数据
        self.replay_buffer = self._load_auv_data(data_path)
        
        # 创建训练/验证划分
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # 创建采样器
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.train_mask = train_mask

    def _load_auv_data(self, data_path: str) -> ReplayBuffer:
        """加载AUV数据"""
        if data_path.endswith('.zarr'):
            return ReplayBuffer.copy_from_path(data_path, keys=self.data_keys)
        elif data_path.endswith('.npz'):
            return self._load_from_npz(data_path)
        else:
            raise ValueError(f"不支持的数据格式: {data_path}")

    def _load_from_npz(self, data_path: str) -> ReplayBuffer:
        """从NPZ文件加载数据"""
        data = np.load(data_path, allow_pickle=True)
        
        replay_buffer = ReplayBuffer.create_empty_numpy()
        
        # 假设NPZ文件包含episodes数组
        episodes = data['episodes']
        
        for episode_idx, episode_data in enumerate(episodes):
            episode = {}
            
            # 处理动作数据 (N, action_dim)
            # 动作：[thrust_x, thrust_y, thrust_z, torque_x, torque_y, torque_z]
            episode['action'] = episode_data['actions'].astype(np.float32)
            
            # 处理AUV状态 (N, state_dim)
            # 状态：[x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
            episode['auv_state'] = episode_data['auv_states'].astype(np.float32)
            
            # 处理目标状态 (N, target_dim)
            # 目标：[target_x, target_y, target_z, relative_distance, relative_bearing]
            episode['target_state'] = episode_data['target_states'].astype(np.float32)
            
            # 处理图像数据
            if self.use_image and 'camera_images' in episode_data:
                # 图像: (N, H, W, C)
                images = episode_data['camera_images']
                episode['camera_image'] = images.astype(np.uint8)
            
            # 处理声纳数据
            if self.use_sonar and 'sonar_data' in episode_data:
                # 声纳: (N, n_beams) 或 (N, H, W) 如果是成像声纳
                episode['sonar_data'] = episode_data['sonar_data'].astype(np.float32)
            
            replay_buffer.add_episode(episode)
        
        return replay_buffer

    def get_validation_dataset(self):
        """创建验证集"""
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        """获取数据标准化器"""
        data = {}
        
        # 动作标准化
        data['action'] = self.replay_buffer['action']
        
        # AUV状态标准化
        auv_states = self.replay_buffer['auv_state']
        data['auv_pos'] = auv_states[..., :3]      # 位置 [x, y, z]
        data['auv_euler'] = auv_states[..., 3:6]   # 姿态 [roll, pitch, yaw]
        data['auv_vel'] = auv_states[..., 6:9]     # 线速度 [vx, vy, vz]
        data['auv_ang_vel'] = auv_states[..., 9:12] # 角速度 [wx, wy, wz]
        
        # 目标状态标准化
        target_states = self.replay_buffer['target_state']
        data['target_pos'] = target_states[..., :3]  # 目标位置
        data['relative_dist'] = target_states[..., 3:4]  # 相对距离
        data['relative_bearing'] = target_states[..., 4:5]  # 相对方位
        
        # 声纳数据标准化
        if self.use_sonar:
            data['sonar'] = self.replay_buffer['sonar_data']
        
        # 创建标准化器
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # 图像标准化器
        if self.use_image:
            normalizer['camera_image'] = get_image_range_normalizer()
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """转换采样数据为模型输入格式"""
        # 基本观测字典
        obs = {}
        
        # 处理图像数据
        if self.use_image and 'camera_image' in sample:
            # 图像预处理：调整大小，通道转换，归一化
            images = sample['camera_image']  # (T, H, W, C)
            
            # 如果需要调整图像大小
            if images.shape[1:3] != self.image_size:
                # 这里应该使用适当的图像调整方法
                # 为简化示例，假设图像已经是正确尺寸
                pass
            
            # 转换通道顺序：HWC -> CHW，归一化到[0,1]
            images = np.moveaxis(images, -1, -2) / 255.0  # (T, C, H, W)
            obs['camera_image'] = images.astype(np.float32)
        
        # 处理声纳数据
        if self.use_sonar and 'sonar_data' in sample:
            sonar = sample['sonar_data'].astype(np.float32)
            # 归一化声纳数据到[0,1]
            obs['sonar'] = sonar / self.sonar_range
        
        # 处理AUV状态
        auv_state = sample['auv_state'].astype(np.float32)
        obs['auv_pos'] = auv_state[..., :3]        # 位置
        obs['auv_euler'] = auv_state[..., 3:6]     # 欧拉角
        obs['auv_vel'] = auv_state[..., 6:9]       # 线速度
        obs['auv_ang_vel'] = auv_state[..., 9:12]  # 角速度
        
        # 处理目标状态
        target_state = sample['target_state'].astype(np.float32)
        obs['target_pos'] = target_state[..., :3]     # 目标位置
        obs['relative_dist'] = target_state[..., 3:4] # 相对距离
        obs['relative_bearing'] = target_state[..., 4:5] # 相对方位角
        
        # 计算额外特征
        obs['relative_pos'] = obs['target_pos'] - obs['auv_pos']  # 相对位置
        
        # 动作数据
        action = sample['action'].astype(np.float32)
        
        return {
            'obs': obs,
            'action': action
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取训练样本"""
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def get_dataset_stats(self) -> Dict:
        """获取数据集统计信息"""
        stats = {
            'n_episodes': self.replay_buffer.n_episodes,
            'total_steps': len(self.replay_buffer),
            'action_dim': self.replay_buffer['action'].shape[-1],
            'auv_state_dim': self.replay_buffer['auv_state'].shape[-1],
            'target_state_dim': self.replay_buffer['target_state'].shape[-1],
            'horizon': self.horizon,
            'pad_before': self.pad_before,
            'pad_after': self.pad_after,
        }
        
        if self.use_image:
            stats['image_shape'] = self.replay_buffer['camera_image'].shape[1:]
        
        if self.use_sonar:
            stats['sonar_shape'] = self.replay_buffer['sonar_data'].shape[1:]
            
        return stats

    def visualize_sample(self, idx: int = 0):
        """可视化数据样本"""
        sample = self[idx]
        
        print(f"=== 数据样本 {idx} ===")
        print(f"观测数据:")
        for key, value in sample['obs'].items():
            print(f"  {key}: {value.shape}")
        
        print(f"动作数据: {sample['action'].shape}")
        
        # 如果有图像，可以保存第一帧
        if 'camera_image' in sample['obs']:
            import matplotlib.pyplot as plt
            image = sample['obs']['camera_image'][0].permute(1, 2, 0).numpy()
            plt.figure(figsize=(8, 6))
            plt.imshow(image)
            plt.title(f"Sample {idx} - First Frame")
            plt.axis('off')
            plt.savefig(f'sample_{idx}_image.png')
            print(f"  图像已保存: sample_{idx}_image.png")


# 使用示例
def create_auv_dataset():
    """创建AUV跟踪数据集的示例"""
    
    # 数据集配置
    config = {
        'data_path': '/path/to/auv_tracking_data.npz',  # 你的数据路径
        'horizon': 8,           # 预测8个时间步
        'pad_before': 2,        # 使用2个历史观测
        'pad_after': 0,         # 不使用未来观测
        'val_ratio': 0.15,      # 15%作为验证集
        'image_size': (224, 224),
        'use_image': True,
        'use_sonar': True,
        'use_imu': True,
    }
    
    try:
        # 创建数据集
        train_dataset = AUVTrackingDataset(**config)
        val_dataset = train_dataset.get_validation_dataset()
        
        # 获取标准化器
        normalizer = train_dataset.get_normalizer()
        
        # 输出统计信息
        stats = train_dataset.get_dataset_stats()
        print("=== AUV数据集统计 ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print(f"\n训练集大小: {len(train_dataset)}")
        print(f"验证集大小: {len(val_dataset)}")
        
        # 测试数据加载
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\n=== 样本数据结构 ===")
            print(f"观测数据:")
            for key, value in sample['obs'].items():
                print(f"  {key}: {value.shape} ({value.dtype})")
            print(f"动作数据: {sample['action'].shape} ({sample['action'].dtype})")
            
            # 可视化第一个样本
            train_dataset.visualize_sample(0)
        
        return train_dataset, val_dataset, normalizer
        
    except Exception as e:
        print(f"数据集创建失败: {e}")
        print("请检查数据路径和格式")
        return None, None, None


if __name__ == "__main__":
    train_dataset, val_dataset, normalizer = create_auv_dataset()
    if train_dataset is not None:
        print("AUV数据集创建成功！")
