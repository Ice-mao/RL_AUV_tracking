"""
AUV跟踪任务的数据集实现

这个文件展示了如何为AUV跟踪任务创建专门的数据集类
包含了传感器数据（声纳、相机）和控制动作的处理
使用zarr格式存储数据
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
        key: ['action', 'camera_image', 'state', 'sonar_image']
    """
    def __init__(self,
            data_path: str,
            key: list,
            horizon: int = 8,
            pad_before: int = 1,
            pad_after: int = 0,
            seed: int = 42,
            val_ratio: float = 0.1,
            max_train_episodes: Optional[int] = None,
            image_size: tuple = (224, 224),
            sonar_range: float = 50.0,  # 声纳最大探测距离
            ):
        
        super().__init__()
        
        # 保存配置
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.image_size = image_size
        self.sonar_range = sonar_range
        
        self.data_keys = key
        
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
        """加载AUV数据（zarr格式）"""
        if not data_path.endswith('.zarr'):
            raise ValueError(f"只支持zarr格式的数据文件，当前文件: {data_path}")
        
        return ReplayBuffer.copy_from_path(data_path, keys=self.data_keys)

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
        normalizer['camera_image'] = get_image_range_normalizer()
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """转换采样数据为模型输入格式"""
        # 基本观测字典
        obs = {}
        
        # 处理图像数据
        if 'camera_image' in self.data_keys:
            images = sample['camera_image']  # (T, H, W, C)
            obs['camera_image'] = images.astype(np.float32)
        
        # 处理声纳数据
        if 'sonar_image' in self.data_keys:
            sonar = sample['sonar_data'].astype(np.float32)
            # 归一化声纳数据到[0,1]
            obs['sonar'] = sonar / self.sonar_range
        
        # 处理状态
        if 'state' in self.data_keys:
            obs['state'] = sample['state']
        
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
