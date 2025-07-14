from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class CustomDataset(BaseImageDataset):
    """
    自定义数据集类模板
    
    这个类展示了如何创建自己的数据集，用于扩散策略训练
    你需要根据自己的数据格式修改相应的方法
    """
    
    def __init__(self,
            data_path,              # 数据文件路径（可以是zarr、hdf5等格式）
            horizon=1,              # 序列长度：预测多少个时间步
            pad_before=0,           # 序列前填充：历史观测步数
            pad_after=0,            # 序列后填充：未来观测步数
            seed=42,                # 随机种子
            val_ratio=0.0,          # 验证集比例 (0.0-1.0)
            max_train_episodes=None, # 最大训练episode数量
            # 以下是你可能需要添加的自定义参数
            obs_keys=None,          # 观测数据的键名列表
            action_dim=None,        # 动作维度
            image_size=None,        # 图像尺寸
            ):
        
        super().__init__()
        
        # 保存配置参数
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.obs_keys = obs_keys or ['image', 'state']  # 默认观测键
        self.action_dim = action_dim
        self.image_size = image_size or (96, 96)
        
        # 加载数据 - 根据你的数据格式修改这部分
        self.replay_buffer = self._load_data(data_path)
        
        # 创建训练/验证集划分
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        
        # 下采样训练集（如果指定了最大episode数）
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # 创建序列采样器
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.train_mask = train_mask

    def _load_data(self, data_path):
        """
        加载数据的方法 - 根据你的数据格式实现
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            ReplayBuffer: 包含数据的回放缓冲区
        """
        # 方法1: 如果你的数据是zarr格式
        if data_path.endswith('.zarr'):
            # 根据你的数据键名修改这里
            keys = ['image', 'state', 'action']  # 或者你的数据键名
            return ReplayBuffer.copy_from_path(data_path, keys=keys)
        
        # 方法2: 如果你的数据是其他格式（numpy、pickle等）
        # 你需要实现自己的加载逻辑
        else:
            # 示例：加载numpy数据
            # data = np.load(data_path, allow_pickle=True)
            # return self._create_replay_buffer_from_data(data)
            raise NotImplementedError("请实现你自己的数据加载逻辑")

    def _create_replay_buffer_from_data(self, data):
        """
        从原始数据创建ReplayBuffer的辅助方法
        
        Args:
            data: 原始数据
            
        Returns:
            ReplayBuffer: 创建的回放缓冲区
        """
        # 这里是一个示例实现，你需要根据你的数据格式修改
        replay_buffer = ReplayBuffer.create_empty_numpy()
        
        # 假设data是一个字典，包含episodes列表
        for episode_data in data['episodes']:
            episode = {
                'image': episode_data['observations']['image'],
                'state': episode_data['observations']['state'], 
                'action': episode_data['actions']
            }
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
        """
        获取数据标准化器
        
        Args:
            mode: 标准化模式 ('limits', 'gaussian', etc.)
            
        Returns:
            LinearNormalizer: 配置好的标准化器
        """
        # 准备需要标准化的数据
        data = {
            'action': self.replay_buffer['action'],
        }
        
        # 添加状态数据（根据你的状态表示修改）
        if 'state' in self.replay_buffer.keys():
            # 示例：如果状态包含位置、速度等信息
            state_data = self.replay_buffer['state']
            data['agent_pos'] = state_data[...,:2]  # 假设前2维是位置
            # data['agent_vel'] = state_data[...,2:4]  # 假设接下来2维是速度
        
        # 创建并拟合标准化器
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # 为图像添加标准化器（如果有图像数据）
        if 'image' in self.replay_buffer.keys():
            normalizer['image'] = get_image_range_normalizer()
        
        return normalizer

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """
        将采样的原始数据转换为模型需要的格式
        
        Args:
            sample: 从replay_buffer采样的原始数据
            
        Returns:
            dict: 格式化后的数据
        """
        data = {
            'obs': {},
            'action': sample['action'].astype(np.float32)
        }
        
        # 处理图像数据（如果存在）
        if 'image' in sample:
            # 转换图像格式：HWC -> CHW，并归一化到[0,1]
            image = np.moveaxis(sample['image'], -1, 1) / 255.0
            data['obs']['image'] = image.astype(np.float32)
        
        # 处理状态数据（根据你的状态表示修改）
        if 'state' in sample:
            state = sample['state'].astype(np.float32)
            
            # 示例：分解状态为不同组件
            data['obs']['agent_pos'] = state[..., :2]  # 位置
            # data['obs']['agent_vel'] = state[..., 2:4]  # 速度
            # data['obs']['other_info'] = state[..., 4:]  # 其他信息
        
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        PyTorch Dataset接口：获取一个数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            Dict[str, torch.Tensor]: 包含观测和动作的字典
        """
        # 采样序列数据
        sample = self.sampler.sample_sequence(idx)
        
        # 转换数据格式
        data = self._sample_to_data(sample)
        
        # 转换为torch张量
        torch_data = dict_apply(data, torch.from_numpy)
        
        return torch_data

    # 可选：添加数据集统计信息方法
    def get_dataset_stats(self):
        """获取数据集统计信息"""
        stats = {
            'n_episodes': self.replay_buffer.n_episodes,
            'total_steps': len(self.replay_buffer),
            'action_dim': self.replay_buffer['action'].shape[-1],
        }
        
        if 'image' in self.replay_buffer.keys():
            stats['image_shape'] = self.replay_buffer['image'].shape[1:]
        
        if 'state' in self.replay_buffer.keys():
            stats['state_dim'] = self.replay_buffer['state'].shape[-1]
            
        return stats


# 使用示例
def create_custom_dataset_example():
    """
    创建自定义数据集的使用示例
    """
    # 基本配置
    dataset_config = {
        'data_path': '/path/to/your/data.zarr',  # 你的数据路径
        'horizon': 16,          # 预测16个时间步
        'pad_before': 1,        # 使用1个历史观测
        'pad_after': 0,         # 不使用未来观测
        'val_ratio': 0.1,       # 10%作为验证集
        'max_train_episodes': 1000,  # 最多使用1000个训练episode
    }
    
    # 创建数据集
    train_dataset = CustomDataset(**dataset_config)
    val_dataset = train_dataset.get_validation_dataset()
    
    # 获取标准化器
    normalizer = train_dataset.get_normalizer()
    
    # 查看数据集信息
    print("数据集统计:", train_dataset.get_dataset_stats())
    print("训练集大小:", len(train_dataset))
    print("验证集大小:", len(val_dataset))
    
    # 测试数据加载
    sample = train_dataset[0]
    print("样本数据结构:")
    for key, value in sample.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for subkey, subvalue in value.items():
                print(f"    {subkey}: {subvalue.shape}")
        else:
            print(f"  {key}: {value.shape}")
    
    return train_dataset, val_dataset, normalizer


if __name__ == "__main__":
    # 运行示例
    try:
        train_dataset, val_dataset, normalizer = create_custom_dataset_example()
        print("数据集创建成功！")
    except Exception as e:
        print(f"示例运行失败: {e}")
        print("请根据你的实际数据格式修改相应代码")
