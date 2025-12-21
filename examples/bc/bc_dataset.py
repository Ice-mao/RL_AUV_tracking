"""
BC Dataset - 单步图像-动作对数据集
用于行为克隆训练
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import random
from diffusion_policy.common.replay_buffer import ReplayBuffer


class BCDataset(Dataset):
    """
    BC 数据集
    从 zarr 格式的 replay buffer 加载数据
    按 episode 划分训练/验证集（避免数据泄漏）
    返回单步的 (image, action) 对
    支持数据增强（镜像翻转）来平衡左右偏差
    """

    def __init__(self, data_path, val_ratio=0.1, is_val=False, seed=42,
                 use_augmentation=False, aug_prob=0.5):
        """
        Args:
            data_path: zarr 数据集路径
            val_ratio: 验证集比例（按 episode 数量）
            is_val: 是否返回验证集
            seed: 随机种子
            use_augmentation: 是否使用数据增强（镜像翻转）
            aug_prob: 数据增强概率
        """
        self.use_augmentation = use_augmentation
        self.aug_prob = aug_prob
        self.keys = ['camera_image', 'action']
        self.replay_buffer = ReplayBuffer.copy_from_path(data_path, keys=self.keys)

        # 获取 episode 信息
        n_episodes = self.replay_buffer.n_episodes
        episode_ends = self.replay_buffer.episode_ends[:]  # 每个 episode 的结束索引

        # 按 episode 划分训练/验证集
        rng = np.random.default_rng(seed)
        episode_indices = np.arange(n_episodes)
        rng.shuffle(episode_indices)

        n_val_episodes = max(1, int(n_episodes * val_ratio))
        if is_val:
            selected_episodes = episode_indices[:n_val_episodes]
        else:
            selected_episodes = episode_indices[n_val_episodes:]

        # 根据选中的 episode 获取对应的步索引
        self.indices = []
        for ep_idx in selected_episodes:
            start_idx = 0 if ep_idx == 0 else episode_ends[ep_idx - 1]
            end_idx = episode_ends[ep_idx]
            self.indices.extend(range(start_idx, end_idx))

        self.indices = np.array(self.indices)
        self.n_steps = self.replay_buffer.n_steps

        print(f"BCDataset: 总 episodes={n_episodes}, 总步数={self.n_steps}")
        print(f"  {'验证' if is_val else '训练'}集: {len(selected_episodes)} episodes, "
              f"{len(self.indices)} 步")
        if use_augmentation and not is_val:
            print(f"  数据增强已启用: 镜像翻转概率={aug_prob}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        """
        返回单步数据
        Returns:
            dict: {
                'obs': (3, H, W) float32 图像，归一化到 [0, 1]
                'action': (action_dim,) float32 动作
            }
        """
        real_idx = self.indices[idx]

        # 获取图像和动作
        image = self.replay_buffer['camera_image'][real_idx]  # (3, H, W) uint8
        action = self.replay_buffer['action'][real_idx]  # (action_dim,) float32

        # 图像归一化到 [0, 1]
        image = image.astype(np.float32) / 255.0

        # 数据增强：镜像翻转（仅训练集）
        if self.use_augmentation and random.random() < self.aug_prob:
            # 左右翻转图像 (CHW format，翻转W维度)
            image = np.flip(image, axis=2).copy()
            # 翻转转向动作 (action[1] 是 yaw 转向角)
            action = action.copy()
            action[1] = -action[1]

        return {
            'obs': torch.from_numpy(image),
            'action': torch.from_numpy(action.astype(np.float32))
        }

    def get_action_stats(self):
        """获取动作的统计信息，用于归一化"""
        actions = self.replay_buffer['action'][:]
        return {
            'mean': actions.mean(axis=0),
            'std': actions.std(axis=0),
            'min': actions.min(axis=0),
            'max': actions.max(axis=0)
        }


class BCDatasetNormalized(BCDataset):
    """带动作归一化的 BC 数据集"""

    def __init__(self, data_path, val_ratio=0.1, is_val=False, seed=42,
                 action_stats=None):
        super().__init__(data_path, val_ratio, is_val, seed)

        # 计算或使用提供的动作统计信息
        if action_stats is None:
            self.action_stats = self.get_action_stats()
        else:
            self.action_stats = action_stats

        self.action_mean = self.action_stats['mean']
        self.action_std = self.action_stats['std'] + 1e-8  # 防止除零

    def __getitem__(self, idx):
        data = super().__getitem__(idx)

        # 归一化动作
        action = data['action'].numpy()
        action_normalized = (action - self.action_mean) / self.action_std
        data['action'] = torch.from_numpy(action_normalized.astype(np.float32))

        return data

    def denormalize_action(self, action):
        """反归一化动作"""
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        return action * self.action_std + self.action_mean
