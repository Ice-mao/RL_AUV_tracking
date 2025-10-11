"""
AUV tracking task dataset implementation

This file demonstrates how to create specialized dataset classes for AUV tracking tasks
Contains processing of sensor data (sonar, camera) and control actions
Uses zarr format for data storage
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
            sonar_range: float = 50.0,  # Maximum sonar detection range
            ):
        
        super().__init__()
        
        # Save configuration
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.image_size = image_size
        self.sonar_range = sonar_range
        
        self.data_keys = key
        
        # Load data
        self.replay_buffer = self._load_auv_data(data_path)
        
        # Create train/validation split
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # Create sampler
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.train_mask = train_mask

    def _load_auv_data(self, data_path: str) -> ReplayBuffer:
        """Load AUV data (zarr format)"""
        if not data_path.endswith('.zarr'):
            raise ValueError(f"Only zarr format data files are supported, current file: {data_path}")
        
        return ReplayBuffer.copy_from_path(data_path, keys=self.data_keys)

    def get_validation_dataset(self):
        """Create validation set"""
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
        data = {
            'action': self.replay_buffer['action'],
            'state': self.replay_buffer['state']
        } 
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        
        # Image normalizer
        normalizer['camera_image'] = get_image_range_normalizer()
        
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        """Convert sampled data to model input format"""
        # Basic observation dictionary
        obs = {}
        
        # Process image data
        if 'camera_image' in self.data_keys:
            images = sample['camera_image']  # (T, H, W, C)
            obs['camera_image'] = images.astype(np.float32)
        
        # Process sonar data
        if 'sonar_image' in self.data_keys:
            sonar = sample['sonar_data'].astype(np.float32)
            # Normalize sonar data to [0,1]
            obs['sonar'] = sonar / self.sonar_range
        
        # Process state
        if 'state' in self.data_keys:
            obs['state'] = sample['state']
        
        # Action data
        action = sample['action'].astype(np.float32)
        
        return {
            'obs': obs,
            'action': action
        }
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training sample"""
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics"""
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
        """Visualize data sample"""
        sample = self[idx]
        
        print(f"=== Data Sample {idx} ===")
        print(f"Observation data:")
        for key, value in sample['obs'].items():
            print(f"  {key}: {value.shape}")
        
        print(f"Action data: {sample['action'].shape}")
        
        # If there are images, save the first frame
        if 'camera_image' in sample['obs']:
            import matplotlib.pyplot as plt
            image = sample['obs']['camera_image'][0].permute(1, 2, 0).numpy()
            plt.figure(figsize=(8, 6))
            plt.imshow(image)
            plt.title(f"Sample {idx} - First Frame")
            plt.axis('off')
            plt.savefig(f'sample_{idx}_image.png')
            print(f"  Image saved: sample_{idx}_image.png")
