from typing import Dict, Callable
import torch
import numpy as np
import copy
from torchvision import transforms
from imitation.data import huggingface_utils
import datasets
from auv_track_launcher.dataset.sampler import SequenceSampler
import torchvision.transforms.functional as F

class HoloOceanImageDataset(torch.utils.data.Dataset):
    def __init__(self,
            dataset, 
            horizon=6,
            obs_horizon=2,
            pred_horizon=4,
            pad_before=0,
            pad_after=0,
            ):
        
        super().__init__()
        _dataset = copy.copy(dataset)
        self.replay_buffer = huggingface_utils.TrajectoryDatasetSequence(_dataset)
        del _dataset
        # zarr_path, keys=[acts, obs, rews, infos])

        assert horizon >= obs_horizon + pred_horizon, \
            f"horizon size ({horizon}) must Greater than or equal to obs_horizon ({obs_horizon}) + pred_horizon ({pred_horizon}) = {obs_horizon + pred_horizon}"
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after)
        self.horizon = horizon
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_horizon(self, sample):
        """
            slice
        """
        if "obs" in sample:
            sample["obs"] = sample['obs'][:self.obs_horizon]
        else:
            assert False, "obs not in sample"
            
        if "acts" in sample:
            sample["acts"] = sample['acts'][-self.pred_horizon:]
        else:
            assert False, "acts not in sample"
            
        return sample
    
    def _sample_to_data(self, sample):
        """
            data precessing func. v1: ignore reward
            obs: normalize
            action: to [-1, 1]
        """
        data = {
            'obs': sample["obs"], # T, 3, 224, 224
            'action': sample["acts"], # T, 3
            # 'reward': sample["rews"] # T,
        }
        return data
    
    def _custom_transform(self, torch_data):
        """
            custom transform func
            1. obs: resize, normalize, data augment
            2. action: to [-1, 1]
        """
        torch_data['obs'] = F.resize(torch_data['obs'], (128, 128))
        
        # 随机数据增强 - 每次选择一种方法应用
        aug_choice = torch.rand(1).item()
        
        # 1. 随机水平翻转 (5%概率)
        if aug_choice < 0.05:
            torch_data['obs'] = F.hflip(torch_data['obs'])
        
        # 2. 随机平移 (5%概率)
        elif aug_choice < 0.1:
            # 随机生成平移量，最大为图像尺寸的10%
            max_dx = int(128 * 0.1)
            max_dy = int(128 * 0.1)
            dx = torch.randint(-max_dx, max_dx+1, (1,)).item()
            dy = torch.randint(-max_dy, max_dy+1, (1,)).item()
            
            # 使用仿射变换实现平移
            torch_data['obs'] = F.affine(
                torch_data['obs'], 
                angle=0.0,  # 不旋转
                translate=[dx, dy],  # 平移量
                scale=1.0,  # 不缩放
                shear=0.0,  # 不剪切
                interpolation=F.InterpolationMode.BILINEAR
            )
        
        # 3. 随机旋转 (5%概率)
        elif aug_choice < 0.15:
            # 随机旋转±15度
            angle = torch.FloatTensor(1).uniform_(-15, 15).item()
            torch_data['obs'] = F.rotate(
                torch_data['obs'], 
                angle,
                interpolation=F.InterpolationMode.BILINEAR,
                fill=0  # 边缘填充黑色
            )
        
        # 4. 随机亮度和对比度 (5%概率)
        elif aug_choice < 0.2:
            # 亮度调整因子
            brightness_factor = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
            torch_data['obs'] = F.adjust_brightness(torch_data['obs'], brightness_factor)
            
            # 对比度调整因子
            contrast_factor = torch.FloatTensor(1).uniform_(0.8, 1.2).item()
            torch_data['obs'] = F.adjust_contrast(torch_data['obs'], contrast_factor)
            
        # 按照您要求的参数进行归一化: 均值0.5, 标准差sqrt(1/12)≈0.289
        torch_data['obs'] = F.normalize(
            torch_data['obs'], 
            mean=[0.485, 0.456, 0.406],  # ImageNet标准均值
            std=[0.229, 0.224, 0.225]    # ImageNet标准差
        )

        torch_data['action'][:, 0] = 2.0 * torch_data['action'][:, 0] - 1.0
        return torch_data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        sample = self._sample_to_horizon(sample)
        sample = self._sample_to_data(sample)
        torch_data = dict_apply(sample, torch.from_numpy)
        # 标准化动作第一列
        torch_data = self._custom_transform(torch_data)
        return torch_data
    
def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def test():
    import os
    dataset_0 = datasets.load_from_disk("log/sample/trajs_dam/traj_1")
    dataset_1 = datasets.load_from_disk("log/sample/trajs_dam/traj_2")
    _dataset = datasets.concatenate_datasets([dataset_0, dataset_1])
    dataset = HoloOceanImageDataset(_dataset, horizon=16, obs_horizon=4, pred_horizon=10)
    dataset.__getitem__(0)
    print("debug")


if __name__ == '__main__':
    test()