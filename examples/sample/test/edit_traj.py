import numpy as np
import datasets
from imitation.data import huggingface_utils, rollout
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import copy
from auv_track_launcher.dataset.holoocean_image_dataset import HoloOceanImageDataset

# class PushTImageDataset(torch.utils.data.Dataset):
#     def __init__(self,
#             zarr_path, 
#             horizon=1,
#             pad_before=0,
#             pad_after=0,
#             seed=42,
#             val_ratio=0.0,
#             max_train_episodes=None
#             ):
        
#         super().__init__()
#         self.replay_buffer = ReplayBuffer.copy_from_path(
#             zarr_path, keys=['img', 'state', 'action'])

#         self.sampler = SequenceSampler(
#             replay_buffer=self.replay_buffer, 
#             sequence_length=horizon,
#             pad_before=pad_before, 
#             pad_after=pad_after)
#         self.horizon = horizon
#         self.pad_before = pad_before
#         self.pad_after = pad_after

#     def get_validation_dataset(self):
#         val_set = copy.copy(self)
#         val_set.sampler = SequenceSampler(
#             replay_buffer=self.replay_buffer, 
#             sequence_length=self.horizon,
#             pad_before=self.pad_before, 
#             pad_after=self.pad_after,
#             episode_mask=~self.train_mask
#             )
#         val_set.train_mask = ~self.train_mask
#         return val_set

#     def get_normalizer(self, mode='limits', **kwargs):
#         normalizer = LinearNormalizer()
#         normalizer['action'] = IdentityNormalizer()
#         normalizer['agent_pos'] = IdentityNormalizer()
#         normalizer['image'] = IdentityNormalizer()
#         return normalizer

#     def __len__(self) -> int:
#         return len(self.sampler)

#     def _sample_to_data(self, sample):
#         agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
#         image = np.moveaxis(sample['img'],-1,1)/255

#         data = {
#             'obs': {
#                 'image': image, # T, 3, 96, 96
#                 'agent_pos': agent_pos, # T, 2
#             },
#             'action': sample['action'].astype(np.float32) # T, 2
#         }
#         return data
    
#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         sample = self.sampler.sample_sequence(idx)
#         data = self._sample_to_data(sample)
#         torch_data = dict_apply(data, torch.from_numpy)
#         return torch_data

from datasets import load_dataset, Image
from torchvision.transforms import RandomRotation

# dataset = load_dataset("AI-Lab-Makerere/beans", split="train")
# rotate = RandomRotation(degrees=(0, 90))
# def transforms(examples):
#     examples["pixel_values"] = [rotate(image) for image in examples["image"]]
#     return examples

dataset_0 = datasets.load_from_disk("log/sample/trajs_dam/traj_1")
dataset_1 = datasets.load_from_disk("log/sample/trajs_dam/traj_2")
dataset = datasets.concatenate_datasets([dataset_0, dataset_1])
# transitions = huggingface_utils.TrajectoryDatasetSequence(dataset)
dataset = HoloOceanImageDataset(
    dataset=dataset,
    horizon=6,
    obs_horizon=2,
    pred_horizon=4
)

# create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=16,
    num_workers=2,
    shuffle=True,
    # accelerate cpu-gpu transfer
    pin_memory=True,
    # don't kill worker process afte each epoch
    persistent_workers=True
)

print("start sampling")
# visualize data in batch
batch = next(iter(dataloader))
# print("batch['obs']['image'].shape:", batch['obs']['image'].shape)
# print("batch['obs']['agent_pos'].shape:", batch['obs']['agent_pos'].shape)
print("batch['action'].shape", batch['action'].shape)
print("batch['reward'].shape", batch['reward'].shape)