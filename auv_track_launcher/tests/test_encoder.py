import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import torch
from gymnasium import spaces
from auv_track_launcher.networks.feature_network import Encoder
batch_size = 4
num_images = 5
image_size = (3, 224, 224)

images = torch.rand(batch_size, num_images, *image_size, device='cuda')  # [batch_size, num_images, 3, 224, 224]
obs_space = spaces.Box(low=-3, high=3, shape=(5, 3, 224, 224), dtype=np.float32)
model = Encoder(observation_space=obs_space, features_dim=512, num_images=num_images, resnet_output_dim=128).to('cuda')
output = model(images)

print("output.shape:", output.shape)  # [batch_size, tcn_output_dim]