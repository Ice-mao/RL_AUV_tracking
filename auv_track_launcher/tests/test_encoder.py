import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import torch
from gymnasium import spaces
from auv_track_launcher.networks.student_network import Encoder
batch_size = 4
num_images = 5  # n+1 张图片
image_size = (3, 224, 224)  # 符合 ResNet 的输入要求

# 随机生成输入数据
images = torch.rand(batch_size, num_images, *image_size, device='cuda')  # [batch_size, num_images, 3, 224, 224]

obs_space = spaces.Box(low=-3, high=3, shape=(5, 3, 224, 224), dtype=np.float32)
# 初始化网络
model = Encoder(observation_space=obs_space, features_dim=512, num_images=num_images, resnet_output_dim=128).to('cuda')
output = model(images)  # 输出编码结果

print("输出编码结果的形状:", output.shape)  # [batch_size, tcn_output_dim]