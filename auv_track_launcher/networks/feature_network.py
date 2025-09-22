# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from auv_track_launcher.networks.tcn import TemporalConvNet
from auv_track_launcher.networks.rgb_net import EncoderResNet
from auv_track_launcher import utils


class Encoder(BaseFeaturesExtractor):
    """
        Visual encoder
    """
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512, resnet_output_dim=64):
        super().__init__(observation_space, 1)
        self.resnet = EncoderResNet(encoder_dim=resnet_output_dim)
        # num_channels = [128, 64]
        # self.tcn = TemporalConvNet(num_inputs=resnet_output_dim, num_channels=num_channels, kernel_size=2,
        #                            dropout=0.2)
        self.trunk = nn.Sequential(nn.Linear(resnet_output_dim, features_dim),
                                   nn.LayerNorm(features_dim), nn.Tanh())
        self.apply(utils.weight_init)

        # Update the features dim manually
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        images = observations['images']
        # 输入 images: [batch_size, 3, H, W]
        batch_size, C, H, W = images.size()
        feature = self.resnet(images)  # 提取特征: [batch_size, resnet_output_dim]
        output = self.trunk(feature)
        return output


if __name__ == "__main__":
    # 参数设置
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
