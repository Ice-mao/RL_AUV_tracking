# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torchvision
from torchvision import models
from auv_track_launcher.networks.tcn import TemporalConvNet

from gymnasium import spaces
from auv_track_launcher.networks.utils import get_resnet, replace_submodules, replace_bn_with_gn

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class EncoderResNet(nn.Module):
    def __init__(self, encoder_dim=128):
        super(EncoderResNet, self).__init__()
        # 加载预训练的 ResNet-18 模型
        resnet = models.resnet18('IMAGENET1K_V1')
        # 去掉最后的全连接层，保留到倒数第二层
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]) # 去掉 fc 层
        self.fc = nn.Linear(in_features=512, out_features=encoder_dim, bias=True)
        self._init_fc()

    def _init_fc(self):
        # 初始化新增的 fc 层
        init.xavier_normal_(self.fc.weight, gain=init.calculate_gain('tanh'))
        init.zeros_(self.fc.bias)
    def forward(self, x):
        # 输入 x: [batch_size, 3, H, W]
        features = self.feature_extractor(x)  # 输出: [batch_size, 512, 1, 1]
        features_fc_input = features.view(features.size(0), -1)
        output = self.fc(features_fc_input)
        return output  # 展平: [batch_size, encoder_dim]


class Encoder(BaseFeaturesExtractor):
    """
        Viusal encoder
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512, num_images: int = 5):
        super().__init__(observation_space, features_dim)
        self.num_images = num_images
        # self.resnet = EncoderResNet(encoder_dim=resnet_output_dim)
        self.resnet = get_resnet('resnet18', "IMAGENET1K_V1")
        self.resnet = replace_bn_with_gn(self.resnet)
        
        num_channels = [256, 128]
        self.tcn = TemporalConvNet(num_inputs=512, num_channels=num_channels, kernel_size=3,
                                   dropout=0.2)
        self.trunk = nn.Sequential(nn.Linear(num_channels[-1], features_dim),
                                   nn.LayerNorm(features_dim), nn.Tanh())
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.tcn.modules():
            if isinstance(module, nn.Conv1d):
                init.orthogonal_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

        # 初始化 Trunk 的 Linear 层
        init.xavier_normal_(self.trunk[0].weight, gain=init.calculate_gain('tanh'))
        init.zeros_(self.trunk[0].bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 输入 images: [batch_size, num_images, 3, H, W]
        batch_size, num_images, C, H, W = images.size()
        assert num_images == self.num_images, "Input number of images must match num_images."

        # 提取每张图像的特征
        features = []
        for i in range(num_images):
            img = images[:, i, :, :, :]  # 取出第 i 张图像: [batch_size, 3, H, W]
            feature = self.resnet(img)  # 提取特征: [batch_size, resnet_output_dim]
            features.append(feature)

        # 拼接特征: [batch_size, N, resnet_output_dim]
        features = torch.stack(features, dim=1)
        features = features.permute(0, 2, 1)

        # 输入 TCN 进行时序建模
        encoding = self.tcn(features)  # 输出: [batch_size, tcn_output_dim, num_images]
        encoding = torch.mean(encoding, dim=-1)
        output = self.trunk(encoding)
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
    model = Encoder(observation_space=obs_space, features_dim=512, num_images=num_images).to('cuda')
    output = model(images)  # 输出编码结果

    print("输出编码结果的形状:", output.shape)  # [batch_size, tcn_output_dim]
