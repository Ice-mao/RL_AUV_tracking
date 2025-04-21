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

class Encoder(nn.Module):
    """
        Diffusion Viusal Encoder
    """
    def __init__(self, num_channels: list = [512, 256]):
        super(Encoder, self).__init__()
        self.resnet = get_resnet('resnet18', "IMAGENET1K_V1")
        self.resnet = replace_bn_with_gn(self.resnet)
        num_channels
        self.tcn = TemporalConvNet(num_inputs=512, num_channels=num_channels, kernel_size=3,
                                   dropout=0.2)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.tcn.modules():
            if isinstance(module, nn.Conv1d):
                init.orthogonal_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [batch_size, num_images, 3, 128, 128]
        features = self.resnet(images.flatten(end_dim=1))
        features = features.reshape(*images.shape[:2],-1)
        features = features.permute(0, 2, 1)

        encoding = self.tcn(features)
        encoding = encoding.flatten(start_dim=1)
        return encoding
    
if __name__ == "__main__":
    # 参数设置
    batch_size = 4
    num_images = 5  # n+1 张图片
    image_size = (3, 224, 224)  # 符合 ResNet 的输入要求

    # 随机生成输入数据
    images = torch.rand(batch_size, num_images, *image_size)  # [batch_size, num_images, 3, 224, 224]

    model = Encoder()
    output = model(images)  # 输出编码结果

    print("输出编码结果的形状:", output.shape)  # [batch_size, tcn_output_dim]