# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from auv_track_launcher.networks.tcn import TemporalConvNet

from auv_track_launcher import utils


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
    def __init__(self, encoder_dim=64):
        super(EncoderResNet, self).__init__()
        resnet = models.resnet50('IMAGENET1K_V1')
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]) # 去掉 fc 层
        self.fc = nn.Linear(in_features=2048, out_features=encoder_dim, bias=True)

    def forward(self, x):
        # x: [batch_size, 3, H, W]
        features = self.feature_extractor(x)  # [batch_size, 512, 1, 1]
        features_fc_input = features.view(features.size(0), -1)
        output = self.fc(features_fc_input)
        return output  # [batch_size, 128]


class EncoderTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, kernel_size=3, device='cuda'):
        super(EncoderTCN, self).__init__()
        layers = []
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=(kernel_size - 1)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
        self.tcn = nn.Sequential(*layers).to(device)
        self.fc = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, x):
        x = x.permute(0, 2, 1) # [batch_size, seq_len, input_dim] -> [batch_size, input_dim, seq_len]
        x = self.tcn(x)  # [batch_size, hidden_dim, seq_len]
        x = torch.mean(x, dim=-1)  # [batch_size, hidden_dim]
        return self.fc(x)  # [batch_size, output_dim]


class Encoder(nn.Module):
    def __init__(self, N, resnet_output_dim=64, output_dim=256, tcn_hidden_dim=512, tcn_output_dim=256, tcn_layers=2,
                 device='cuda'):
        super(Encoder, self).__init__()
        self.num_images = N
        self.resnets = nn.ModuleList([EncoderResNet().to(device) for _ in range(self.num_images)])
        # self.tcn = EncoderTCN(input_dim=resnet_output_dim*N, hidden_dim=tcn_hidden_dim,
        #                       output_dim=tcn_output_dim, num_layers=tcn_layers, device=device)
        num_channels = [128, 64]
        self.tcn = TemporalConvNet(num_inputs=resnet_output_dim, num_channels=num_channels, kernel_size=2,
                                   dropout=0.2).to(device)
        self.linear = nn.Linear(num_channels[-1], output_dim).to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 输入 images: [batch_size, num_images, 3, H, W]
        batch_size, num_images, C, H, W = images.size()
        assert num_images == self.num_images, "Input number of images must match num_images."

        features = []
        for i in range(num_images):
            img = images[:, i, :, :, :]  # i-th image : [batch_size, 3, H, W]
            feature = self.resnets[i](img)  # [batch_size, resnet_output_dim]
            features.append(feature)

        # [batch_size, N, resnet_output_dim]
        features = torch.stack(features, dim=1)
        features = features.permute(0, 2, 1)

        encoding = self.tcn(features)  # [batch_size, tcn_output_dim]
        encoding = encoding.mean(dim=-1)
        output = self.linear(encoding)
        return output

if __name__ == "__main__":
    batch_size = 4
    num_images = 5
    image_size = (3, 224, 224)

    images = torch.rand(batch_size, num_images, *image_size, device='cuda')  # [batch_size, num_images, 3, 224, 224]
    model = Encoder(N=num_images)
    output = model(images)

    print("output.shape:", output.shape)  # [batch_size, tcn_output_dim]
