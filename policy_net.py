from stable_baselines3.dqn import policies
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import random

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

SEED1 = 1145


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    任务输入的图像信息为5*28*28
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        # predined
        norm_layer = nn.BatchNorm2d  # 归一化

        # set our network
        self.conv1 = nn.Conv2d(5, 20, kernel_size=4, stride=3, padding=1,
                               bias=False)
        self.bn1 = norm_layer(20)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = norm_layer(40)

        self.linear_fc = nn.Sequential(
            nn.Linear(1000 + 8, features_dim),
            # nn.BatchNorm1d(features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            # nn.BatchNorm1d(features_dim),
            nn.ReLU(),
        )

        # self.lstm = nn.LSTM(1000 + 8, 512, batch_first=True)
        #  初始化网络
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):  # add by xzt
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
            # elif isinstance(m, nn.LSTM):
            #     nn.init.xavier_normal_(m.weight)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # preprocessing:
        # 如果不使用cuda，则需要改成torch.FloatTensor
        map = observations[:, :3920].type(torch.cuda.FloatTensor)
        state = observations[:, 3920:3928].type(torch.cuda.FloatTensor)
        return self._forward_impl(map, state)

    def _forward_impl(self, map, state):
        ###### Start of fusion net ######
        map_in = map.reshape(-1, 5, 28, 28)

        # See note [TorchScript super()]
        x = self.conv1(map_in)  # (1,20,9,9)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)  # (1,40,5,5)
        x = self.bn2(x)
        x = self.relu(x)

        map_out = torch.flatten(x, 1)  # [1,1000]
        ###### End of fusion net ######

        ###### Start of state net #######
        state_out = state.reshape(-1, 8)

        ###### End of goal net #######
        # Combine
        fc_in = torch.cat((map_out, state_out), dim=1)  # [1,1008]
        # fc_in = self.lstm(fc_in)
        x = self.linear_fc(fc_in)  # [1,512]

        return x


if __name__ == "__main__":
    test_cnn = CustomCNN
    observations = torch.from_numpy(np.ones((1, 3928)))
    map = observations[:, :3920].type(torch.FloatTensor)
    map_in = map.reshape(-1, 5, 28, 28)
    # predined
    norm_layer = nn.BatchNorm2d  # 归一化

    # set our network
    conv1 = nn.Conv2d(5, 20, kernel_size=4, stride=3, padding=1,
                           bias=False)
    bn1 = norm_layer(20)
    relu = nn.ReLU(inplace=True)
    conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=2, padding=1,
                           bias=False)
    bn2 = norm_layer(40)

    linear_fc = nn.Sequential(
        nn.Linear(1000 + 8, 512),
        # nn.BatchNorm1d(features_dim),
        nn.ReLU())
    # See note [TorchScript super()]
    x = conv1(map_in)  # (1,20,9,9)
    x = bn1(x)
    x = relu(x)
    x = conv2(x)  # (1,40,5,5)
    x = bn2(x)
    x = relu(x)