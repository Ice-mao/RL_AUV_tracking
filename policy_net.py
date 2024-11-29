from stable_baselines3.dqn import policies
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.conv1 = nn.Conv2d(1, 8, kernel_size=4, stride=3, padding=1,
                               bias=False)
        self.bn1 = norm_layer(8)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = norm_layer(32)

        self.conv1_ = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2_ = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3_ = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

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
        map = observations[:, :65535].type(torch.cuda.FloatTensor)
        state = observations[:, 65535:].type(torch.cuda.FloatTensor)
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

class PPO_withgridmap(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        # CNN for sonar grid map
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)

        # Fully connected layer for CNN output
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # Assuming input size 64x64

        # Fully connected layer for robot state vector
        self.fc2 = nn.Linear(10, 64)

        # Combined fully connected layers
        self.fc3 = nn.Linear(256 + 64, features_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # preprocessing:
        # 如果不使用cuda，则需要改成torch.FloatTensor
        map = observations[:, :4096].type(torch.cuda.FloatTensor)
        state = observations[:, 4096:].type(torch.cuda.FloatTensor)
        return self._forward_impl(map, state)

    def _forward_impl(self, grid_map, state_vector):
        # CNN feature extraction
        grid_map = grid_map.reshape(-1, 1, 64, 64)
        x = F.relu(self.conv1(grid_map))
        x = self.bn1(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool(x)

        # Flatten CNN output
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        # Process state vector
        state = state_vector.reshape(-1, 10)
        s = F.relu(self.fc2(state))

        # Concatenate features
        combined = torch.cat((x, s), dim=1)
        combined = F.relu(self.fc3(combined)) # (1, 128)

        return combined


if __name__ == "__main__":
    test_cnn = PPO_withgridmap(spaces.Box(low=1, high=2)).to("cuda")
    observations = torch.from_numpy(np.ones((1, 4096+10)))
    map = observations[:, :4096].type(torch.cuda.FloatTensor)
    state = observations[:, 4096:].type(torch.cuda.FloatTensor)
    map_in = map.reshape(-1, 1, 64, 64)
    test_cnn.forward(observations)
    x = test_cnn(observations)
    x = test_cnn.conv1_(map_in) # (1,16,64,64)
    x = test_cnn.relu(x)
    x = test_cnn.pool(x) # (1,16,32,32)
    x = test_cnn.conv2_(x)  # (1,32，32，32）
    x = test_cnn.relu(x)
    x = test_cnn.pool(x) # (1,32，16，16）
    x = test_cnn.conv3_(x) # (1,64，16，16）
    x = test_cnn.relu(x)
    x = test_cnn.pool(x) # (1,64，8，8）
    print("debug")
    # # set our network
    # conv1 = nn.Conv2d(5, 20, kernel_size=4, stride=3, padding=1,
    #                        bias=False)
    # bn1 = norm_layer(20)
    # relu = nn.ReLU(inplace=True)
    # conv2 = nn.Conv2d(20, 40, kernel_size=3, stride=2, padding=1,
    #                        bias=False)
    # bn2 = norm_layer(40)
    #
    # linear_fc = nn.Sequential(
    #     nn.Linear(1000 + 8, 512),
    #     # nn.BatchNorm1d(features_dim),
    #     nn.ReLU())
    # # See note [TorchScript super()]
    # x = conv1(map_in)  # (1,20,9,9)
    # x = bn1(x)
    # x = relu(x)
    # x = conv2(x)  # (1,40,5,5)
    # x = bn2(x)
    # x = relu(x)