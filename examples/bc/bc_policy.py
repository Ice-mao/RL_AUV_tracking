"""
Behavioral Cloning (BC) Policy with CNN + MLP architecture
用于 AUV 跟踪任务的图像输入行为克隆策略
"""

import torch
import torch.nn as nn
import torchvision.models as models


class CNNEncoder(nn.Module):
    """CNN 编码器，使用 ResNet18 提取图像特征"""

    def __init__(self, pretrained=True, freeze_backbone=False):
        super().__init__()
        # 使用预训练的 ResNet18
        resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

        # 移除最后的全连接层，保留特征提取部分
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512  # ResNet18 输出维度

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 图像输入，值域 [0, 255] uint8 或 [0, 1] float
        Returns:
            features: (B, 512) 图像特征
        """
        # 确保输入是 float 且归一化到 [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # ImageNet 标准化
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x - mean) / std

        features = self.backbone(x)
        features = features.squeeze(-1).squeeze(-1)  # (B, 512, 1, 1) -> (B, 512)
        return features


class MLPHead(nn.Module):
    """MLP 头部，将特征映射到动作空间"""

    def __init__(self, input_dim, action_dim, hidden_dims=[256, 128], dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class BCPolicy(nn.Module):
    """
    Behavioral Cloning Policy
    CNN (ResNet18) + MLP 架构
    输入: 单帧图像 (B, 3, 64, 64)
    输出: 动作 (B, action_dim)
    """

    def __init__(self, action_dim=4, pretrained=True, freeze_backbone=False,
                 hidden_dims=[256, 128], dropout=0.3):
        super().__init__()

        self.encoder = CNNEncoder(pretrained=pretrained,
                                   freeze_backbone=freeze_backbone)
        self.mlp = MLPHead(input_dim=self.encoder.feature_dim,
                          action_dim=action_dim,
                          hidden_dims=hidden_dims,
                          dropout=dropout)

        self.action_dim = action_dim

    def forward(self, obs):
        """
        前向传播
        Args:
            obs: (B, 3, H, W) 图像观测
        Returns:
            action: (B, action_dim) 预测的动作
        """
        features = self.encoder(obs)
        action = self.mlp(features)
        return action

    def predict(self, obs):
        """
        推理时使用，不计算梯度
        Args:
            obs: (3, H, W) 或 (B, 3, H, W) 图像观测
        Returns:
            action: (action_dim,) 或 (B, action_dim) 预测的动作
        """
        self.eval()
        with torch.no_grad():
            if obs.dim() == 3:
                obs = obs.unsqueeze(0)
                action = self.forward(obs)
                return action.squeeze(0)
            else:
                return self.forward(obs)

    def get_action(self, obs, device='cuda'):
        """
        从 numpy 观测获取动作，用于环境交互
        Args:
            obs: numpy array (3, H, W) 或 (H, W, 3)
        Returns:
            action: numpy array (action_dim,)
        """
        import numpy as np

        if isinstance(obs, np.ndarray):
            # 如果是 HWC 格式，转换为 CHW
            if obs.shape[-1] == 3 and obs.ndim == 3:
                obs = obs.transpose(2, 0, 1)
            obs = torch.from_numpy(obs).float()

        obs = obs.to(device)
        action = self.predict(obs)
        return action.cpu().numpy()
