# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
自定义 SAC Policy 实现 Actor 和 Critic 分离输入

Actor: 使用图像特征 (通过 Encoder → 512维)
Critic: 使用 state + action (6维 + 3维 = 9维)
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import create_mlp


class StateCritic(nn.Module):
    """
    自定义 Critic 网络，直接使用 state 而不是图像特征
    
    输入: state (6维) + action (3维) = 9维
    输出: Q值 (每个 Q 网络输出 1 维)
    
    Args:
        state_dim: state 的维度 (6)
        action_dim: action 的维度 (3)
        net_arch: Q 网络的隐藏层结构，例如 [512, 256]
        activation_fn: 激活函数
        n_critics: Q 网络的数量 (SAC 使用 2 个 Q 网络)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        n_critics: int = 2,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_critics = n_critics
        
        # 创建多个 Q 网络 (SAC 使用 2 个)
        self.q_networks = nn.ModuleList()
        for _ in range(n_critics):
            # 输入维度 = state_dim + action_dim
            q_net = create_mlp(
                input_dim=state_dim + action_dim,
                output_dim=1,
                net_arch=net_arch,
                activation_fn=activation_fn,
            )
            q_net = nn.Sequential(*q_net)
            self.q_networks.append(q_net)
    
    def forward(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            obs: 观测字典，包含 'image' 和 'state'
            actions: 动作张量 [batch_size, action_dim]
        
        Returns:
            Tuple of Q-values from each Q-network
        """
        # 从观测字典中提取 state
        state = obs['state']  # [batch_size, state_dim]
        
        # 拼接 state 和 action
        qvalue_input = torch.cat([state, actions], dim=1)  # [batch_size, state_dim + action_dim]
        
        # 通过所有 Q 网络
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)
    
    def q1_forward(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        """
        仅使用第一个 Q 网络进行前向传播 (用于 policy loss 计算)
        """
        state = obs['state']
        qvalue_input = torch.cat([state, actions], dim=1)
        return self.q_networks[0](qvalue_input)

    def set_training_mode(self, mode: bool) -> None:
        """
        设置训练模式

        Args:
            mode: True for training mode, False for evaluation mode
        """
        self.train(mode)


class CustomSACPolicy(SACPolicy):
    """
    自定义 SAC Policy
    
    关键特性:
    - Actor 使用图像编码器 (Encoder) → 512维特征
    - Critic 使用 state-only → 直接使用 6维 state + 3维 action
    - share_features_extractor=False (Actor 和 Critic 不共享特征提取器)
    """
    
    def __init__(self, *args, **kwargs):
        # 强制不共享特征提取器
        kwargs['share_features_extractor'] = False
        super().__init__(*args, **kwargs)
    
    def make_critic(self, features_extractor: Optional[nn.Module] = None) -> StateCritic:
        """
        创建自定义 Critic 网络
        
        注意: 此方法忽略 features_extractor 参数，因为我们直接使用 state
        
        Args:
            features_extractor: 特征提取器 (被忽略)
        
        Returns:
            StateCritic 实例
        """
        # 获取 state 和 action 的维度
        state_dim = self.observation_space['state'].shape[0]  # 6
        action_dim = get_action_dim(self.action_space)  # 3
        
        # 从 net_arch 中获取 critic 网络结构
        critic_arch = self.net_arch.get('qf', [512, 256])
        
        # 创建自定义 StateCritic
        critic = StateCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            net_arch=critic_arch,
            activation_fn=self.activation_fn,
            n_critics=2,  # SAC 使用 2 个 Q 网络
        ).to(self.device)
        
        return critic
    
    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> torch.Tensor:
        """
        前向传播 - 用于采样动作
        
        Args:
            obs: 观测字典
            deterministic: 是否使用确定性策略
        
        Returns:
            actions: 采样的动作
        """
        return self._predict(obs, deterministic=deterministic)
