import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn

env = gym.make("CarRacing-v2", render_mode="human")

class Encoder(BaseFeaturesExtractor):
    """
        Viusal encoder
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512, num_images: int = 5, resnet_output_dim=64):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        # 计算卷积层输出的形状
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations = observations.permute(0, 3, 1, 2)
        return self.linear(self.cnn(observations))
    
policy_kwargs = dict(
    features_extractor_class=Encoder,
    net_arch=dict(pi=[256, 256, 128], vf=[512, 512]),
    activation_fn=torch.nn.ReLU,
)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=512,                # 每次迭代的步数
    batch_size=64,              # 批大小
    n_epochs=10,                # 优化迭代次数
    gamma=0.99,                 # 折扣因子
    gae_lambda=0.95,
    clip_range=0.2,
    tensorboard_log="./logs/",  # TensorBoard日志路径
    policy_kwargs=policy_kwargs
)

model = PPO.load("ppo_carracing", env=env)

# 测试训练后的模型
obs, _ = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones,_ ,info = env.step(action)
    env.render()
env.close()