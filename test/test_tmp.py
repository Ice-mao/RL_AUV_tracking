import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# 1. 环境预处理
# def wrap_env(env):
#     """
#     对CarRacing环境进行图像预处理：
#     - 调整图像尺寸为96x96（默认已是，但需显式处理）
#     - 归一化像素值到[0,1]
#     - 堆叠4帧以捕捉时序信息
#     """
#     env = gym.wrappers.ResizeObservation(env, (96, 96, 3))  # 显式调整尺寸
#     # env = gym.wrappers.GrayScaleObservation(env)          # 转换为灰度图（可选）
#     env = gym.wrappers.NormalizeObservation(env)          # 归一化到[0,1]
#     return env

# 创建矢量化环境
env = make_vec_env(
    lambda: gym.make("CarRacing-v2"),
    n_envs=4,              # 并行环境数量，加速训练
    vec_env_cls=DummyVecEnv
)

# 2. 定义PPO模型参数
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torchvision.models as models
class Encoder(BaseFeaturesExtractor):
    """
        Viusal encoder
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        # 加载预训练的ResNet模型
        resnet = models.resnet18(pretrained=True)
        # 去除最后的分类层
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # 计算ResNet输出的形状
        with torch.no_grad():
            n_flatten = self.resnet(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        resnet_output = self.resnet(observations).view(observations.size(0), -1)
        return self.linear(resnet_output)
    
policy_kwargs = dict(
    features_extractor_class=Encoder,
    net_arch=dict(pi=[256, 128], vf=[256]),
    activation_fn=torch.nn.ReLU,
)

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=512,                # 每次迭代的步数
    batch_size=128,              # 批大小
    n_epochs=10,                # 优化迭代次数
    gamma=0.99,                 # 折扣因子
    gae_lambda=0.95,
    clip_range=0.2,
    tensorboard_log="./test/logs/",  # TensorBoard日志路径
    policy_kwargs=policy_kwargs
)

checkpoint_callback = CheckpointCallback(
    save_freq=500000,            # 每10000步保存检查点
    save_path="./test/logs/"
)

model.learn(total_timesteps=1000000, tb_log_name="first_run", log_interval=1, callback=checkpoint_callback)
model.save("./test/logs/ppo_carracing")

# 6. 测试训练后的模型
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    env.render()
    if dones.any():
        obs = env.reset()
env.close()