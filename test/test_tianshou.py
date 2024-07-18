import gymnasium as gym
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import BasePolicy, PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

device = "cuda" if torch.cuda.is_available() else "cpu"

env = gym.make("CartPole-v1")
train_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(20)])
test_envs = DummyVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(10)])