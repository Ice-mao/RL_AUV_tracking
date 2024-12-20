import auv_env
import numpy as np
import gymnasium as gym

from tianshou.utils.space_info import SpaceInfo
from tianshou.env import DummyVectorEnv, SubprocVectorEnv, ShmemVectorEnv
from atrl_launcher.wrapper import TeachObsWrapper
from examples.mujoco.mujoco_env import make_mujoco_env

# print(env.observation_space)

train_env = SubprocVectorEnv([lambda: gym.make('Teacher-v0') for _ in range(3)],)
test_env = SubprocVectorEnv([lambda: gym.make('Teacher-v0') for _ in range(1)],)
env = gym.make('Teacher-v0')

space_info = SpaceInfo.from_env(env)

state_shape = space_info.observation_info.obs_shape
action_shape = space_info.action_info.action_shape

(obs, info) = env.reset()
# a = env.reset()
while True:
    action = env.action_space.sample()
    print(action)
    # action = np.array([0.0, 0.0, 0.0])
    obs, reward, done, _, inf = env.step(action)
