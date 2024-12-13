import auv_env
import numpy as np
import gymnasium as gym

from tianshou.utils.space_info import SpaceInfo
from atrl_launcher.wrapper import TeachObsWrapper

env = auv_env.make("AUVTracking_rgb",
                   render=1,
                   num_targets=1,
                   eval=True,
                   is_training=True,
                   t_steps=200,
                   )
# print(env.observation_space)
# env = gym.wrappers.FlattenObservation(env)
env = TeachObsWrapper(env)
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
