import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import auv_env
import numpy as np
import gym
from gym import spaces
from tianshou.utils.net.common import SpaceInfo

from auv_track_launcher.wrapper import TeachObsWrapper


if __name__ == '__main__':
    # env = auv_env.make("AUVTracking_rgb",
    #                    render=1,
    #                    num_targets=1,
    #                    eval=True,
    #                    is_training=True,
    #                    t_steps=200,
    #                    )
    env = gym.make('Teacher-v0')
    # print(env.observation_space)
    # env = gym.wrappers.FlattenObservation(env)
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
