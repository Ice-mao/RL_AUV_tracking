"""
Target Tracking Environment Base Model.
"""
from typing import List

import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding

import numpy as np
from numpy import linalg as LA
import os, copy

import holoocean

from auv_control import scenario

from auv_env.maps import map_utils
from metadata import METADATA
import auv_env.util as util

from auv_env.world import World


class TargetTrackingBase(gym.Env):
    """
    生成一个3D的HoloOcean环境
    """
    def __init__(self, map="TestMap", num_targets=1,
                 is_training=True, show=True, verbose=True, **kwargs):
        gym.Env.__init__(self)
        np.random.seed()
        self.state = None
        # init some params
        self.num_targets = num_targets
        self.is_training = is_training
        self.polict_period = 0.5  # 控制周期
        self.sensor_r_sd = METADATA['sensor_r_sd']
        self.sensor_b_sd = METADATA['sensor_b_sd']
        self.sensor_r = METADATA['sensor_r']
        self.fov = METADATA['fov']
        self.action_range_high = METADATA['action_range_high']
        self.action_range_low = METADATA['action_range_low']
        self.action_dim = METADATA['action_dim']
        # init the scenario
        self.world = World(map=map, show=show, verbose=verbose, num_targets=self.num_targets)
        # init the action space
        self.action_space = spaces.Box(low=np.float32(self.action_range_low), high=np.float32(self.action_range_high)
                                       , shape=(len(self.action_range_high),))
        self.observation_space = self.world.observation_space
        self.target_init_cov = METADATA['target_init_cov']
        self.reset_num = 0

    def reset(self, **kwargs):
        np.random.seed()
        self.reset_num += 1
        self.has_discovered = [1] * self.num_targets  # if initial state is observed:1
        self.state = []
        return self.world.reset()

    def step(self, action):
        """
        I want to get a waypoint to guidance the learning,and make the action command
        every 0.5s or more time, so how do I calculate the reward?W
        :param action:
        :return:
        """
        # get the policy of (r,theta,angle) in local corrdinate
        if self.is_training is True:
            action_waypoint = np.array([action[0] + 0.1 * np.random.normal(),
                                        action[1] + 0.005 * np.random.normal(),
                                        action[2] + 0.005 * np.random.normal()])
        else:
            action_waypoint = action

        return self.world.step(action_waypoint=action_waypoint)

if __name__ == '__main__':
    env = TargetTrackingBase()
    env.world.ocean.should_render_viewport(False)
    obs, _ = env.reset()
    while True:
        action = [1*np.random.rand(), 1*np.random.rand(), 0.1*np.random.rand(),
                  0.01*np.random.rand(), 0.01*np.random.rand(), 0.01*np.random.rand()]
        obs, reward, done, _, inf = env.step(action)
        print(reward)
