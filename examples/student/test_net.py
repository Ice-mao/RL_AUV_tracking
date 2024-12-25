import gymnasium as gym
import auv_env
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv, VecMonitor

import auv_env
import torch
import numpy as np
from numpy import linalg as LA
import csv
import argparse
from policy_net import SEED1, set_seed, CustomCNN
from atrl_launcher.callbacks import SaveOnBestTrainingRewardCallback

# tools
import os
import datetime
from metadata import METADATA


env = gym.make('Teacher-v0')
env = gym.make('Student-v0')
obs, info = env.reset()
# a = env.reset()
while True:
    action = env.action_space.sample()
    print(action)
    # action = np.array([0.0, 0.0, 0.0])
    obs, reward, done, _, inf = env.step(action)


