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


def make_teacher_env(
        task: str,
        num_train_envs: int,
        monitor_dir: str
) -> VecEnv:
    if 'Teacher' not in task:
        raise ValueError("you should use teacher env.")
    train_envs = SubprocVecEnv([lambda: gym.make(task) for _ in range(num_train_envs)], )
    env = VecMonitor(train_envs, monitor_dir)
    return env


a = SubprocVecEnv([lambda: gym.make('Teacher-v0') for _ in range(3)])
vec_env = make_teacher_env('Teacher-v0', 3, '../../log')
obs = vec_env.reset()
print(obs)


