import numpy as np
import os
import datasets
import torch
from typing import Mapping, Sequence, cast

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.utils import get_schedule_fn

# from imitation.algorithms import bc
from imitation.data import rollout, serialize, huggingface_utils
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data.types import AnyPath, Trajectory, TrajectoryWithRew
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

from auv_track_launcher.algorithms.bc import BC
from auv_track_launcher.networks import Encoder

import auv_env

from typing import Mapping, Sequence, cast
from imitation.data.types import AnyPath, Trajectory, TrajectoryWithRew
from imitation.data import huggingface_utils
from imitation.util import util
import logging

if __name__ == "__main__":
    print("Load the transitions")
    dataset_1 = datasets.load_from_disk("/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/trajs_dam/traj_1")
    dataset_2 = datasets.load_from_disk("/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/trajs_dam/traj_2")
    dataset_3 = datasets.load_from_disk("/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/trajs_dam/traj_3")
    dataset_4 = datasets.load_from_disk("/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/trajs_dam/traj_4")
    dataset_5 = datasets.load_from_disk("/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/trajs_dam/traj_5")
    dataset_6 = datasets.load_from_disk("/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/trajs_dam/traj_6")
    dataset_7 = datasets.load_from_disk("/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/trajs_dam/traj_7")
    dataset_8 = datasets.load_from_disk("/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/trajs_dam/traj_8")
    dataset_9 = datasets.load_from_disk("/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/trajs_dam/traj_9")
    # dataset = datasets.concatenate_datasets([dataset_0, dataset_1, dataset_2, dataset_3])
    dataset = datasets.concatenate_datasets([dataset_1, dataset_2, dataset_3, dataset_4, dataset_5, dataset_6, dataset_7, dataset_8, dataset_9])
    dataset.save_to_disk("/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/trajs_dam/trajs_dam", max_shard_size="100GB")