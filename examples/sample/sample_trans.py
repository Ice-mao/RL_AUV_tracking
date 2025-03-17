import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
import gymnasium as gym

from imitation.algorithms import bc
import custom_rollout
# from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data import serialize

import auv_env
def sample_expert_transitions(expert: BasePolicy, env: VecEnv):
    print("Sampling expert transitions.")
    rollouts = custom_rollout.rollout(
        expert,
        env,
        custom_rollout.make_sample_until(min_episodes=1),
        rng=rng,
        unwrap=False,
    )
    return rollouts
    # return custom_rollout.flatten_trajectories(rollouts)

from typing import Mapping, Sequence, cast
from imitation.data.types import AnyPath, Trajectory, TrajectoryWithRew
from imitation.data import huggingface_utils
from imitation.util import util
import logging
def custom_save(path: AnyPath, trajectories: Sequence[Trajectory]) -> None:
    """Save a sequence of Trajectories to disk using HuggingFace's datasets library.

    Args:
        path: Trajectories are saved to this path.
        trajectories: The trajectories to save.
    """
    p = util.parse_path(path)
    print("debug")
    huggingface_utils.trajectories_to_dataset(trajectories).save_to_disk(p, max_shard_size="10GB")
    logging.info(f"Dumped demonstrations to {p}.")

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("Get the expert policy")
    expert_env = make_vec_env(
        "Student-v0-sample-teacher",
        rng=rng,
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
    )
    expert = load_policy("sac", venv=expert_env,
                         path="/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/teacher/SAC/01-13_12/rl_model_1240000_steps.zip")
    print("Sample the transitions")
    env = make_vec_env(
        "Student-v0-sample",
        rng=rng,
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
    )
    rollouts = sample_expert_transitions(expert, env)
    # serialize.save(path="trajs_0", trajectories=rollouts)
    custom_save(path="../../log/sample/trajs_dam/traj_1", trajectories=rollouts)
    print("debug before")