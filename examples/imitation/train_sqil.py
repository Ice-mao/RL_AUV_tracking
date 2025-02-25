import numpy as np
import os
import datasets
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

from sb3_launcher.algorithms.bc import BC
from sb3_launcher.networks import Encoder

import auv_env


def custom_load(path: AnyPath) -> Sequence[Trajectory]:
    """Loads a sequence of trajectories saved by `save()` from `path`."""
    # Interestingly, np.load will just silently load a normal pickle file when you
    # set `allow_pickle=True`. So this call should succeed for both the new compressed
    # .npz format and the old pickle based format. To tell the difference, we need to
    # look at the type of the resulting object. If it's the new compressed format,
    # it should be a Mapping that we need to decode, whereas if it's the old format,
    # it's just the sequence of trajectories, and we can return it directly.

    if os.path.isdir(path):  # huggingface datasets format
        dataset = datasets.load_from_disk(str(path))
        if not isinstance(dataset, datasets.Dataset):  # pragma: no cover
            raise ValueError(
                f"Expected to load a `datasets.Dataset` but got {type(dataset)}",
            )

        return huggingface_utils.TrajectoryDatasetSequence(dataset)

    data = np.load(path, allow_pickle=True)  # works for both .npz and .pkl

    if isinstance(data, Sequence):  # pickle format
        warnings.warn("Loading old pickle version of Trajectories", DeprecationWarning)
        return data
    if isinstance(data, Mapping):  # .npz format
        warnings.warn("Loading old npz version of Trajectories", DeprecationWarning)
        num_trajs = len(data["indices"])
        fields = [
            # Account for the extra obs in each trajectory
            np.split(data["obs"], data["indices"] + np.arange(num_trajs) + 1),
            np.split(data["acts"], data["indices"]),
            np.split(data["infos"], data["indices"]),
            data["terminal"],
        ]
        if "rews" in data:
            fields = [
                *fields,
                np.split(data["rews"], data["indices"]),
            ]
            return [TrajectoryWithRew(*args) for args in zip(*fields)]
        else:
            return [Trajectory(*args) for args in zip(*fields)]  # pragma: no cover
    else:  # pragma: no cover
        raise ValueError(
            f"Expected either an .npz file or a pickled sequence of trajectories; "
            f"got a pickled object of type {type(data).__name__}",
        )


def create_ppo_policy(env) -> ActorCriticPolicy:
    policy_kwargs = dict(
        # features_extractor_class=CustomCNN,
        # features_extractor_kwargs=dict(features_dim=256),
        # net_arch=[512, 512],
        net_arch=dict(pi=[512, 512, 512], vf=[512, 512]),
    )
    policy = ActorCriticPolicy(
        env.observation_space, env.action_space, use_sde=False, **policy_kwargs
    )
    return policy


def create_sac_policy(env) -> SACPolicy:
    schedule = get_schedule_fn(0.0001)
    policy_kwargs = dict(
        features_extractor_class=Encoder,
        features_extractor_kwargs=dict(features_dim=512, num_images=5, resnet_output_dim=128),
        net_arch=dict(pi=[512, 512, 512], qf=[512, 512]),  # for AC policy
    )
    policy = SACPolicy(
        env.observation_space, env.action_space, lr_schedule=schedule, use_sde=False, **policy_kwargs
    )
    return policy


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    env = make_vec_env(
        "Student-v0-norender",
        rng=rng,
        n_envs=4,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
    )
    from gymnasium import spaces

    action_space = spaces.Box(low=np.float32([0, -1, -1]),
                              high=np.float32([1, 1, 1]),
                              dtype=np.float32)
    obs_space = spaces.Box(low=-3, high=3, shape=(5, 3, 224, 224), dtype=np.float32)
    print("Load the transitions")
    dataset_0 = datasets.load_from_disk("../../log/imitation/trajs_0")
    dataset_1 = datasets.load_from_disk("../../log/imitation/trajs_1")
    dataset_2 = datasets.load_from_disk("../../log/imitation/trajs_2")
    dataset_3 = datasets.load_from_disk("../../log/imitation/trajs_3")
    dataset_4 = datasets.load_from_disk("../../log/imitation/trajs_4")
    # dataset = datasets.concatenate_datasets([dataset_0])
    dataset = datasets.concatenate_datasets([dataset_0, dataset_1, dataset_2, dataset_3])
    transitions = huggingface_utils.TrajectoryDatasetSequence(dataset)
    del dataset, dataset_0, dataset_1, dataset_2, dataset_3, dataset_4
    # transitions = serialize.load(path="trajs_0")
    # transitions = custom_load(path="trajs_0")

    policy_kwargs = dict(
        features_extractor_class=Encoder,
        features_extractor_kwargs=dict(features_dim=256, num_images=5, resnet_output_dim=128),
        net_arch=dict(pi=[256, 256, 256], qf=[256, 256]),  # for AC policy
    )
    # model = SAC("CnnPolicy", env, verbose=1, buffer_size=10,
    #             policy_kwargs=policy_kwargs, device="cuda"
    #             )
    # model.load("../log/auv_student_data_10_epoch_100_0216_1358.zip")
    from sb3_launcher.algorithms import sqil
    sqil_trainer = sqil.SQIL(
        venv=env,
        demonstrations=transitions,
        policy="CnnPolicy",
        rl_algo_class=SAC,
        rl_kwargs=dict(verbose=1, buffer_size=20000, learning_rate=0.0003,
                    learning_starts=1000, batch_size=64,
                    train_freq=2, gradient_steps=1,
                    target_update_interval=10, tensorboard_log="../../log/imitation/sqil/",
                    policy_kwargs=policy_kwargs, device="cuda"),
    )
    del transitions
    print("build bc_trainer")

    print("Training a policy using Behavior Cloning")
    sqil_trainer.train(
        total_timesteps=1000000,
    )  # Note: set to 300_000 to obtain good results
    import datetime

    now = datetime.datetime.now().strftime("%m%d_%H%M")

    # 构建保存路径
    save_path = f"../../log/imitation/sqil/auv_student_{now}"
    sqil_trainer.policy.save(save_path)

    # print("Evaluating the trained policy.")
    # reward, _ = evaluate_policy(
    #    model.actor,  # type: ignore[arg-type]
    #    evaluation_env,
    #    n_eval_episodes=3,
    #    render=True,  # comment out to speed up
    # )
    # print(f"Reward after training: {reward}")
