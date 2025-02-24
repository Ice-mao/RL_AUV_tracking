import datasets
from imitation.data import huggingface_utils

# Download some expert trajectories from the HuggingFace Datasets Hub.
dataset = datasets.load_dataset('hugging_data')

# Convert the dataset to a format usable by the imitation library.
expert_trajectories = huggingface_utils.TrajectoryDatasetSequence(dataset["train"])

from imitation.data import rollout

trajectory_stats = rollout.rollout_stats(expert_trajectories)

print(
    f"We have {trajectory_stats['n_traj']} trajectories. "
    f"The average length of each trajectory is {trajectory_stats['len_mean']}. "
    f"The average return of each trajectory is {trajectory_stats['return_mean']}."
)

from sb3_launcher.algorithms import sqil
from imitation.util.util import make_vec_env
import numpy as np
from stable_baselines3 import sac

SEED = 42

venv = make_vec_env(
    "Pendulum-v1",
    rng=np.random.default_rng(seed=SEED),
    # parallel=True,
)

sqil_trainer = sqil.SQIL(
    venv=venv,
    demonstrations=expert_trajectories,
    policy="MlpPolicy",
    rl_algo_class=sac.SAC,
    rl_kwargs=dict(verbose=1, buffer_size=100000, learning_rate=0.0003,
                   learning_starts=1000, batch_size=256,
                   train_freq=2, gradient_steps=1,
                   target_update_interval=10, tensorboard_log="../../log/imitation/sqil/",
                   device="cuda"),
)

from stable_baselines3.common.evaluation import evaluate_policy

reward_before_training, _ = evaluate_policy(sqil_trainer.policy, venv, 100)
print(f"Reward before training: {reward_before_training}")

sqil_trainer.train(
    total_timesteps=300000,
)  # Note: set to 300_000 to obtain good results
reward_after_training, _ = evaluate_policy(sqil_trainer.policy, venv, 100)
print(f"Reward after training: {reward_after_training}")
sqil_trainer.policy.save("sqil_pendulum_trained")