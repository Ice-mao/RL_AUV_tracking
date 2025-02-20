"""This is a simple example demonstrating how to clone the behavior of an expert.

Refer to the jupyter notebooks for more detailed examples of how to use the algorithms.
"""
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.utils import get_schedule_fn

# from imitation.algorithms import bc
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data import serialize

from sb3_launcher.algorithms.bc import BC

import sys
sys.path.append('/data/RL/RL_AUV_tracking/RL_AUV_tracking')

def sample_expert_transitions(expert):
    print("Sampling expert transitions.")
    rollouts = rollout.rollout(
        expert,
        env,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
        unwrap=False,
    )
    serialize.save(path="trajectories", trajectories=rollouts)
    return rollouts
    # return rollout.flatten_trajectories(rollouts)


def create_ppo_policy(env) -> ActorCriticPolicy:
    schedule = get_schedule_fn(0.0001)
    policy_kwargs = dict(
        # features_extractor_class=CustomCNN,
        # features_extractor_kwargs=dict(features_dim=256),
        # net_arch=[512, 512],
        net_arch=dict(pi=[512, 512, 512], vf=[512, 512]),
    )
    policy = ActorCriticPolicy(
        env.observation_space, env.action_space, lr_schedule=schedule, use_sde=False, **policy_kwargs
    )
    return policy


def create_sac_policy(env) -> SACPolicy:
    schedule = get_schedule_fn(0.0001)
    policy_kwargs = dict(
        # features_extractor_class=CustomCNN,
        # features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 64], qf=[64, 64]),  # for AC policy
    )
    policy = SACPolicy(
        env.observation_space, env.action_space, lr_schedule=schedule, use_sde=False, **policy_kwargs
    )
    return policy


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    env = make_vec_env(
        "BipedalWalker-v3",
        rng=rng,
        n_envs=4,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
    )
    evaluation_env = make_vec_env(
        "BipedalWalker-v3",
        rng=rng,
        env_make_kwargs={"render_mode": "rgb_array"},  # for rendering
    )
    ## expert policy
    # expert = PPO(
    #     policy=MlpPolicy,
    #     env=env,
    #     seed=0,
    #     batch_size=64,
    #     ent_coef=0.0,
    #     learning_rate=0.0003,
    #     n_epochs=10,
    #     n_steps=64,
    #     device="cuda"
    # )
    # expert.learn(5000000)  # Note: change this to 100_000 to train a decent expert.
    # expert.save('bipedal_expert')
    expert = load_policy("ppo", env, path="bipedal_expert.zip")

    ## get transitions
    # transitions = sample_expert_transitions(expert)
    transitions = serialize.load(path="trajectories")

    ## get policy
    # ppo_policy = create_ppo_policy(env)
    # sac_policy = create_sac_policy(env)
    # sac_policy.to("cuda")
    policy_kwargs = dict(
        # features_extractor_class=CustomCNN,
        # features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 64], qf=[64, 64]),  # for AC policy
    )
    model = SAC("MlpPolicy", env, verbose=1, learning_rate=0.0001, buffer_size=200000,
                learning_starts=100, batch_size=128, tau=0.005, gamma=0.99, train_freq=1,
                gradient_steps=1, action_noise=None,
                policy_kwargs=policy_kwargs, tensorboard_log="log", device="cuda"
                )
    print("bc")
    bc_trainer = BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
        policy=model.actor,
        device="cuda",
        optimizer_kwargs=dict(lr=0.0001),
        l2_weight=0.0001,
        ent_weight=0.0001
    )

    # print("Evaluating the expert policy.")
    # reward, _ = evaluate_policy(
    #     expert,  # type: ignore[arg-type]
    #     evaluation_env,
    #     n_eval_episodes=3,
    #     render=True,  # comment out to speed up
    # )
    # print(f"Reward expert: {reward}")
    #
    # print("Evaluating the untrained policy.")
    # reward, _ = evaluate_policy(
    #     bc_trainer.policy,  # type: ignore[arg-type]
    #     evaluation_env,
    #     n_eval_episodes=3,
    #     render=True,  # comment out to speed up
    # )
    # print(f"Reward before training: {reward}")

    print("Training a policy using Behavior Cloning")
    bc_trainer.train(n_epochs=100)

    model.actor = bc_trainer.policy
    model.save("bipedal_trained")

    print("Evaluating the trained policy.")
    reward, _ = evaluate_policy(
        bc_trainer.policy,  # type: ignore[arg-type]
        evaluation_env,
        n_eval_episodes=3,
        render=False,  # comment out to speed up
    )
    print(f"Reward after training: {reward}")
