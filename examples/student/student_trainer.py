#!/usr/bin/env python3

# import system lib
import argparse
import os
import pprint
import datetime

# import tool lib
import gymnasium as gym
from gymnasium import Env
import numpy as np
from numpy import linalg as LA
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from torch.distributions import Distribution, Independent, Normal

# import tianshou
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv, SubprocVectorEnv, ShmemVectorEnv, BaseVectorEnv
from tianshou.exploration import OUNoise
from tianshou.policy import SACPolicy, PPOPolicy
from tianshou.policy.base import BasePolicy
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net, ActorCritic
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.space_info import SpaceInfo

# import custom lib
from metadata import METADATA
import auv_env
from atrl_launcher.wrapper.teacher_obs_wrapper import TeachObsWrapper
from atrl_launcher.trainer import OffpolicyTrainer, OnpolicyTrainer
from atrl_launcher.networks.rgb_net import Encoder, Actor, Critic


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--choice', type=int, choices=[0, 1, 2, 3, 4], help='0:train; 1:keep train; 2:eval; 3:test',
                        default=0)
    # for env set
    parser.add_argument('--env', type=str, choices=['TargetTracking1', 'TargetTracking2', 'AUVTracking_rgb'],
                        help='environment ID', default='TargetTracking1')
    parser.add_argument('--policy', type=str, choices=['PPO', 'SAC', 'BC'], help='algorithm select',
                        default='SAC')
    parser.add_argument('--render', help='whether to render', type=int, default=0)
    parser.add_argument('--record', help='whether to record', type=int, default=0)
    parser.add_argument('--nb_targets', help='the number of targets', type=int, default=1)
    parser.add_argument('--nb_envs', help='the number of env', type=int, default=3)
    parser.add_argument('--max_episode_step', type=int, default=200)
    # for reinforcement learning set
    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--logdir", type=str, default="../log")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # # for SAC Policy Set
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--noise_std", type=float, default=1.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--auto_alpha", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.2)
    # # for PPO Policy Set
    parser.add_argument("--rew-norm", type=int, default=True)
    # In theory, `vf-coef` will not make any difference if using Adam optimizer.
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--bound-action-method", type=str, default="clip")
    parser.add_argument("--lr-decay", type=int, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=0)
    parser.add_argument("--recompute-adv", type=int, default=1)
    # # Collect Set
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    # # Trainer Set
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--step-per-epoch", type=int, default=12000)
    parser.add_argument("--step-per-collect", type=int, default=5)
    parser.add_argument("--update-per-step", type=float, default=0.2)
    parser.add_argument("--n-step", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test_episode", type=int, default=10)

    return parser.parse_args()


def make_teacher_env(
        task: str,
        num_train_envs: int,
        num_test_envs: int,
) -> tuple[Env, BaseVectorEnv, BaseVectorEnv]:
    if 'Teacher' not in task:
        raise ValueError("you should use teacher env.")
    train_envs = SubprocVectorEnv([lambda: gym.make(task) for _ in range(num_train_envs)], )
    test_envs = SubprocVectorEnv([lambda: gym.make(task) for _ in range(num_test_envs)], )
    env = gym.make(task)
    return env, train_envs, test_envs


def train_sac(args: argparse.Namespace = get_args()) -> None:
    # env
    env, train_envs, test_envs = make_teacher_env('Teacher-v0', args.nb_envs, 1)
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    # model
    encoder = Encoder()
    net = Net(state_shape=args.state_shape, hidden_sizes=[256, 256, 256], device=args.device)
    actor = ActorProb(net, args.action_shape, device=args.device, unbounded=True, preprocess_net_output_dim=256).to(
        args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=[256, 256],
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    net_c2 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=[256, 256],
        concat=True,
        device=args.device,
    )
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    action_dim = space_info.action_info.action_dim
    if args.auto_alpha:
        target_entropy = -action_dim
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy: SACPolicy = SACPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic1,
        critic_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        exploration_noise=OUNoise(0.0, args.noise_std),
        estimation_step=args.n_step,
        action_space=env.action_space,
    )

    if args.choice == 1 or args.choice == 2:
        # load a previous policy
        if args.resume_path:
            policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
            print("Loaded agent from: ", args.resume_path)
        else:
            raise ValueError("Resume path is not provided. "
                             "Please check your --choice and specify --resume_path to load the agent.")
    print('setup policy!!!')
    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs)
    # if args.start_timesteps != 0:
    #     train_collector.reset()
    #     train_collector.collect(n_step=args.start_timesteps, random=True)

    # log
    current_time = datetime.datetime.now().strftime('%m-%d_%H')
    log_name = os.path.join("teacher", "sac", current_time)
    log_path = os.path.join(args.logdir, log_name)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_epoch_fn(policy: BasePolicy, epoch: int) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, f"policy_{epoch}.pth"))

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    # trainer
    if args.choice == 0 or args.choice == 1:
        print('setup trainer!!!')
        result = OffpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_episode,
            batch_size=args.batch_size,
            update_per_step=args.update_per_step,
            # stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            save_epoch_fn=save_epoch_fn
        ).run()
        pprint.pprint(result)
    elif args.choice == 2:
        render_envs = TeachObsWrapper(auv_env.make(args.env,
                                                   render=1,
                                                   record=args.record,
                                                   num_targets=args.nb_targets,
                                                   is_training=False,
                                                   eval=True,
                                                   t_steps=args.max_episode_step,
                                                   ))
        render_collector = Collector(policy, render_envs)
        render_collector.reset()
        collector_stats = render_collector.collect(n_episode=10, render=0.0, reset_before_collect=True)
        print(collector_stats)


def train_ppo(args: argparse.Namespace = get_args()) -> None:
    # env
    env, train_envs, test_envs = make_teacher_env('Teacher-v0', args.nb_envs, 1)
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    # model
    net_a = Net(state_shape=args.state_shape, hidden_sizes=[256, 256, 256], device=args.device)
    actor = ActorProb(net_a, args.action_shape, device=args.device, unbounded=True).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=[256, 256],
        concat=True,
        device=args.device,
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)

    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)
    lr_scheduler = None
    if args.lr_decay:
        # decay learning rate to 0 linearly
        max_update_num = np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch

        lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    def dist(loc_scale: tuple[torch.Tensor, torch.Tensor]) -> Distribution:
        loc, scale = loc_scale
        return Independent(Normal(loc, scale), 1)

    policy: PPOPolicy = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        action_scaling=True,
        action_bound_method=args.bound_action_method,
        lr_scheduler=lr_scheduler,
        action_space=test_envs.action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    )

    if args.choice == 1 or args.choice == 2:
        # load a previous policy
        if args.resume_path:
            ckpt = torch.load(args.resume_path, map_location=args.device)
            policy.load_state_dict(ckpt["model"])
            train_envs.set_obs_rms(ckpt["obs_rms"])
            test_envs.set_obs_rms(ckpt["obs_rms"])
            print("Loaded agent from: ", args.resume_path)
        else:
            raise ValueError("Resume path is not provided. "
                             "Please check your --choice and specify --resume_path to load the agent.")

    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs)

    # log
    current_time = datetime.datetime.now().strftime('%m-%d_%H')
    log_name = os.path.join("teacher", "ppo", current_time)
    log_path = os.path.join(args.logdir, log_name)
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_epoch_fn(policy: BasePolicy, epoch: int) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, f"policy_{epoch}.pth"))

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    # trainer
    if args.choice == 0 or args.choice == 1:
        result = OnpolicyTrainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=args.epoch,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_episode,
            batch_size=args.batch_size,
            update_per_step=args.update_per_step,
            # stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
            save_epoch_fn=save_epoch_fn
        ).run()
        pprint.pprint(result)
    elif args.choice == 2:
        render_envs = TeachObsWrapper(auv_env.make(args.env,
                                                   render=1,
                                                   record=args.record,
                                                   num_targets=args.nb_targets,
                                                   is_training=False,
                                                   eval=True,
                                                   t_steps=args.max_episode_step,
                                                   ))
        render_collector = Collector(policy, render_envs)
        render_collector.reset()
        collector_stats = render_collector.collect(n_episode=10, render=0.0, reset_before_collect=True)
        print(collector_stats)


if __name__ == "__main__":
    args = get_args()
    if args.render:
        METADATA['render'] = True
    else:
        METADATA['render'] = False

    if args.policy == 'SAC':
        print('SAC')
        train_sac(args)
    elif args.policy == 'PPO':
        train_ppo(args)
