# import system lib
import argparse
import os
import pprint
import datetime

# import tool lib
import gymnasium as gym
import numpy as np
from numpy import linalg as LA
import torch
from torch.utils.tensorboard import SummaryWriter

# import tianshou
from tianshou.data import Collector, CollectStats, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.exploration import OUNoise
from tianshou.policy import SACPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.space_info import SpaceInfo

# import custom lib
from metadata import METADATA
import auv_env
from policy_net import SEED1, set_seed, CustomCNN, PPO_withgridmap

# get the time
current_time = datetime.datetime.now()
time_string = current_time.strftime('%m-%d_%H')

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--choice', choices=['0', '1', '2', '3', '4'], help='0:train; 1:keep train; 2:eval; 3:test',
                        default=3)
    # for env parse in
    parser.add_argument('--env', choices=['TargetTracking1', 'TargetTracking2', 'AUVTracking_rgb'],
                        help='environment ID', default='TargetTracking1')
    parser.add_argument('--render', help='whether to render', type=int, default=0)
    parser.add_argument('--record', help='whether to record', type=int, default=0)
    parser.add_argument('--nb_envs', help='the number of env', type=int, default=3)
    parser.add_argument('--max_episode_step', type=int, default=200)

    parser.add_argument("--seed", type=int, default=1626)
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--noise_std", type=float, default=1.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--auto_alpha", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--step-per-epoch", type=int, default=12000)
    parser.add_argument("--step-per-collect", type=int, default=5)
    parser.add_argument("--update-per-step", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 128])
    parser.add_argument("--training-num", type=int, default=5)
    parser.add_argument("--test-num", type=int, default=10)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def train_sac(args: argparse.Namespace = get_args()) -> None:
    # env
    train_envs = DummyVectorEnv([lambda: auv_env.make(args.env,
                                                      render=args.render,
                                                      record=args.record,
                                                      ros=args.ros,
                                                      num_targets=args.nb_targets,
                                                      is_training=True,
                                                      eval=False,
                                                      t_steps=args.max_episode_step,
                                                      ) for _ in range(args.nb_envs)])

    test_envs = auv_env.make(args.env,
                             render=args.render,
                             record=args.record,
                             ros=args.ros,
                             num_targets=args.nb_targets,
                             is_training=False,
                             eval=False,
                             t_steps=args.max_episode_step,
                             )
    space_info = SpaceInfo.from_env(test_envs)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(net, args.action_shape, device=args.device, unbounded=True).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    net_c2 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
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
        action_space=test_envs.action_space,
    )
    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs)
    # train_collector.collect(n_step=args.buffer_size)
    # log
    log_path = os.path.join(args.logdir, args.task, "sac")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    # def stop_fn(mean_rewards: float) -> bool:
    #     if env.spec:
    #         if not env.spec.reward_threshold:
    #             return False
    #         else:
    #             return mean_rewards >= env.spec.reward_threshold
    #     return False

    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        # stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()

    # assert stop_fn(result.best_reward)
    if __name__ == "__main__":
        pprint.pprint(result)
        # Let's watch its performance!
        render_envs = gym.make(args.task, render_mode="human")
        render_collector = Collector(policy, render_envs)
        render_collector.reset()
        collector_stats = render_collector.collect(n_episode=10, render=0.0, reset_before_collect=True)
        print(collector_stats)

def train_sac_expert(args: argparse.Namespace = get_args()) -> None:
    # env
    train_envs = DummyVectorEnv([lambda: auv_env.make(args.env,
                                                      render=args.render,
                                                      record=args.record,
                                                      ros=args.ros,
                                                      num_targets=args.nb_targets,
                                                      is_training=True,
                                                      eval=False,
                                                      t_steps=args.max_episode_step,
                                                      ) for _ in range(args.nb_envs)])

    test_envs = auv_env.make(args.env,
                             render=args.render,
                             record=args.record,
                             ros=args.ros,
                             num_targets=args.nb_targets,
                             is_training=False,
                             t_steps=args.max_episode_step,
                             )
    space_info = SpaceInfo.from_env(test_envs)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(state_shape=args.state_shape, hidden_sizes=args.hidden_sizes, device=args.device)
    actor = ActorProb(net, args.action_shape, device=args.device, unbounded=True).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    net_c2 = Net(
        state_shape=args.state_shape,
        action_shape=args.action_shape,
        hidden_sizes=args.hidden_sizes,
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
        action_space=test_envs.action_space,
    )
    # collector
    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True,
    )
    test_collector = Collector(policy, test_envs)
    # train_collector.collect(n_step=args.buffer_size)
    # log
    log_path = os.path.join(args.logdir, args.task, "sac")
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    # def stop_fn(mean_rewards: float) -> bool:
    #     if env.spec:
    #         if not env.spec.reward_threshold:
    #             return False
    #         else:
    #             return mean_rewards >= env.spec.reward_threshold
    #     return False

    # trainer
    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        step_per_collect=args.step_per_collect,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        # stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger,
    ).run()

    # assert stop_fn(result.best_reward)
    if __name__ == "__main__":
        pprint.pprint(result)
        # Let's watch its performance!
        render_envs = gym.make(args.task, render_mode="human")
        render_collector = Collector(policy, render_envs)
        render_collector.reset()
        collector_stats = render_collector.collect(n_episode=10, render=0.0, reset_before_collect=True)
        print(collector_stats)
if __name__ == "__main__":
    if
        test_sac()
