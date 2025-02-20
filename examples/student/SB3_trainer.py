import gymnasium as gym

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecEnv
# from stable_baselines3.common.logger

import auv_env
import torch
import numpy as np
from numpy import linalg as LA
import csv
import argparse
from policy_net import set_seed
from sb3_launcher.common.callbacks import SaveOnBestTrainingRewardCallback
from sb3_launcher.networks import Encoder

# tools
import os
import datetime
from metadata import METADATA

current_time = datetime.datetime.now()
time_string = current_time.strftime('%m-%d_%H')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--choice', choices=['0', '1', '2', '3', '4'], help='0:train; 1:keep train; 2:eval; 3:test; 4:cal',
                    default=0)
# for env set
parser.add_argument('--env', type=str, choices=['TargetTracking1', 'TargetTracking2', 'AUVTracking_rgb'],
                    help='environment ID', default='TargetTracking1')
parser.add_argument('--policy', type=str, choices=['PPO', 'SAC', 'BC'], help='algorithms select',
                    default='SAC')
parser.add_argument('--render', help='whether to render', type=int, default=0)
parser.add_argument('--record', help='whether to record', type=int, default=0)
parser.add_argument('--nb_targets', help='the number of targets', type=int, default=1)
parser.add_argument('--nb_envs', help='the number of env', type=int, default=6)
parser.add_argument('--max_episode_step', type=int, default=200)
# for reinforcement learning set
parser.add_argument("--seed", type=int, default=1626)
parser.add_argument("--resume-path-model", type=str, default=None)
parser.add_argument("--log-dir", type=str, default="../log")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
# # for SAC Policy Set
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--alpha-lr", type=float, default=3e-4)
parser.add_argument("--noise_std", type=float, default=1.2)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--tau", type=float, default=0.005)
parser.add_argument("--auto_alpha", type=int, default=1)
parser.add_argument("--alpha", type=float, default=0.2)
# # for PPO Policy Set
parser.add_argument("--n-steps", type=int, default=128)
parser.add_argument("--vf-coef", type=float, default=0.25)
parser.add_argument("--ent-coef", type=float, default=0.0)
parser.add_argument("--gae-lambda", type=float, default=0.95)
parser.add_argument("--max-grad-norm", type=float, default=0.5)
parser.add_argument("--eps-clip", type=float, default=0.2)
parser.add_argument("--value-clip", type=float, default=0)
parser.add_argument("--norm-adv", type=int, default=0)
# # Collect Set
parser.add_argument("--buffer-size", type=int, default=50000)
parser.add_argument("--start-timesteps", type=int, default=10000)
# # Trainer Set
parser.add_argument("--timesteps", type=int, default=100000)
parser.add_argument("--step-per-collect", type=int, default=5)
parser.add_argument("--update-per-step", type=float, default=0.2)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--test_episode", type=int, default=10)
args = parser.parse_args()

if args.render:
    METADATA['render'] = True
else:
    METADATA['render'] = False


def make_student_env(
        task: str,
        num_train_envs: int,
        monitor_dir: str
) -> VecEnv:
    if 'Student' not in task:
        raise ValueError("you should use student env.")
    train_envs = SubprocVecEnv([lambda: gym.make(task) for _ in range(num_train_envs)], )
    env = VecMonitor(train_envs, monitor_dir)
    return env


def make_callback(
        log_dir: str,
) -> CallbackList:
    callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir, save_path=log_dir)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(20000 // args.nb_envs, 1),
        save_path=log_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callback = CallbackList([callback, checkpoint_callback])
    return callback


def learn(env, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    # callback
    callback = make_callback(log_dir)

    if args.policy == 'SAC':
        policy_kwargs = dict(
            features_extractor_class=Encoder,
            features_extractor_kwargs=dict(features_dim=512, num_images=5, resnet_output_dim=128),
            net_arch=dict(pi=[512, 512, 512], qf=[512, 512]),  # for AC policy
        )
        model = SAC("CnnPolicy", env, verbose=1, learning_rate=args.lr, buffer_size=args.buffer_size,
                    learning_starts=args.start_timesteps, batch_size=args.batch_size, tau=args.tau, gamma=args.gamma,
                    train_freq=2, gradient_steps=1,
                    action_noise=NormalActionNoise(np.array([0.0, 0.0, 0.0]), np.array([0.05, 0.05, 0.03])),
                    target_update_interval=10,
                    policy_kwargs=policy_kwargs, tensorboard_log=log_dir, device=args.device
                    )
        model.learn(total_timesteps=args.timesteps, tb_log_name="first_run", log_interval=5, callback=callback)
        model.save(args.log_dir + 'final_model')
    elif args.policy == 'PPO':
        # policy_kwargs = dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256]))  # set for off-policy network
        policy_kwargs = dict(
            features_extractor_class=Encoder,
            features_extractor_kwargs=dict(features_dim=512, num_images=5, resnet_output_dim=128),
            net_arch=dict(pi=[256, 256, 128], vf=[512, 512]),
        )
        model = PPO("CnnPolicy", env, verbose=1, learning_rate=args.lr, batch_size=args.batch_size,
                    n_epochs=10, n_steps=args.n_steps,
                    gae_lambda=args.gae_lambda, clip_range=args.eps_clip, clip_range_vf=args.value_clip,
                    ent_coef=args.ent_coef, vf_coef=args.vf_coef,
                    max_grad_norm=args.max_grad_norm, normalize_advantage=bool(args.norm_adv),
                    policy_kwargs=policy_kwargs, tensorboard_log=log_dir, device=args.device
                    )
        model.learn(total_timesteps=args.timesteps, tb_log_name="first_run", log_interval=1, callback=callback)
        model.save(args.log_dir + 'final_model')


def keep_learn(env, log_dir, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(log_dir, exist_ok=True)
    callback = make_callback(log_dir)

    if args.policy == 'SAC':
        model = SAC.load(model_name, device='cuda', env=env,
                         custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    elif args.policy == 'PPO':
        model = PPO.load(model_name, device='cuda', env=env,
                         custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})

    model.learn(total_timesteps=args.timesteps, tb_log_name="second_run", reset_num_timesteps=False,
                log_interval=1, callback=callback)
    model.save(log_dir + 'final_model')


def evaluate(model_name: str):
    """
    2
    :param model_name:
    :return:
    """
    from metadata import TTENV_EVAL_SET
    # 0 tracking 1 discovery 2 navagation
    METADATA.update(TTENV_EVAL_SET[0])

    env = SubprocVecEnv([lambda: gym.make('Teacher-v0-render') for _ in range(1)], )

    if args.policy == 'SAC':
        model = SAC.load(model_name, device='cuda', env=env,
                         custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    elif args.policy == 'PPO':
        model = PPO.load(model_name, device='cuda', env=env,
                         custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})

    obs = env.reset()
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        print(action)
        obs, reward, dones, inf = env.step(action)


def eval_greedy(model_dir):
    """
    4
    :param model_dir:
    :return:
    """
    from metadata import TTENV_EVAL_SET
    # 0 tracking 1 discovery 2 navagation
    METADATA.update(TTENV_EVAL_SET[0])
    env = auv_env.make(args.env,
                       render=args.render,
                       record=args.record,
                       ros=args.ros,
                       directory=model_dir,
                       num_targets=args.nb_targets,
                       map=args.map,
                       eval=True,
                       is_training=False,
                       t_steps=args.max_episode_step
                       )
    from auv_baseline.greedy import Greedy
    greedy = Greedy(env.env.env)

    for i in range(10):
        # init the eval data
        prior_data = []  # sigma t+1
        posterior_data = []  # sigma t+1|t
        observed = []
        is_col = []

        obs, _ = env.reset()
        for _ in range(200):
            action = greedy.predict(obs)
            obs, reward, done, _, inf = env.step(action)

            prior_data.append(-np.log(LA.det(env.env.env.world.belief_targets[0].cov)))
            posterior_data.append(-np.log(LA.det(env.env.env.world.record_cov_posterior[0])))
            observed.append(env.env.env.world.record_observed[0])
            is_col.append(env.env.env.world.is_col)

        # 对于列表
        with open('../data_record/greedy_comparion_l/greedy_500_l_' + str(METADATA['lqr_l_p']) + '_' + str(
                i + 1) + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            # 遍历列表并写入数据
            for j in range(len(prior_data)):
                writer.writerow([prior_data[j], posterior_data[j], observed[j]])


def env_test():
    "3"
    model_dir = '../models/test'
    env = auv_env.make(args.env,
                       render=args.render,
                       record=args.record,
                       ros=args.ros,
                       directory=model_dir,
                       num_targets=args.nb_targets,
                       map=args.map,
                       eval=True,
                       is_training=False,
                       t_steps=args.max_episode_step
                       )
    obs, _ = env.reset()
    while True:
        action = env.action_space.sample()
        action = np.array([1.0, 0.0, 0.5])
        print(action)
        obs, reward, done, _, inf = env.step(action)


if __name__ == "__main__":
    if args.choice == '0' or args.choice == '1':
        set_seed(args.seed)
        if args.choice == '0':
            log_dir = os.path.join(args.log_dir, args.policy, time_string)
            env = make_student_env('Student-v0-norender', args.nb_envs, log_dir)
            args.state_space = env.observation_space
            args.action_space = env.action_space
            learn(env, log_dir)
        if args.choice == '1':
            model_name = args.resume_path_model
            log_dir = os.path.dirname(model_name)
            env = make_student_env('Student-v0-norender', args.nb_envs, log_dir)
            keep_learn(env, log_dir, model_name)

    elif args.choice == '2':
        model_name = args.resume_path_model
        evaluate(model_name)

    elif args.choice == '3':
        env_test()

    elif args.choice == '4':
        model_dir = ''
        eval_greedy(model_dir)
