import gymnasium as gym

import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3 import PPO, SAC
from stable_baselines3 import HerReplayBuffer
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

# from stable_baselines3.common.logger

import auv_env
import torch
import numpy as np
import argparse
from policy_net import SEED1, set_seed, CustomCNN

# tools
import os
import datetime
from metadata import METADATA

current_time = datetime.datetime.now()
time_string = current_time.strftime('%m-%d_%H')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--choice', choices=['0', '1', '2', '3'], help='0:train; 1:keep train; 2:eval; 3:test',
                    default=0)
parser.add_argument('--env', help='environment ID', type=str, default='TargetTracking')
parser.add_argument('--render', help='whether to render', type=int, default=0)
parser.add_argument('--record', help='whether to record', type=int, default=0)
parser.add_argument('--ros', help='whether to use ROS', type=int, default=0)
parser.add_argument('--map', help='choose your map in holoocean', type=str, default='TestMap')
parser.add_argument('--nb_targets', help='the number of targets', type=int, default=1)
parser.add_argument('--nb_envs', help='the number of env', type=int, default=6)
parser.add_argument('--log_dir', help='a path to a directory to log your data', type=str,
                    default='./models/dqn_cnn-' + time_string + '/')
# parser.add_argument('--map', type=str, default="obstacles02")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_episode_step', type=int, default=200)
args = parser.parse_args()

if args.render:
    METADATA['render'] = True
else:
    METADATA['render'] = False


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, save_path: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path_best = os.path.join(save_path, 'best_model')
        self.path_process = save_path
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path_best is not None:
            os.makedirs(self.save_path_best, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {}".format(self.num_timesteps))
                    print(
                        "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward,
                                                                                                 mean_reward))

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path_best))
                    self.model.save(self.save_path_best)

        # # save model every 100000 timesteps:
        # if self.num_timesteps % (20000) == 0:
        #     # Retrieve training reward
        #     path = self.path_process + str(self.num_timesteps) + '_model'
        #     self.model.save(path)

        return True


def main():
    if args.choice == '0' or args.choice == '1':
        # new training
        log_dir = '../log/ppo_' + time_string + '/'
        model_dir = '../models/ppo_' + time_string + '/'

        # keep training
        # model_dir = "../models/sac_04-01_18/"
        # log_dir = "../log/sac_04-01_18/"
        model_name = "120000_model"

        monitor_dir = log_dir
        os.makedirs(monitor_dir, exist_ok=True)
        # env = Monitor(env, monitor_dir)
        env = SubprocVecEnv([lambda: auv_env.make(args.env,
                                                  render=args.render,
                                                  record=args.record,
                                                  ros=args.ros,
                                                  directory=model_dir,
                                                  num_targets=args.nb_targets,
                                                  map=args.map,
                                                  is_training=True,
                                                  t_steps=args.max_episode_step
                                                  ) for _ in range(args.nb_envs)])
        env = VecMonitor(env, monitor_dir)
        set_seed(41)
        if args.choice == '0':
            learn(env, model_dir, log_dir)
        if args.choice == '1':
            keep_learn(env, model_dir, log_dir, model_name)

    elif args.choice == '2':
        model_dir = '/home/dell-t3660tow/Documents/RL/RL_AUV_tracking/models/sac_04-06_18/rl_model_720000_steps.zip'
        evaluate(model_dir)

    elif args.choice == '3':
        env_test()


def learn(env, model_dir, log_dir):
    # 获取当前时间
    os.makedirs(log_dir, exist_ok=True)
    callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir, save_path=model_dir)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(120000 // args.nb_envs, 1),
        save_path=model_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callback = CallbackList([callback, checkpoint_callback])
    # 网络架构选择
    # policy_kwargs = dict(net_arch=[256, 256, 256])  # 设置网络结构为3层256节点的感知机
    policy_kwargs = dict(net_arch=dict(pi=[256, 256, 256], qf=[256, 256]))  # set for off-policy network
    # policy_kwargs = dict(
    #     features_extractor_class=CustomCNN,
    #     features_extractor_kwargs=dict(features_dim=512),
    #     net_arch=[512, 512]
    #     # net_arch=[dict(pi=[512, 512], vf=[512, 512])],  # for AC policy
    #     # shared_lstm=True,  # use for RNNPPO
    #     # enable_critic_lstm=False  # use for RNNPPO
    # )  # 设置网络结构为自定义的网络架构（支持自定义输入）

    # 算法选择
    # DQN
    # model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.0001, buffer_size=10000,
    #             batch_size=64, target_update_interval=50, tensorboard_log=("./log/DQN_" + time_string), device="cuda",
    #             exploration_fraction=0.8, exploration_initial_eps=1.0, exploration_final_eps=0.4, seed=41)
    # model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.0001, buffer_size=10000,
    #             batch_size=64, target_update_interval=50, tensorboard_log=("./log/DQN_" + time_string), device="cuda",
    #             exploration_fraction=0.8, exploration_initial_eps=1.0, exploration_final_eps=0.4, seed=41)
    # model = DQN.load("./models/dqn_cnn-2023-12-02_18/final_model.zip", device='cuda', env=env)

    # PPO
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0001, batch_size=200, n_epochs=10,
                gae_lambda=0.9, clip_range=0.2, ent_coef=0.1, vf_coef=0.5, target_kl=0.02,
                policy_kwargs=policy_kwargs, tensorboard_log=log_dir, device="cuda")
    # model = SAC("MlpPolicy", env, verbose=1, learning_rate=0.0001, buffer_size=200000,
    #             learning_starts=100, batch_size=64, tau=0.005, gamma=0.99, train_freq=1,
    #             gradient_steps=1, action_noise=None,
    #             policy_kwargs=policy_kwargs, tensorboard_log=log_dir, device="cuda"
    #             )
    # model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.001, clip_range=0.1,
    #             clip_range_vf=0.1,
    #             batch_size=64, tensorboard_log=("./log/PPO_" + time_string), device="cuda")
    # model = PPO.load("./models/dqn_cnn-2023-12-02_18/final_model.zip", device='cuda', env=env)

    # model = RecurrentPPO("CnnLstmPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
    #                      learning_rate=0.001, clip_range=0.2, batch_size=64,
    #                      tensorboard_log=("./log/PPO_LSTM_" + time_string), device="cuda")

    model.learn(total_timesteps=1000000, tb_log_name="first_run", log_interval=5, callback=callback)
    model.save(args.log_dir + 'final_model')


def keep_learn(env, model_dir, log_dir, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    callback = SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=log_dir, save_path=model_dir)
    model = SAC.load(model_dir + model_name, device='cuda', env=env,
                     custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    model.learn(total_timesteps=1000000, tb_log_name="second_run", reset_num_timesteps=False,
                log_interval=5, callback=callback)
    model.save(model_dir + 'final_model')


def evaluate(model_dir):
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
    # get render parmater true
    # model = PPO.load(model_dir, device='cuda', env=env,
    #                  custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    model = SAC.load(model_dir, device='cuda', env=env,
                     custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    # model = DQN.load("./models/dqn_cnn-2023-12-01_14/final_model.zip", device='cuda')
    obs, _ = env.reset()
    for _ in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, inf = env.step(action)


def env_test():
    model_dir = '../models/test'
    env = auv_env.make(args.env,
                       render=args.render,
                       record=args.record,
                       ros=args.ros,
                       directory=model_dir,
                       num_targets=args.nb_targets,
                       map=args.map,
                       is_training=False,
                       t_steps=args.max_episode_step
                       )
    obs, _ = env.reset()
    while True:
        action = env.action_space.sample()
        print(action)
        # action = np.array([0.5, 1.0, 0.5])
        obs, reward, done, _, inf = env.step(action)


if __name__ == "__main__":
    main()
