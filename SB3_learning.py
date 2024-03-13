import gymnasium as gym

import stable_baselines3
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3 import HerReplayBuffer
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy

import ttenv
import torch
import numpy as np
import argparse
from policy_net import SEED1, set_seed, CustomCNN

# tools
import os
import datetime

current_time = datetime.datetime.now()
time_string = current_time.strftime('%Y-%m-%d_%H')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--env', help='environment ID', type=str, default='TargetTracking-v5')
# TargetTracking-v1
parser.add_argument('--render', help='whether to render', type=int, default=1)
parser.add_argument('--record', help='whether to record', type=int, default=0)
parser.add_argument('--ros', help='whether to use ROS', type=int, default=0)
parser.add_argument('--nb_targets', help='the number of targets', type=int, default=1)
parser.add_argument('--log_dir', help='a path to a directory to log your data', type=str,
                    default='./models/dqn_cnn-' + time_string + '/')
# parser.add_argument('--map', type=str, default="obstacles02")
parser.add_argument('--map', type=str, default="dynamic_map")  # dynamic_map
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--im_size', type=int, default=28)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--max_episode_step', type=int, default=500)


# 自定义类型函数，将输入的字符串解析为矩阵
def matrix_type(matrix_string):
    matrix = np.array(eval(matrix_string))  # 将字符串解析为矩阵
    return matrix


parser.add_argument('--target_path', type=matrix_type, default=[[1, 2], [3, 4]])  # episode*T*4

args = parser.parse_args()


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

        # save model every 100000 timesteps:
        if self.n_calls % (20000) == 0:
            # Retrieve training reward
            path = self.path_process + str(self.n_calls) + '_model'
            self.model.save(path)

        return True


def main():
    env = ttenv.make(args.env,
                     render=args.render,
                     record=args.record,
                     ros=args.ros,
                     map_name=args.map,
                     directory=args.log_dir,
                     num_targets=args.nb_targets,
                     is_training=False,
                     im_size=args.im_size,
                     t_steps=args.max_episode_step
                     )
    monitor_dir = './models/monitor/'
    os.makedirs(monitor_dir, exist_ok=True)
    env = Monitor(env, monitor_dir)
    set_seed(41)
    # env = make_vec_env(env, n_envs=4)
    # env = Monitor(env, monitor_dir)

    # learn(env, monitor_dir)
    evaluate(env)
    # env_test(env)


def learn(env, monitor_dir):
    # 获取当前时间
    os.makedirs(args.log_dir, exist_ok=True)
    callback = SaveOnBestTrainingRewardCallback(check_freq=2000, log_dir=monitor_dir, save_path=args.log_dir)

    # policy_kwargs = dict(net_arch=[128, 128, 128])  # 设置网络结构为3层128节点的感知机
    # model = DQN("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.001, buffer_size=1000,
    #             batch_size=64, target_update_interval=50, tensorboard_log=("./log/DQN_" + time_string))
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[512, 512]
        # net_arch=[dict(pi=[512, 512], vf=[512, 512])],  # for AC policy
        # shared_lstm=True,  # use for RNNPPO
        # enable_critic_lstm=False  # use for RNNPPO
    )  # 设置网络结构为自定义的网络架构（支持自定义输入）

    # 算法选择
    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.0001, buffer_size=10000,
                batch_size=64, target_update_interval=50, tensorboard_log=("./log/DQN_" + time_string), device="cuda",
                exploration_fraction=0.8, exploration_initial_eps=1.0, exploration_final_eps=0.4, seed=41)
    # model = DQN.load("./models/dqn_cnn-2023-12-02_18/final_model.zip", device='cuda', env=env)

    # model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.001, clip_range=0.1,
    #             clip_range_vf=0.1,
    #             batch_size=64, tensorboard_log=("./log/PPO_" + time_string), device="cuda")
    # model = PPO.load("./models/dqn_cnn-2023-12-02_18/final_model.zip", device='cuda', env=env)

    # model = RecurrentPPO("CnnLstmPolicy", env, policy_kwargs=policy_kwargs, verbose=1,
    #                      learning_rate=0.001, clip_range=0.2, batch_size=64,
    #                      tensorboard_log=("./log/PPO_LSTM_" + time_string), device="cuda")

    model.learn(total_timesteps=2000000, log_interval=5, callback=callback)
    model.save(args.log_dir + 'final_model')


def evaluate(env):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN.load("./models/dqn_cnn-2023-12-13_14/final_model.zip", device='cuda', env=env,
                     custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    # model = DQN.load("./models/dqn_cnn-2023-12-01_14/final_model.zip", device='cuda')
    # model = RecurrentPPO.load("./models/dqn_cnn-2023-12-12_23/best_model.zip", device='cuda')
    obs, _ = env.reset()
    obs, _ = env.reset()
    obs, _ = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=False)
        obs, reward, done, _, inf = env.step(action)
        env.render()


def env_test(env):
    obs, _ = env.reset()
    while True:
        action = np.array(np.random.randint(0, 12))  # 生成 0 到 11 之间的整数随机数
        obs, reward, done, _, inf = env.step(action)
        env.render()


if __name__ == "__main__":
    main()
