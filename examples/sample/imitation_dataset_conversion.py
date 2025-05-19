if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import numpy as np
import zarr
import argparse
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import time
import math


left_camera_shape = (3, 224, 224)
right_camera_shape = (3, 224, 224)
sonar_shape = (1, 128, 128)
state_shape = (1)
def get_imitation_episode(replay_buffer, env, policy):
    obs, _ = env.reset()
    for i in range(200):
        action, _ = policy.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

    obs = transition.obs[:-1] # in imitation, obs is longer 1 step than actions
    assert obs.shape[0] == transition.acts.shape[0], "obs and acts shape mismatch"
    assert obs.shape[1] == math.prod(left_camera_shape) + math.prod(right_camera_shape) + \
        math.prod(sonar_shape) + state_shape, "obs self-mismatch"
    
    left_camera_img_size = math.prod(left_camera_shape)
    left_camera_img = obs[:, :left_camera_img_size].reshape(-1, *left_camera_shape)
    
    right_camera_img_size = math.prod(right_camera_shape)
    right_camera_img = obs[:, left_camera_img_size:left_camera_img_size + left_camera_img_size].reshape(-1, *right_camera_shape)

    sonar_size = math.prod(sonar_shape)
    sonar_offset = left_camera_img_size + right_camera_img_size
    sonar_img = obs[:, sonar_offset:sonar_offset+sonar_size].reshape(-1, *sonar_shape)

    state = obs[:, -1]

    return {
        'left_camera_img': left_camera_img,
        'right_camera_img': right_camera_img,
        'sonar_img': sonar_img,
        'state': state,
        'action': transition.acts
    }

def create_dataset(output_path):
    """创建包含多个episode的数据集并保存为zarr格式"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建ReplayBuffer
    replay_buffer = ReplayBuffer.create_empty_zarr()
    return replay_buffer


# for ep_idx in tqdm(range(len(raw_dataset)), desc="episodes traversal"):
#     # 生成一个轨迹
#     episode_data = get_imitation_episode(raw_dataset[ep_idx])
#     # 添加到replay buffer
#     replay_buffer.add_episode(
#         data=episode_data,
#         compressors='disk'  # 使用适合磁盘存储的压缩方式
#     )
    
# # 保存到指定路径
# replay_buffer.save_to_path(output_path)
# print(f"数据集已保存至: {output_path}")
# print(f"包含 {replay_buffer.n_episodes} 个episodes，共 {replay_buffer.n_steps} 步")

# # 返回统计信息
# return {
#     'n_episodes': replay_buffer.n_episodes,
#     'n_steps': replay_buffer.n_steps,
#     'episode_lengths': replay_buffer.episode_lengths
# }

def add_episode_existed_dataset(output_path,):
    """添加新的episode到已存在的数据集"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 创建ReplayBuffer
    replay_buffer = ReplayBuffer.copy_from_path(output_path, keys=['left_camera_img', 'right_camera_img', 'sonar_img', 'state', 'action'])
    print(f"原数据集包含 {replay_buffer.n_episodes} 个episodes，共 {replay_buffer.n_steps} 步")
    return replay_buffer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成模拟数据并保存为zarr格式")
    parser.add_argument('--mode', '-m', default='create', choices=['create', 'add'],
                        help='输出的zarr文件路径')
    args = parser.parse_args()
    output_path = "data/track/track_icemao_test_replay.zarr"
    # 创建数据集
    if args.mode == 'create':
        replay_buffer = create_dataset(output_path)
    elif args.mode == 'add':
        replay_buffer = add_episode_existed_dataset(output_path)
    else:
        assert False, "mode must be create or add"

    from stable_baselines3 import SAC
    import gymnasium as gym
    import auv_env
    from auv_env.wrappers.obs_wrapper import TeachObsWrapper
    from stable_baselines3.common.vec_env import SubprocVecEnv
    env = gym.make("v2-sample-render")
    env = SubprocVecEnv([lambda: gym.make("v2-sample-render") for _ in range(1)], )
    expert_env = TeachObsWrapper(env)
    policy = SAC.load("/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/teacher/SAC/04-23_12/rl_model_1500000_steps.zip",
                        device='cuda', env=expert_env,
                        custom_objects={'observation_space': expert_env.observation_space, 'action_space': expert_env.action_space})
    stats = get_imitation_episode(replay_buffer, env, policy)
    print(stats)