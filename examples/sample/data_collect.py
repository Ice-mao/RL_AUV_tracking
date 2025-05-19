if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import math
from tqdm import tqdm

# 图像数据的形状定义
left_camera_shape = (3, 224, 224)
right_camera_shape = (3, 224, 224)
sonar_shape = (1, 128, 128)
state_shape = (1)

def collect_dataset(env, policy, output_dir, max_steps=200):
    """
    Args:
        env: 环境
        policy: 策略
        output_dir: 输出目录
        max_steps: 最大步数
    """
    # 创建保存数据的目录
    left_camera_dir = os.path.join(output_dir, "left_camera")
    right_camera_dir = os.path.join(output_dir, "right_camera")
    sonar_dir = os.path.join(output_dir, "sonar")
    state_dir = os.path.join(output_dir, "state")
    action_dir = os.path.join(output_dir, "action")
    
    os.makedirs(left_camera_dir, exist_ok=True)
    os.makedirs(right_camera_dir, exist_ok=True)
    os.makedirs(sonar_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(action_dir, exist_ok=True)
    
    # 初始化环境
    obs, _ = env.reset()
    
    # 执行一个episode
    for step in tqdm(range(max_steps), desc="do the episode"):
        # 使用策略选择动作
        action, _ = policy.predict(obs['state'])
        obs = obs['images']
        # 解析观测数据
        left_camera_img_size = math.prod(left_camera_shape)
        left_camera_img = obs[:left_camera_img_size].reshape(*left_camera_shape)
        
        right_camera_img_size = math.prod(right_camera_shape)
        right_camera_img = obs[left_camera_img_size:left_camera_img_size + right_camera_img_size].reshape(*right_camera_shape)
        
        sonar_size = math.prod(sonar_shape)
        sonar_offset = left_camera_img_size + right_camera_img_size
        sonar_img = obs[sonar_offset:sonar_offset+sonar_size].reshape(*sonar_shape)
        
        state = obs[-1]
        
        # 保存为文件
        # 左相机图像保存为jpg
        left_img = left_camera_img.transpose(1, 2, 0).astype(np.uint8)  # 转换为HWC格式
        cv2.imwrite(os.path.join(left_camera_dir, f"step_{step:03d}.jpg"), 
                   cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
        
        # 右相机图像保存为jpg
        right_img = right_camera_img.transpose(1, 2, 0).astype(np.uint8)  # 转换为HWC格式
        cv2.imwrite(os.path.join(right_camera_dir, f"step_{step:03d}.jpg"), 
                   cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))
        
        # 声呐图像保存为png
        sonar_img_2d = sonar_img.squeeze(0).astype(np.uint8)
        cv2.imwrite(os.path.join(sonar_dir, f"step_{step:03d}.png"), sonar_img_2d)
        
        # state保存
        np.save(os.path.join(state_dir, f"step_{step:03d}.npy"), state)

        # action保存
        np.save(os.path.join(state_dir, f"step_{step:03d}.npy"), action)

        # 执行环境步骤
        obs, reward,_,  terminated, info = env.step(action)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="收集原始数据并保存为不同文件格式")
    parser.add_argument('--output', '-o', default="data/raw_collection、traj_0",
                        help='输出目录路径')
    parser.add_argument('--max_steps', '-e', type=int, default=200,
                        help='最大步数')
    parser.add_argument('--env', default="v2-sample-render",
                        help='环境名称')
    args = parser.parse_args()
    
    output_path = args.output
    
    # 导入必要的库
    from stable_baselines3 import SAC
    import gymnasium as gym
    import auv_env
    from auv_env.wrappers.obs_wrapper import TeachObsWrapper
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
    
    # 创建环境
    print(f"创建环境: {args.env}")
    env = auv_env.make(env_name='AUVTracking_v2_sample',
                        record=False,
                        show_viewport=True,
                        num_targets=1,
                        is_training=False,
                        eval=False,
                        t_steps=200,
                        map="AUV_RGB_Dam_sonar"
                        )
    # env = gym.make(args.env)
    # expert_env = TeachObsWrapper(DummyVecEnv([lambda: gym.make(args.env) for _ in range(1)]))
    
    # 加载策略
    print("加载专家策略")
    policy = SAC.load("/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/teacher/SAC/04-23_12/rl_model_1500000_steps.zip",
                     device='cpu')
    
    # 收集数据
    os.makedirs(output_path, exist_ok=True)
    collect_dataset(env, policy, output_path, max_steps=args.max_steps)
