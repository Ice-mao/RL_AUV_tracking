import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import argparse
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import auv_env
from config_loader import load_config

def main():
    # 加载配置文件
    config_path = os.path.join('configs', '3d_v0_config.yml')
    config = load_config(config_path)
    
    # 使用加载的配置创建环境
    # 其他参数如 num_targets, eval 等现在从 config 文件中读取
    env = auv_env.make("AUVTracking_v0", 
                       config=config,
                       eval=True, t_steps=200,
                       show_viewport=True,
                       )

    obs, info = env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminate, truncate, info = env.step(action)
        if terminate or truncate:
            obs, info = env.reset()
    env.close()

if __name__ == '__main__':
    main()