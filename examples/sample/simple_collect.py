import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import auv_env
from config_loader import load_config
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
import time
from auv_track_launcher.dataset.data_collector import AUVCollector

def collect_auv_data(collect, n_episodes=50):
    collector = collect
    
    config_path = os.path.join('configs', 'envs', 'v1_config.yml')
    config = load_config(config_path)
    
    env = auv_env.make("AUVTracking_v1", 
                       config=config,
                       eval=False,
                       t_steps=500,
                       show_viewport=True,
                    )
    
    print(f"开始采集 {n_episodes} 个episodes")

    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        collector.start_episode()
        
        obs, info = env.reset()
        for step in range(500):
            action = env.action_space.sample()
            collector.add_step(obs, action)
            obs, reward, terminate, truncate, info = env.step(action)
            if terminate or truncate:
                break
        
        collector.finish_episode()
        if (episode + 1) % 10 == 0:
            collector.save_data(f"auv_data_partial_{episode + 1}.zarr")

    collector.save_data("auv_data_final.zarr")
    print("数据采集完成！")


if __name__ == '__main__':
    # 简单调用
    collector = AUVCollector(exist_replay_path="/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/auv_data/auv_data_final.zarr")
    collect_auv_data(collector, n_episodes=3)  # 采集3个episodes作为测试
