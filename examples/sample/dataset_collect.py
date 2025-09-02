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
from stable_baselines3 import SAC
from auv_env.wrappers import StateOnlyWrapper

def collect_auv_data(collect, n_episodes=50):
    collector = collect
    
    config_path = os.path.join('configs', 'envs', '3d_v1_config.yml')
    config = load_config(config_path)
    
    env = auv_env.make("AUVTracking3D_v1", 
                       config=config,
                       eval=False,
                       t_steps=1000,
                       show_viewport=True,
                    )

    model_path = "log/AUVTracking3D_v0/LQR/SAC/08-31_18/rl_model_1800000_steps.zip"
    model = SAC.load(model_path, device='cuda', env=StateOnlyWrapper(env))
    
    print(f"开始采集 {n_episodes} 个episodes")
    for episode in range(n_episodes):
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        collector.start_episode()
        
        obs, info = env.reset()
        for step in range(1000):
            action = model.predict(obs['state'], deterministic=True)
            collector.add_step(obs, action[0])
            obs, reward, terminate, truncate, info = env.step(action[0])
            if terminate or truncate:
                break
        
        collector.finish_episode()
        if (episode + 1) % 10 == 0:
            collector.save_data(f"auv_data_partial_{episode + 1}.zarr")

    collector.save_data("auv_data_final.zarr")
    print("数据采集完成！")


if __name__ == '__main__':
    # collector = AUVCollector(save_dir="log/sample/simple", exist_replay_path="log/sample/auv_data/auv_data_final.zarr")
    collector = AUVCollector(save_dir="log/sample/simple", exist_replay_path=None)
    collect_auv_data(collector, n_episodes=100)  # 采集3个episodes作为测试
