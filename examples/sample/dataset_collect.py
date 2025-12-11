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
                       show_viewport=False,
                    )

    model_path = "log/AUVTracking3D_v0/LQR/SAC/08-31_18/rl_model_1800000_steps.zip"
    model = SAC.load(model_path, device='cuda', env=StateOnlyWrapper(env))
    
    print(f"开始采集 {n_episodes} 个episodes")
    num = 0
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
        
        flag = collector.finish_episode()
        num += 1 if flag else 0
        print(f"当前有效episode数量: {num}")
        if (num) % 50 == 0:
            collector.save_data(f"auv_data_partial_{num}.zarr")

    collector.save_data("auv_data_final.zarr")


if __name__ == '__main__':
    collector = AUVCollector(
        save_dir="log/sample/3d_auv_data",
        exist_replay_path=None,
        min_length=300,      # episode长度小于300则舍弃
        truncate_tail=100    # 舍弃最后100步(跟踪效果不好的部分)
    )
    collect_auv_data(collector, n_episodes=250)  # 采集250个，预期得到200+有效episode
