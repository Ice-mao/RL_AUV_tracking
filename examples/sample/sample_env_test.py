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


if __name__ == '__main__':
    config_path = os.path.join('configs', 'envs', 'v1_sample_config.yml')
    config = load_config(config_path)
    
    env = StateOnlyWrapper(auv_env.make("AUVTracking_v1_sample", 
                       config=config,
                       eval=False,
                       t_steps=500,
                       show_viewport=True,
                    ))
    model_path = "log/AUVTracking_v0/PID/SAC/07-28_16/rl_model_2000000_steps.zip"
    model = SAC.load(model_path, device='cuda', env=env,
                         custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    
    obs, info = env.reset()
    for step in range(200):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminate, truncate, info = env.step(action)
        if terminate or truncate:
            break