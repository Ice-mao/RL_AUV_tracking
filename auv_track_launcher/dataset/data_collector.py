import os
import auv_env
from config_loader import load_config
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
import time


class AUVCollector:
    def __init__(self, save_dir="log/sample/auv_data", exist_replay_path=None):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.keys = ['camera_image', 'state', 'action']
        if exist_replay_path is not None:
            self.replay_buffer = ReplayBuffer.copy_from_path(exist_replay_path, keys=self.keys)
        else:
            self.replay_buffer = ReplayBuffer.create_empty_numpy()

    def start_episode(self):
        self.current_episode = {
            'obs': [],
            'action': []
        }

    def add_step(self, obs, action):
        self.current_episode['obs'].append(obs)
        self.current_episode['action'].append(action)

    def finish_episode(self):
        episode_data = {}
        
        # data processing
        episode_data['action'] = np.array(self.current_episode['action'], dtype=np.float32)
        
        camera_images = []
        sonar_images = []
        state = []

        for obs in self.current_episode['obs']:
            if 'images' in obs:
                camera_images.append(obs['images'])

            if 'sonar' in obs:
                sonar_images.append(obs['sonar'])
            
            if 'state' in obs:
                state.append(obs['state'])

        episode_data['camera_image'] = np.array(camera_images, dtype=np.uint8)
        # episode_data['sonar_data'] = np.array(sonar_images, dtype=np.float32)
        episode_data['state'] = np.array(state, dtype=np.float32)
        self.replay_buffer.add_episode(episode_data)
        print(f"Episode完成, 共{len(self.current_episode['obs'])}步")

    def save_data(self, filename=None):       
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"auv_data_{timestamp}.zarr"
        
        filepath = os.path.join(self.save_dir, filename)
        self.replay_buffer.save_to_path(filepath)
        
        print(f"数据已保存到: {filepath}")
        print(f"Episodes: {self.replay_buffer.n_episodes}")

