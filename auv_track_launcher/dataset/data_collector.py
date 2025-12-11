import os
import numpy as np
from diffusion_policy.common.replay_buffer import ReplayBuffer
import time


class AUVCollector:
    def __init__(self, save_dir="log/sample/auv_data", exist_replay_path=None,
                 min_length=300, truncate_tail=100):
        """
        Args:
            save_dir: 保存目录
            exist_replay_path: 已有数据路径，用于继续采集
            min_length: episode最小长度，小于此值的episode将被舍弃
            truncate_tail: 截断尾部步数，有效episode会舍弃最后这么多步
        """
        self.save_dir = save_dir
        self.min_length = min_length
        self.truncate_tail = truncate_tail
        self.discarded_count = 0  # 统计被舍弃的episode数量
        self.valid_count = 0      # 统计有效的episode数量

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
        episode_length = len(self.current_episode['obs'])

        # 检查episode长度是否满足最小要求
        if episode_length < self.min_length:
            self.discarded_count += 1
            print(f"Episode 舍弃: 长度 {episode_length} < {self.min_length}")
            return False

        # 截断尾部
        valid_length = episode_length - self.truncate_tail
        obs_list = self.current_episode['obs'][:valid_length]
        action_list = self.current_episode['action'][:valid_length]

        episode_data = {}
        episode_data['action'] = np.array(action_list, dtype=np.float32)

        camera_images = []
        state = []

        for obs in obs_list:
            if 'image' in obs:
                camera_images.append(obs['image'])
            if 'state' in obs:
                state.append(obs['state'])

        episode_data['camera_image'] = np.array(camera_images, dtype=np.uint8)
        episode_data['state'] = np.array(state, dtype=np.float32)
        self.replay_buffer.add_episode(episode_data)
        self.valid_count += 1
        print(f"Episode 保存: 原始 {episode_length} 步, 截断后 {valid_length} 步")
        return True

    def save_data(self, filename=None):
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"auv_data_{timestamp}.zarr"

        filepath = os.path.join(self.save_dir, filename)
        self.replay_buffer.save_to_path(filepath)

        print("\n=== 数据保存完成 ===")
        print(f"保存路径: {filepath}")
        print(f"有效 episodes: {self.valid_count}")
        print(f"舍弃 episodes: {self.discarded_count}")
        print(f"总步数: {self.replay_buffer.n_steps}")