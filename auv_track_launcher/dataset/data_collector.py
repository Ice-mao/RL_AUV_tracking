"""
AUV数据采集器

这个模块负责在仿真或实验过程中收集AUV跟踪数据
使用ReplayBuffer进行实时数据存储，然后保存为训练数据集
"""

import numpy as np
import time
from typing import Dict, Any, Optional
from pathlib import Path
# import zarr  # 注释掉，避免导入错误
from diffusion_policy.common.replay_buffer import ReplayBuffer


class AUVDataCollector:
    """
    AUV数据采集器
    
    在仿真运行过程中实时收集数据并存储到ReplayBuffer
    支持多episode收集和自动保存
    """
    
    def __init__(self, 
                 save_dir: str = "./collected_data",
                 auto_save_episodes: int = 10,  # 每收集多少episode自动保存一次
                 max_episode_length: int = 1000):
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.auto_save_episodes = auto_save_episodes
        self.max_episode_length = max_episode_length
        
        # 创建ReplayBuffer用于实时数据收集
        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        
        # 当前episode的数据缓存
        self.current_episode = {}
        self.episode_count = 0
        self.step_count = 0
        
        print(f"数据采集器初始化完成")
        print(f"保存目录: {self.save_dir}")
        print(f"自动保存间隔: {self.auto_save_episodes} episodes")

    def start_new_episode(self):
        """开始新的episode数据收集"""
        # 如果有未完成的episode，先保存它
        if self.current_episode:
            self._save_current_episode()
        
        # 初始化新episode
        self.current_episode = {
            'camera_image': [],
            'sonar_data': [],
            'auv_state': [],
            'target_state': [],
            'action': [],
            'reward': [],
            'done': []
        }
        self.step_count = 0
        print(f"开始新episode: {self.episode_count + 1}")

    def collect_step_data(self, 
                         obs: Dict[str, Any], 
                         action: np.ndarray,
                         reward: float = 0.0,
                         done: bool = False):
        """
        收集单步数据
        
        Args:
            obs: 观测数据字典，包含：
                - camera_image: (H, W, C) 相机图像
                - sonar_data: (n_beams,) 声纳数据  
                - auv_state: (12,) AUV状态 [pos, euler, vel, ang_vel]
                - target_state: (5,) 目标状态 [pos, rel_dist, rel_bearing]
            action: (6,) 控制动作 [thrust_xyz, torque_xyz]
            reward: 奖励值
            done: 是否结束
        """
        
        # 验证数据格式
        self._validate_step_data(obs, action)
        
        # 存储数据到当前episode
        self.current_episode['camera_image'].append(obs.get('camera_image', np.zeros((224, 224, 3), dtype=np.uint8)))
        self.current_episode['sonar_data'].append(obs.get('sonar_data', np.zeros(360, dtype=np.float32)))
        self.current_episode['auv_state'].append(obs['auv_state'].astype(np.float32))
        self.current_episode['target_state'].append(obs['target_state'].astype(np.float32))
        self.current_episode['action'].append(action.astype(np.float32))
        self.current_episode['reward'].append(float(reward))
        self.current_episode['done'].append(bool(done))
        
        self.step_count += 1
        
        # 检查episode是否结束
        if done or self.step_count >= self.max_episode_length:
            self._save_current_episode()
            
            # 检查是否需要自动保存到文件
            if (self.episode_count + 1) % self.auto_save_episodes == 0:
                self.save_to_file()

    def _validate_step_data(self, obs: Dict, action: np.ndarray):
        """验证单步数据格式"""
        required_keys = ['auv_state', 'target_state']
        for key in required_keys:
            if key not in obs:
                raise ValueError(f"观测数据缺少必需的键: {key}")
        
        if obs['auv_state'].shape != (12,):
            raise ValueError(f"auv_state应该是(12,)形状，实际: {obs['auv_state'].shape}")
        
        if obs['target_state'].shape != (5,):
            raise ValueError(f"target_state应该是(5,)形状，实际: {obs['target_state'].shape}")
        
        if action.shape != (6,):
            raise ValueError(f"action应该是(6,)形状，实际: {action.shape}")

    def _save_current_episode(self):
        """保存当前episode到ReplayBuffer"""
        if not self.current_episode or self.step_count == 0:
            return
        
        # 转换列表为numpy数组
        episode_data = {}
        for key, data_list in self.current_episode.items():
            episode_data[key] = np.array(data_list)
        
        # 添加到ReplayBuffer
        self.replay_buffer.add_episode(episode_data)
        
        self.episode_count += 1
        print(f"Episode {self.episode_count} 保存完成，共 {self.step_count} 步")
        
        # 清空当前episode缓存
        self.current_episode = {}

    def save_to_file(self, filename: Optional[str] = None):
        """
        保存ReplayBuffer到文件
        
        Args:
            filename: 文件名，如果为None则自动生成
        """
        if self.replay_buffer.n_episodes == 0:
            print("没有数据可保存")
            return
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"auv_data_{timestamp}_{self.replay_buffer.n_episodes}episodes.zarr"
        
        filepath = self.save_dir / filename
        
        # 保存为zarr格式（推荐，支持增量加载）
        if filepath.suffix == '.zarr':
            self.replay_buffer.save(str(filepath))
            print(f"数据已保存到zarr文件: {filepath}")
        
        # 也可以保存为npz格式（兼容性更好）
        elif filepath.suffix == '.npz':
            self._save_to_npz(str(filepath))
            print(f"数据已保存到npz文件: {filepath}")
        
        else:
            # 默认保存为zarr
            filepath = filepath.with_suffix('.zarr')
            self.replay_buffer.save(str(filepath))
            print(f"数据已保存到zarr文件: {filepath}")

    def _save_to_npz(self, filepath: str):
        """保存为NPZ格式"""
        episodes = []
        
        for episode_idx in range(self.replay_buffer.n_episodes):
            episode_data = {}
            start_idx = self.replay_buffer.episode_ends[episode_idx-1] if episode_idx > 0 else 0
            end_idx = self.replay_buffer.episode_ends[episode_idx]
            
            for key in self.replay_buffer.keys():
                episode_data[key.replace('_', '_')] = self.replay_buffer[key][start_idx:end_idx]
            
            episodes.append(episode_data)
        
        np.savez_compressed(filepath, episodes=episodes)

    def get_stats(self) -> Dict:
        """获取采集统计信息"""
        if self.replay_buffer.n_episodes == 0:
            return {"episodes": 0, "total_steps": 0}
        
        episode_lengths = []
        for i in range(self.replay_buffer.n_episodes):
            start_idx = self.replay_buffer.episode_ends[i-1] if i > 0 else 0
            end_idx = self.replay_buffer.episode_ends[i]
            episode_lengths.append(end_idx - start_idx)
        
        return {
            "episodes": self.replay_buffer.n_episodes,
            "total_steps": len(self.replay_buffer),
            "avg_episode_length": np.mean(episode_lengths),
            "min_episode_length": np.min(episode_lengths),
            "max_episode_length": np.max(episode_lengths),
        }

    def clear_buffer(self):
        """清空ReplayBuffer"""
        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        self.episode_count = 0
        print("ReplayBuffer已清空")


class AUVSimulationDataCollector(AUVDataCollector):
    """
    专门用于仿真环境的数据采集器
    提供与仿真环境的集成接口
    """
    
    def __init__(self, env, controller, **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.controller = controller

    def collect_episodes(self, n_episodes: int = 10):
        """
        自动收集多个episodes的数据
        
        Args:
            n_episodes: 要收集的episode数量
        """
        print(f"开始收集 {n_episodes} 个episodes")
        
        for episode_idx in range(n_episodes):
            print(f"\n=== Episode {episode_idx + 1}/{n_episodes} ===")
            
            # 开始新episode
            self.start_new_episode()
            
            # 重置环境
            obs = self.env.reset()
            done = False
            step = 0
            
            while not done and step < self.max_episode_length:
                # 控制器生成动作
                if hasattr(self.controller, 'get_action'):
                    action = self.controller.get_action(obs)
                else:
                    # 如果控制器是函数
                    action = self.controller(obs)
                
                # 执行动作
                next_obs, reward, done, info = self.env.step(action)
                
                # 收集数据
                self.collect_step_data(obs, action, reward, done)
                
                obs = next_obs
                step += 1
            
            print(f"Episode {episode_idx + 1} 完成，共 {step} 步")
        
        # 保存所有数据
        self.save_to_file()
        
        # 输出统计信息
        stats = self.get_stats()
        print(f"\n=== 数据收集完成 ===")
        for key, value in stats.items():
            print(f"{key}: {value}")


# 使用示例
def example_data_collection():
    """数据收集使用示例"""
    
    # 方式1: 手动逐步收集
    collector = AUVDataCollector(
        save_dir="./auv_collected_data",
        auto_save_episodes=5
    )
    
    # 模拟收集过程
    for episode in range(3):
        collector.start_new_episode()
        
        for step in range(100):
            # 模拟观测数据
            obs = {
                'camera_image': np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
                'sonar_data': np.random.uniform(0, 50, 360).astype(np.float32),
                'auv_state': np.random.randn(12).astype(np.float32),
                'target_state': np.random.randn(5).astype(np.float32)
            }
            
            # 模拟动作
            action = np.random.randn(6).astype(np.float32)
            
            # 模拟奖励和结束条件
            reward = np.random.random()
            done = (step == 99)  # 最后一步结束
            
            # 收集数据
            collector.collect_step_data(obs, action, reward, done)
    
    # 最终保存
    collector.save_to_file("manual_collection.zarr")
    
    print("手动数据收集示例完成")
    return collector


if __name__ == "__main__":
    # 运行示例
    collector = example_data_collection()
    
    # 查看统计信息
    stats = collector.get_stats()
    print("\n收集统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
