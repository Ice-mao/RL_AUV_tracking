"""
AUV跟踪数据预处理脚本

这个脚本展示了如何将原始的AUV仿真/实验数据转换为
适合diffusion policy训练的格式
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


def collect_auv_episode(simulation_env, controller, episode_length: int = 1000):
    """
    从仿真环境收集一个episode的数据
    
    Args:
        simulation_env: AUV仿真环境
        controller: 控制器（人工控制或其他策略）
        episode_length: episode最大长度
    
    Returns:
        dict: 包含完整episode数据的字典
    """
    episode_data = {
        'camera_images': [],
        'sonar_data': [],
        'auv_states': [],
        'target_states': [],
        'actions': [],
        'rewards': [],
        'dones': []
    }
    
    # 重置环境
    obs = simulation_env.reset()
    done = False
    step = 0
    
    while not done and step < episode_length:
        # 获取当前观测
        camera_image = obs.get('camera_image', np.zeros((224, 224, 3)))
        sonar_data = obs.get('sonar_data', np.zeros(360))  # 360度声纳
        auv_state = obs.get('auv_state', np.zeros(12))     # [pos, euler, vel, ang_vel]
        target_state = obs.get('target_state', np.zeros(5)) # [target_pos, rel_dist, rel_bearing]
        
        # 存储观测
        episode_data['camera_images'].append(camera_image)
        episode_data['sonar_data'].append(sonar_data)
        episode_data['auv_states'].append(auv_state)
        episode_data['target_states'].append(target_state)
        
        # 控制器生成动作
        action = controller.get_action(obs)
        episode_data['actions'].append(action)
        
        # 环境步进
        obs, reward, done, info = simulation_env.step(action)
        episode_data['rewards'].append(reward)
        episode_data['dones'].append(done)
        
        step += 1
    
    # 转换为numpy数组
    for key in episode_data:
        episode_data[key] = np.array(episode_data[key])
    
    return episode_data


def create_demonstration_data(n_episodes: int = 100, 
                            save_path: str = 'auv_demo_data.npz'):
    """
    创建演示数据集
    
    这里使用模拟数据，实际使用时你需要替换为真实的数据收集过程
    """
    print(f"正在生成 {n_episodes} 个演示episodes...")
    
    episodes = []
    
    for episode_idx in range(n_episodes):
        print(f"生成 episode {episode_idx + 1}/{n_episodes}")
        
        # 生成随机episode长度
        episode_length = np.random.randint(200, 800)
        
        # 模拟数据生成（实际使用时替换为真实数据收集）
        episode_data = generate_mock_episode(episode_length)
        episodes.append(episode_data)
    
    # 保存数据
    np.savez_compressed(save_path, episodes=episodes)
    print(f"数据已保存到: {save_path}")
    
    # 数据统计
    total_steps = sum(len(ep['actions']) for ep in episodes)
    print(f"总步数: {total_steps}")
    print(f"平均episode长度: {total_steps / n_episodes:.1f}")
    
    return save_path


def generate_mock_episode(episode_length: int) -> Dict:
    """
    生成模拟的episode数据（用于演示）
    实际使用时，你需要从真实的仿真或实验中收集数据
    """
    episode_data = {}
    
    # 生成时间序列
    t = np.linspace(0, episode_length * 0.1, episode_length)
    
    # 生成AUV轨迹（螺旋运动接近目标）
    radius = 10 * np.exp(-t / 50)  # 逐渐缩小的半径
    auv_x = radius * np.cos(t / 5)
    auv_y = radius * np.sin(t / 5)
    auv_z = -5 - t * 0.01  # 逐渐下降
    
    # 生成目标轨迹（缓慢移动）
    target_x = 2 * np.sin(t / 20)
    target_y = 2 * np.cos(t / 20)
    target_z = -6 * np.ones_like(t)
    
    # AUV状态：[x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    auv_states = np.zeros((episode_length, 12))
    auv_states[:, 0] = auv_x  # x
    auv_states[:, 1] = auv_y  # y
    auv_states[:, 2] = auv_z  # z
    auv_states[:, 5] = np.arctan2(np.gradient(auv_y), np.gradient(auv_x))  # yaw
    auv_states[:, 6] = np.gradient(auv_x) * 10  # vx
    auv_states[:, 7] = np.gradient(auv_y) * 10  # vy
    auv_states[:, 8] = np.gradient(auv_z) * 10  # vz
    
    # 目标状态：[target_x, target_y, target_z, relative_distance, relative_bearing]
    target_states = np.zeros((episode_length, 5))
    target_states[:, 0] = target_x
    target_states[:, 1] = target_y
    target_states[:, 2] = target_z
    
    # 计算相对距离和方位
    rel_x = target_x - auv_x
    rel_y = target_y - auv_y
    rel_z = target_z - auv_z
    target_states[:, 3] = np.sqrt(rel_x**2 + rel_y**2 + rel_z**2)  # 距离
    target_states[:, 4] = np.arctan2(rel_y, rel_x)  # 方位角
    
    # 生成控制动作：[thrust_x, thrust_y, thrust_z, torque_x, torque_y, torque_z]
    actions = np.zeros((episode_length, 6))
    
    # 简单的比例控制
    kp_pos = 0.5
    kp_yaw = 1.0
    
    for i in range(episode_length):
        # 位置控制
        pos_error = np.array([rel_x[i], rel_y[i], rel_z[i]])
        actions[i, :3] = kp_pos * pos_error
        
        # 朝向控制
        desired_yaw = np.arctan2(rel_y[i], rel_x[i])
        yaw_error = desired_yaw - auv_states[i, 5]
        actions[i, 5] = kp_yaw * yaw_error
    
    # 添加噪声
    actions += np.random.normal(0, 0.1, actions.shape)
    
    # 裁剪动作范围
    actions = np.clip(actions, -5, 5)
    
    # 生成相机图像（模拟）
    camera_images = np.random.randint(0, 256, (episode_length, 224, 224, 3), dtype=np.uint8)
    
    # 生成声纳数据（模拟）
    sonar_data = np.random.exponential(20, (episode_length, 360)).astype(np.float32)
    sonar_data = np.clip(sonar_data, 0, 50)  # 限制在50米范围内
    
    episode_data = {
        'camera_images': camera_images,
        'sonar_data': sonar_data,
        'auv_states': auv_states.astype(np.float32),
        'target_states': target_states.astype(np.float32),
        'actions': actions.astype(np.float32),
    }
    
    return episode_data


def visualize_episode_data(data_path: str, episode_idx: int = 0):
    """可视化episode数据"""
    data = np.load(data_path, allow_pickle=True)
    episodes = data['episodes']
    
    if episode_idx >= len(episodes):
        print(f"Episode {episode_idx} 不存在，总共有 {len(episodes)} 个episodes")
        return
    
    episode = episodes[episode_idx]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Episode {episode_idx} 数据可视化')
    
    # AUV轨迹
    auv_states = episode['auv_states']
    target_states = episode['target_states']
    
    axes[0, 0].plot(auv_states[:, 0], auv_states[:, 1], 'b-', label='AUV轨迹')
    axes[0, 0].plot(target_states[:, 0], target_states[:, 1], 'r-', label='目标轨迹')
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Y (m)')
    axes[0, 0].set_title('XY平面轨迹')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 相对距离
    axes[0, 1].plot(target_states[:, 3])
    axes[0, 1].set_xlabel('时间步')
    axes[0, 1].set_ylabel('距离 (m)')
    axes[0, 1].set_title('AUV与目标的相对距离')
    axes[0, 1].grid(True)
    
    # 动作序列
    actions = episode['actions']
    for i in range(min(3, actions.shape[1])):
        axes[0, 2].plot(actions[:, i], label=f'推力{i+1}')
    axes[0, 2].set_xlabel('时间步')
    axes[0, 2].set_ylabel('推力')
    axes[0, 2].set_title('控制动作')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    
    # AUV速度
    axes[1, 0].plot(auv_states[:, 6], label='Vx')
    axes[1, 0].plot(auv_states[:, 7], label='Vy')
    axes[1, 0].plot(auv_states[:, 8], label='Vz')
    axes[1, 0].set_xlabel('时间步')
    axes[1, 0].set_ylabel('速度 (m/s)')
    axes[1, 0].set_title('AUV速度')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # 声纳数据示例
    sonar_data = episode['sonar_data']
    im = axes[1, 1].imshow(sonar_data[:100].T, aspect='auto', cmap='viridis')
    axes[1, 1].set_xlabel('时间步')
    axes[1, 1].set_ylabel('声纳波束角度')
    axes[1, 1].set_title('声纳数据（前100步）')
    plt.colorbar(im, ax=axes[1, 1])
    
    # 相机图像示例
    if len(episode['camera_images']) > 0:
        first_image = episode['camera_images'][0]
        axes[1, 2].imshow(first_image)
        axes[1, 2].set_title('首帧相机图像')
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'episode_{episode_idx}_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"可视化已保存: episode_{episode_idx}_visualization.png")


def validate_dataset(data_path: str):
    """验证数据集格式"""
    try:
        data = np.load(data_path, allow_pickle=True)
        episodes = data['episodes']
        
        print(f"=== 数据集验证 ===")
        print(f"Episodes数量: {len(episodes)}")
        
        # 检查第一个episode的数据格式
        first_episode = episodes[0]
        print(f"\n第一个episode的数据结构:")
        for key, value in first_episode.items():
            print(f"  {key}: {value.shape} ({value.dtype})")
        
        # 统计信息
        episode_lengths = [len(ep['actions']) for ep in episodes]
        print(f"\nEpisode长度统计:")
        print(f"  最小长度: {min(episode_lengths)}")
        print(f"  最大长度: {max(episode_lengths)}")
        print(f"  平均长度: {np.mean(episode_lengths):.1f}")
        print(f"  总步数: {sum(episode_lengths)}")
        
        # 检查数据范围
        all_actions = np.vstack([ep['actions'] for ep in episodes])
        print(f"\n动作数据统计:")
        print(f"  形状: {all_actions.shape}")
        print(f"  范围: [{all_actions.min():.3f}, {all_actions.max():.3f}]")
        print(f"  均值: {all_actions.mean(axis=0)}")
        print(f"  标准差: {all_actions.std(axis=0)}")
        
        print("\n✅ 数据集格式验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 数据集验证失败: {e}")
        return False


if __name__ == "__main__":
    # 创建演示数据
    print("=== 创建AUV演示数据集 ===")
    data_path = create_demonstration_data(n_episodes=50, 
                                        save_path='auv_demo_data.npz')
    
    # 验证数据集
    print(f"\n=== 验证数据集 ===")
    if validate_dataset(data_path):
        # 可视化数据
        print(f"\n=== 可视化数据 ===")
        visualize_episode_data(data_path, episode_idx=0)
        
        print(f"\n✅ 数据集创建完成！")
        print(f"   数据路径: {data_path}")
        print(f"   可以用此数据测试 AUVTrackingDataset 类")
    else:
        print("❌ 数据集创建失败")
