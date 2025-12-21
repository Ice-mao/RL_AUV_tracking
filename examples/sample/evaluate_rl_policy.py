"""
评估用于数据采集的 RL Policy
检查 RL policy 在环境中的实际表现
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import numpy as np
from datetime import datetime
from stable_baselines3 import SAC
import auv_env
from config_loader import load_config
from auv_env.wrappers import StateOnlyWrapper


def evaluate_rl_policy(model_path, env_config_path, num_episodes=10, show_viewport=False):
    """
    评估 RL policy 在环境中的表现

    Args:
        model_path: RL 模型路径
        env_config_path: 环境配置路径
        num_episodes: 评估 episode 数量
        show_viewport: 是否显示可视化
    """

    # 加载环境配置
    config = load_config(env_config_path)

    # 创建环境
    env = auv_env.make(
        config['name'],
        config=config,
        eval=False,
        t_steps=config.get('t_steps', 1000),
        show_viewport=show_viewport
    )

    print(f"Environment: {config['name']}")
    print(f"Config: {env_config_path}")

    # 加载模型
    print(f"\nLoading RL model: {model_path}")
    model = SAC.load(model_path, device='cuda', env=StateOnlyWrapper(env))
    print("✓ Model loaded successfully!")

    # 评估统计
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    collision_count = 0

    print(f"\nStarting evaluation for {num_episodes} episodes...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for episode in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'='*60}")

        episode_infos = []
        obs, info = env.reset()
        step = 0
        episode_reward = 0

        while True:
            # 使用 RL policy 预测动作
            action, _ = model.predict(obs['state'], deterministic=True)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            episode_reward += reward

            # 记录信息
            info['step'] = step
            info['episode'] = episode + 1
            info['reward'] = float(reward)
            info['cumulative_reward'] = float(episode_reward)
            info['action'] = action.tolist() if isinstance(action, np.ndarray) else action

            # 处理 numpy 数组
            processed_info = {}
            for key, value in info.items():
                if isinstance(value, np.ndarray):
                    processed_info[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    processed_info[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    processed_info[key] = int(value)
                else:
                    processed_info[key] = value

            episode_infos.append(processed_info)

            # 打印关键信息
            if step % 100 == 0:
                print(f"  Step {step}: reward={reward:.4f}, cumulative={episode_reward:.4f}")

            if terminated or truncated:
                break

        # Episode 结束统计
        episode_rewards.append(episode_reward)
        episode_lengths.append(step)

        # 检查是否成功（根据长度判断）
        if step >= 300:  # 与数据采集的 min_length 一致
            success_count += 1

        # 检查碰撞
        if any(info_step.get('is_collision', False) for info_step in episode_infos):
            collision_count += 1

        print(f"\nEpisode {episode + 1} Summary:")
        print(f"  Total Steps: {step}")
        print(f"  Total Reward: {episode_reward:.4f}")
        print(f"  Average Reward: {episode_reward/step:.4f}")
        print(f"  Status: {'SUCCESS' if step >= 300 else 'FAILED'}")

        # 保存详细信息
        save_dir = f"log/evaluation/RL_policy/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/rl_episode_{episode+1}_data.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(episode_infos, f, indent=2, ensure_ascii=False)

    env.close()

    # 打印总体统计
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total Episodes: {num_episodes}")
    print(f"Success Rate: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"Collision Rate: {collision_count}/{num_episodes} ({collision_count/num_episodes*100:.1f}%)")
    print(f"\nReward Statistics:")
    print(f"  Mean: {np.mean(episode_rewards):.4f}")
    print(f"  Std: {np.std(episode_rewards):.4f}")
    print(f"  Min: {np.min(episode_rewards):.4f}")
    print(f"  Max: {np.max(episode_rewards):.4f}")
    print(f"\nLength Statistics:")
    print(f"  Mean: {np.mean(episode_lengths):.1f}")
    print(f"  Std: {np.std(episode_lengths):.1f}")
    print(f"  Min: {np.min(episode_lengths)}")
    print(f"  Max: {np.max(episode_lengths)}")
    print(f"\nResults saved to: {save_dir}")

    # 保存统计信息
    summary = {
        'num_episodes': num_episodes,
        'success_rate': success_count / num_episodes,
        'collision_rate': collision_count / num_episodes,
        'rewards': {
            'mean': float(np.mean(episode_rewards)),
            'std': float(np.std(episode_rewards)),
            'min': float(np.min(episode_rewards)),
            'max': float(np.max(episode_rewards)),
            'all': [float(r) for r in episode_rewards]
        },
        'lengths': {
            'mean': float(np.mean(episode_lengths)),
            'std': float(np.std(episode_lengths)),
            'min': int(np.min(episode_lengths)),
            'max': int(np.max(episode_lengths)),
            'all': [int(l) for l in episode_lengths]
        }
    }

    summary_path = f"{save_dir}/summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Summary saved to: {summary_path}")

    return save_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate RL policy used for data collection')
    parser.add_argument('--model_path', type=str,
                        default='log/AUVTracking3D_v0/LQR/SAC/08-31_18/rl_model_1800000_steps.zip',
                        help='RL model path')
    parser.add_argument('--env_config', type=str,
                        default='configs/envs/3d_v1_config.yml',
                        help='Environment config path')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--show_viewport', action='store_true',
                        help='Show visualization')

    args = parser.parse_args()

    evaluate_rl_policy(
        model_path=args.model_path,
        env_config_path=args.env_config,
        num_episodes=args.num_episodes,
        show_viewport=args.show_viewport
    )
