"""
BC Policy Evaluation Script
评估 Behavioral Cloning 策略在 AUV 跟踪环境中的性能
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import argparse
import auv_env
from config_loader import load_config
from examples.bc.bc_policy import BCPolicy


def evaluate_bc_and_save(model_path, env_config_path, num_episodes=10, device='cuda:0', show_viewport=False):
    """
    运行 BC 策略评估并保存每步的信息

    Args:
        model_path: BC 模型检查点路径
        env_config_path: 环境配置文件路径
        num_episodes: 评估的 episode 数量
        device: 推理设备
        show_viewport: 是否显示可视化窗口
    """

    # 1. 加载 BC 模型
    print(f"Loading BC model from: {model_path}")
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device)

    # 创建模型
    model = BCPolicy(
        action_dim=4,
        pretrained=True,
        freeze_backbone=False,
        hidden_dims=[256, 128],
        dropout=0.3
    ).to(device)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 加载 action stats (如果有的话，用于检查)
    action_stats = checkpoint.get('action_stats', None)
    if action_stats is not None:
        print(f"Action stats loaded:")
        print(f"  mean: {action_stats['mean']}")
        print(f"  std: {action_stats['std']}")

    print(f"✓ BC model loaded successfully! Device: {device}")

    # 2. 创建环境
    env_config = load_config(env_config_path)
    env = auv_env.make(
        env_config['name'],
        config=env_config,
        eval=False,
        t_steps=env_config.get('t_steps', 1000),
        show_viewport=show_viewport
    )

    print(f"✓ Environment created: {env_config['name']}")

    # 3. 运行评估
    print(f"\nStarting BC algorithm evaluation for {num_episodes} episodes...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for episode in range(num_episodes):
        print(f"\nBC Episode {episode + 1}/{num_episodes}")

        # 存储这个 episode 的所有 info
        episode_infos = []

        obs, info = env.reset()
        step = 0

        while True:
            # 提取图像观测
            if 'image' in obs:
                image = obs['image']  # (3, H, W) CHW format, uint8
            else:
                raise ValueError("No 'image' key in observation!")

            # BC 模型推理
            with torch.no_grad():
                action = model.get_action(image, device=device)

            # 执行动作
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            # 记录步信息
            info['step'] = step
            info['episode'] = episode + 1
            info['algorithm'] = 'BC'
            info['reward'] = float(reward)
            info['action'] = action.tolist() if isinstance(action, np.ndarray) else action

            # 处理 numpy 数组，转换为可 JSON 序列化的格式
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

            if terminated or truncated:
                break

        # Episode 结束，保存所有 info 到文件
        save_dir = f"log/benchmark/BC/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/bc_episode_{episode+1}_data.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(episode_infos, f, indent=2, ensure_ascii=False)

        print(f"  ✓ Saved {len(episode_infos)} steps to {filename}")

    env.close()
    print(f"\n✓ BC algorithm evaluation completed!")
    print(f"Results saved to: {save_dir}")

    return save_dir


def main():
    parser = argparse.ArgumentParser(description='Evaluate BC Policy performance')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to BC model checkpoint (.pth)')
    parser.add_argument('--env_config', type=str,
                        default='configs/envs/3d_v1_config.yml',
                        help='Environment configuration file')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for inference')
    parser.add_argument('--show_viewport', action='store_true',
                        help='Show visualization window')

    args = parser.parse_args()

    # 运行评估
    save_dir = evaluate_bc_and_save(
        model_path=args.model_path,
        env_config_path=args.env_config,
        num_episodes=args.num_episodes,
        device=args.device,
        show_viewport=args.show_viewport
    )

    # 提示如何分析结果
    print("\n" + "="*60)
    print("To analyze the results, run:")
    print(f"python examples/benchmark/benchmark_analyzer.py --data_dir {save_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
