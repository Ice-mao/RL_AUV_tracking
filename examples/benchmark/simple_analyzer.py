"""
简化版分析工具 - 用于分析只有基础字段的benchmark数据
"""
import json
import numpy as np
import os
from pathlib import Path
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze_simple_benchmark(data_dir: str):
    """
    分析只有基础字段的benchmark数据
    
    Parameters:
    -----------
    data_dir : str
        包含JSON文件的目录路径
    """
    data_dir = Path(data_dir)
    json_files = list(data_dir.glob("*.json"))
    
    if not json_files:
        print(f"在目录 {data_dir} 中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    all_rewards = []
    all_steps = []
    all_mean_nlogdetcov = []
    episodes_data = []
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                episode_data = json.load(f)
            
            if not episode_data:
                continue
            
            # 提取数据
            episode_rewards = [step.get('reward', 0) for step in episode_data]
            episode_steps = len(episode_data)
            episode_mean_nlogdetcov = [step.get('mean_nlogdetcov', 0) for step in episode_data]
            
            all_rewards.extend(episode_rewards)
            all_steps.append(episode_steps)
            all_mean_nlogdetcov.extend(episode_mean_nlogdetcov)
            
            episodes_data.append({
                'file': json_file.name,
                'steps': episode_steps,
                'total_reward': sum(episode_rewards),
                'mean_reward': np.mean(episode_rewards),
                'mean_nlogdetcov': np.mean(episode_mean_nlogdetcov),
            })
            
        except Exception as e:
            print(f"加载文件 {json_file.name} 时出错: {e}")
    
    if not episodes_data:
        print("没有有效的数据可以分析")
        return
    
    # 生成报告
    print("\n" + "="*60)
    print("Benchmark 分析报告")
    print("="*60)
    print(f"分析目录: {data_dir}")
    print(f"Episode数量: {len(episodes_data)}")
    
    print(f"\n--- 基本统计 ---")
    print(f"总步数: {sum(all_steps)}")
    print(f"平均步数/Episode: {np.mean(all_steps):.1f} ± {np.std(all_steps):.1f}")
    print(f"最短Episode: {np.min(all_steps)} 步")
    print(f"最长Episode: {np.max(all_steps)} 步")
    
    print(f"\n--- Reward 统计 ---")
    print(f"平均Reward/步: {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")
    print(f"总Reward/Episode (平均): {np.mean([ep['total_reward'] for ep in episodes_data]):.2f} ± {np.std([ep['total_reward'] for ep in episodes_data]):.2f}")
    print(f"最高Reward/步: {np.max(all_rewards):.4f}")
    print(f"最低Reward/步: {np.min(all_rewards):.4f}")
    
    print(f"\n--- Mean Negative Log Determinant Covariance 统计 ---")
    print(f"平均 mean_nlogdetcov: {np.mean(all_mean_nlogdetcov):.4f} ± {np.std(all_mean_nlogdetcov):.4f}")
    print(f"最小值: {np.min(all_mean_nlogdetcov):.4f}")
    print(f"最大值: {np.max(all_mean_nlogdetcov):.4f}")
    
    # 成功率（假设达到最大步数的为成功）
    max_steps = max(all_steps) if all_steps else 1000
    success_threshold = int(max_steps * 0.8)  # 80%的步数
    successful_episodes = sum(1 for steps in all_steps if steps >= success_threshold)
    success_rate = successful_episodes / len(all_steps) if all_steps else 0
    print(f"\n--- 成功率 ---")
    print(f"成功率 (≥{success_threshold}步): {success_rate:.3f} ({success_rate*100:.1f}%)")
    print(f"成功Episode数: {successful_episodes}/{len(all_steps)}")
    
    # 保存报告
    report = {
        'summary': {
            'total_episodes': len(episodes_data),
            'total_steps': sum(all_steps),
            'avg_steps_per_episode': float(np.mean(all_steps)),
            'std_steps_per_episode': float(np.std(all_steps)),
        },
        'rewards': {
            'mean_per_step': float(np.mean(all_rewards)),
            'std_per_step': float(np.std(all_rewards)),
            'mean_per_episode': float(np.mean([ep['total_reward'] for ep in episodes_data])),
            'std_per_episode': float(np.std([ep['total_reward'] for ep in episodes_data])),
        },
        'mean_nlogdetcov': {
            'mean': float(np.mean(all_mean_nlogdetcov)),
            'std': float(np.std(all_mean_nlogdetcov)),
            'min': float(np.min(all_mean_nlogdetcov)),
            'max': float(np.max(all_mean_nlogdetcov)),
        },
        'success_rate': float(success_rate),
    }
    
    report_path = data_dir / "simple_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n报告已保存到: {report_path}")
    
    # 绘制图表
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Steps per episode
    axes[0, 0].bar(range(1, len(all_steps)+1), all_steps, alpha=0.7, color='blue')
    axes[0, 0].axhline(y=success_threshold, color='r', linestyle='--', label=f'Success threshold ({success_threshold})')
    axes[0, 0].set_title('Steps per Episode')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Steps')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Total reward per episode
    total_rewards = [ep['total_reward'] for ep in episodes_data]
    axes[0, 1].plot(range(1, len(total_rewards)+1), total_rewards, 'o-', color='green')
    axes[0, 1].set_title('Total Reward per Episode')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Total Reward')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Reward distribution
    axes[1, 0].hist(all_rewards, bins=50, alpha=0.7, color='orange')
    axes[1, 0].set_title('Reward Distribution')
    axes[1, 0].set_xlabel('Reward')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Mean nlogdetcov per episode
    mean_nlogdetcov_per_ep = [ep['mean_nlogdetcov'] for ep in episodes_data]
    axes[1, 1].plot(range(1, len(mean_nlogdetcov_per_ep)+1), mean_nlogdetcov_per_ep, 'o-', color='purple')
    axes[1, 1].set_title('Mean Negative Log Det Cov per Episode')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Mean nlogdetcov')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = data_dir / "simple_benchmark_metrics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {plot_path}")
    
    return report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='简化版benchmark分析工具')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='包含JSON文件的目录路径')
    
    args = parser.parse_args()
    analyze_simple_benchmark(args.data_dir)

