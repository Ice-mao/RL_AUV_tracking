import json
import numpy as np
import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

class BenchmarkAnalyzer:
    """基准分析器"""
    
    def __init__(self, data_dir: str):
        """
        初始化分析器
        
        Parameters:
        -----------
        data_dir : str
            包含JSON文件的目录路径
        """
        self.data_dir = Path(data_dir)
        self.episodes_data = []
        self.load_data()
    
    def load_data(self):
        """加载所有JSON文件"""
        json_files = list(self.data_dir.glob("*.json"))
        
        if not json_files:
            print(f"在目录 {self.data_dir} 中没有找到JSON文件")
            return
        
        print(f"找到 {len(json_files)} 个JSON文件")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    episode_data = json.load(f)
                    self.episodes_data.append({
                        'file': json_file.name,
                        'data': episode_data
                    })
                print(f"已加载: {json_file.name} ({len(episode_data)} 步)")
            except Exception as e:
                print(f"加载文件 {json_file.name} 时出错: {e}")
    
    def calculate_success_rate(self, success_threshold: float = 0.3) -> float:
        """
        计算成功率 (SR)
        成功定义：运行了总步长的40%以上

        Parameters:
        -----------
        success_threshold : float
            成功的步数阈值比例
            
        Returns:
        --------
        float : 成功率
        """
        if not self.episodes_data:
            return 0.0
        
        successful_episodes = 0
        
        for episode in self.episodes_data:
            data = episode['data']
            total_steps = len(data)
            
            # 检查是否有碰撞
            has_collision = any(step.get('is_collision', False) for step in data)
            
            # 检查是否达到步数阈值
            # 假设最大步数为1000步（可以从配置中获取）
            max_steps = 1000  
            achieved_threshold = total_steps >= (max_steps * success_threshold)
            
            # if not has_collision and achieved_threshold:
            if achieved_threshold:
                successful_episodes += 1
        
        success_rate = successful_episodes / len(self.episodes_data)
        return success_rate
    
    def calculate_mtbe(self) -> Tuple[float, List[float]]:
        """
        计算平均跟踪信念误差 (MTBE)
        
        Returns:
        --------
        Tuple[float, List[float]] : (总体MTBE, 每个episode的MTBE)
        """
        all_errors = []
        episode_errors = []
        
        for episode in self.episodes_data:
            data = episode['data']
            episode_error_list = []
            
            for step in data:
                try:
                    target_pos = np.array(step['targets'])
                    belief_pos = np.array(step['belief_targets'])
                    
                    # 计算欧几里得距离
                    error = np.linalg.norm(target_pos - belief_pos)
                    episode_error_list.append(error)
                    all_errors.append(error)
                    
                except (KeyError, ValueError) as e:
                    print(f"计算误差时出错: {e}")
                    continue
            
            if episode_error_list:
                episode_mtbe = np.mean(episode_error_list)
                episode_errors.append(episode_mtbe)
        
        overall_mtbe = np.mean(all_errors) if all_errors else 0.0
        return overall_mtbe, episode_errors
    
    def calculate_action_smoothness(self) -> Tuple[float, List[float]]:
        """
        计算动作平滑度 (AS)
        使用连续动作间的差异来衡量平滑度
        
        Returns:
        --------
        Tuple[float, List[float]] : (总体AS, 每个episode的AS)
        """
        all_smoothness = []
        episode_smoothness = []
        
        for episode in self.episodes_data:
            data = episode['data']
            action_differences = []
            
            for i in range(1, len(data)):
                try:
                    prev_action = np.array(data[i-1]['action'])
                    curr_action = np.array(data[i]['action'])
                    
                    # 计算动作差异的L2范数
                    action_diff = np.linalg.norm(curr_action - prev_action)
                    action_differences.append(action_diff)
                    all_smoothness.append(action_diff)
                    
                except (KeyError, ValueError) as e:
                    print(f"计算动作平滑度时出错: {e}")
                    continue
            
            if action_differences:
                # 动作平滑度 = 1 / (1 + 平均动作差异)
                # 值越高表示越平滑
                episode_as = 1.0 / (1.0 + np.mean(action_differences))
                episode_smoothness.append(episode_as)
        
        overall_as = 1.0 / (1.0 + np.mean(all_smoothness)) if all_smoothness else 0.0
        return overall_as, episode_smoothness
    
    def generate_report(self, save_path: str = None) -> Dict:
        """
        生成完整的基准分析报告
        
        Parameters:
        -----------
        save_path : str, optional
            报告保存路径
            
        Returns:
        --------
        Dict : 分析结果字典
        """
        print("=== 基准分析报告 ===")
        print(f"分析目录: {self.data_dir}")
        print(f"Episodes数量: {len(self.episodes_data)}")
        
        if not self.episodes_data:
            print("没有可分析的数据")
            return {}
        
        # 计算指标
        sr = self.calculate_success_rate()
        mtbe_overall, mtbe_episodes = self.calculate_mtbe()
        as_overall, as_episodes = self.calculate_action_smoothness()
        
        # 统计信息
        total_steps = sum(len(ep['data']) for ep in self.episodes_data)
        avg_steps_per_episode = total_steps / len(self.episodes_data)
        
        report = {
            'summary': {
                'total_episodes': len(self.episodes_data),
                'total_steps': total_steps,
                'avg_steps_per_episode': avg_steps_per_episode
            },
            'metrics': {
                'success_rate': sr,
                'mtbe_overall': mtbe_overall,
                'action_smoothness_overall': as_overall
            },
            'episode_details': {
                'mtbe_per_episode': mtbe_episodes,
                'as_per_episode': as_episodes
            }
        }
        
        # 打印报告
        print(f"\n--- 基本统计 ---")
        print(f"总Episodes: {report['summary']['total_episodes']}")
        print(f"总步数: {report['summary']['total_steps']}")
        print(f"平均步数/Episode: {report['summary']['avg_steps_per_episode']:.1f}")
        
        print(f"\n--- 关键指标 ---")
        print(f"Success Rate (SR): {sr:.3f} ({sr*100:.1f}%)")
        print(f"Mean Track Belief Error (MTBE): {mtbe_overall:.4f}")
        print(f"Action Smoothness (AS): {as_overall:.4f}")
        
        if len(mtbe_episodes) > 1:
            print(f"\n--- 每Episode详情 ---")
            print(f"MTBE 标准差: {np.std(mtbe_episodes):.4f}")
            print(f"AS 标准差: {np.std(as_episodes):.4f}")
            print(f"最好MTBE: {np.min(mtbe_episodes):.4f}")
            print(f"最差MTBE: {np.max(mtbe_episodes):.4f}")
        
        # 保存报告
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n报告已保存到: {save_path}")
        
        return report
    
    def plot_metrics(self, save_dir: str = None):
        """
        绘制指标可视化图表
        
        Parameters:
        -----------
        save_dir : str, optional
            图表保存目录
        """
        if not self.episodes_data:
            print("没有数据可绘制")
            return
        
        # 计算指标
        _, mtbe_episodes = self.calculate_mtbe()
        _, as_episodes = self.calculate_action_smoothness()
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. MTBE随episode变化
        axes[0, 0].plot(range(1, len(mtbe_episodes)+1), mtbe_episodes, 'b-o')
        axes[0, 0].set_title('Mean Track Belief Error per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('MTBE')
        axes[0, 0].grid(True)
        
        # 2. Action Smoothness随episode变化
        axes[0, 1].plot(range(1, len(as_episodes)+1), as_episodes, 'r-o')
        axes[0, 1].set_title('Action Smoothness per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Action Smoothness')
        axes[0, 1].grid(True)
        
        # 3. 误差分布直方图
        all_errors = []
        for episode in self.episodes_data:
            for step in episode['data']:
                try:
                    target_pos = np.array(step['targets'])
                    belief_pos = np.array(step['belief_targets'])
                    error = np.linalg.norm(target_pos - belief_pos)
                    all_errors.append(error)
                except:
                    continue
        
        axes[1, 0].hist(all_errors, bins=30, alpha=0.7, color='green')
        axes[1, 0].set_title('Belief Error Distribution')
        axes[1, 0].set_xlabel('Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # 4. 步数分布
        steps_per_episode = [len(ep['data']) for ep in self.episodes_data]
        axes[1, 1].bar(range(1, len(steps_per_episode)+1), steps_per_episode, alpha=0.7, color='orange')
        axes[1, 1].set_title('Steps per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir) / "benchmark_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        else:
            # 如果没有指定保存目录，保存到当前目录
            save_path = Path(self.data_dir) / "benchmark_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        # 不显示图表，因为可能在无图形界面环境中运行
        # plt.show()

def main():
    """主函数 - 使用示例"""
    import argparse
    
    parser = argparse.ArgumentParser(description='基准分析工具')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='包含JSON文件的目录路径')
    parser.add_argument('--save_report', type=str, default=None,
                       help='报告保存路径')
    parser.add_argument('--save_plots', type=str, default=None,
                       help='图表保存目录')
    
    args = parser.parse_args()
    
    # 创建分析器并运行分析
    analyzer = BenchmarkAnalyzer(args.data_dir)
    report = analyzer.generate_report(save_path=args.save_report)
    
    if args.save_plots:
        analyzer.plot_metrics(save_dir=args.save_plots)

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    current_dir = "log/benchmark/RL/dam/"
    # current_dir = "log/benchmark/greedy/dam/"
    analyzer = BenchmarkAnalyzer(current_dir)
    report = analyzer.generate_report()

    analyzer.plot_metrics(save_dir=current_dir)
