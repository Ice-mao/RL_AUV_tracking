import json
import numpy as np
import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

class BenchmarkAnalyzer: 
    def __init__(self, data_dir: str, min_steps: int = 0):
        """      
        Parameters:
        -----------
        data_dir : str
        min_steps : int, optional
            Minimum number of steps required for an episode to be included in analysis
        """
        self.data_dir = Path(data_dir)
        self.min_steps = min_steps
        self.episodes_data = []
        self.load_data()
    
    def load_data(self):
        json_files = list(self.data_dir.glob("*.json"))
        
        # Filter out report.json and other non-episode files
        json_files = [f for f in json_files if 'episode' in f.name.lower() or 'rl_' in f.name.lower()]
        
        if not json_files:
            print(f"No episode JSON files found in directory {self.data_dir}")
            return
        
        print(f"Found {len(json_files)} episode JSON files")
        if self.min_steps > 0:
            print(f"Filtering: only episodes with >= {self.min_steps} steps will be included")
        
        loaded_count = 0
        filtered_count = 0
        
        for json_file in sorted(json_files):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    episode_data = json.load(f)
                    
                    # Filter by minimum steps
                    if len(episode_data) < self.min_steps:
                        filtered_count += 1
                        continue
                    
                    self.episodes_data.append({
                        'file': json_file.name,
                        'data': episode_data
                    })
                    loaded_count += 1
                    print(f"Loaded: {json_file.name} ({len(episode_data)} steps)")
            except Exception as e:
                print(f"Error loading file {json_file.name}: {e}")
        
        if self.min_steps > 0:
            print(f"\nFiltering summary: {loaded_count} episodes loaded, {filtered_count} episodes filtered out (< {self.min_steps} steps)")
    
    def calculate_success_rate(self, success_threshold: float = 0.3) -> float:
        """
        Parameters:
        -----------
        success_threshold : float
            Success threshold ratio for number of steps
            
        Returns:
        --------
        float : sucess rate
        """
        if not self.episodes_data:
            return 0.0
        
        successful_episodes = 0
        
        for episode in self.episodes_data:
            data = episode['data']
            total_steps = len(data)
            
            # Check for collisions
            has_collision = any(step.get('is_collision', False) for step in data)
            
            # Check if step threshold is reached
            # Assume maximum steps is 1000 (can be obtained from config)
            max_steps = 1000  
            achieved_threshold = total_steps >= (max_steps * success_threshold)
            
            # if not has_collision and achieved_threshold:
            if achieved_threshold:
                successful_episodes += 1
        
        success_rate = successful_episodes / len(self.episodes_data)
        return success_rate
    
    def calculate_mtbe(self) -> Tuple[float, List[float]]:
        """
        Calculate Mean Tracking Belief Error (MTBE)
        
        Returns:
        --------
        Tuple[float, List[float]] : (Overall MTBE, MTBE per episode)
        """
        all_errors = []
        episode_errors = []
        
        for episode in self.episodes_data:
            data = episode['data']
            episode_error_list = []
            
            for step in data:
                try:
                    target_pos = np.array(step['targets'])
                    belief_raw = step['belief_targets']
                    
                    # Handle belief_targets that may contain lists (e.g., fix_depth as [5, 15])
                    belief_pos = []
                    for i, val in enumerate(belief_raw):
                        if isinstance(val, (list, tuple, np.ndarray)):
                            # If it's a list/array, take the first element or mean
                            if len(val) > 0:
                                belief_pos.append(float(val[0]) if isinstance(val[0], (int, float)) else float(np.mean(val)))
                            else:
                                belief_pos.append(0.0)
                        else:
                            belief_pos.append(float(val))
                    belief_pos = np.array(belief_pos)
                    
                    # Ensure same length
                    min_len = min(len(target_pos), len(belief_pos))
                    target_pos = target_pos[:min_len]
                    belief_pos = belief_pos[:min_len]
                    
                    # Calculate Euclidean distance
                    error = np.linalg.norm(target_pos - belief_pos)
                    episode_error_list.append(error)
                    all_errors.append(error)
                    
                except (KeyError, ValueError, TypeError) as e:
                    # Silently skip errors to avoid flooding output
                    continue
            
            if episode_error_list:
                episode_mtbe = np.mean(episode_error_list)
                episode_errors.append(episode_mtbe)
        
        overall_mtbe = np.mean(all_errors) if all_errors else 0.0
        return overall_mtbe, episode_errors
    
    def calculate_action_smoothness(self) -> Tuple[float, List[float]]:
        """
        Calculate Action Smoothness (AS)
        Use differences between consecutive actions to measure smoothness
        
        Returns:
        --------
        Tuple[float, List[float]] : (Overall AS, AS per episode)
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
                    
                    # Calculate L2 norm of action difference
                    action_diff = np.linalg.norm(curr_action - prev_action)
                    action_differences.append(action_diff)
                    all_smoothness.append(action_diff)
                    
                except (KeyError, ValueError) as e:
                    print(f"Error calculating action smoothness: {e}")
                    continue
            
            if action_differences:
                # Action smoothness = 1 / (1 + average action difference)
                # Higher values indicate smoother actions
                episode_as = 1.0 / (1.0 + np.mean(action_differences))
                episode_smoothness.append(episode_as)
        
        overall_as = 1.0 / (1.0 + np.mean(all_smoothness)) if all_smoothness else 0.0
        return overall_as, episode_smoothness
    
    def generate_report(self, save_path: str = None) -> Dict:
        """
        Generate complete benchmark analysis report
        
        Parameters:
        -----------
        save_path : str, optional
            Report save path
            
        Returns:
        --------
        Dict : Analysis results dictionary
        """
        print("=== Benchmark Analysis Report ===")
        print(f"Analysis directory: {self.data_dir}")
        print(f"Number of episodes: {len(self.episodes_data)}")
        
        if not self.episodes_data:
            print("No data available for analysis")
            return {}
        
        # Calculate metrics
        sr = self.calculate_success_rate()
        mtbe_overall, mtbe_episodes = self.calculate_mtbe()
        as_overall, as_episodes = self.calculate_action_smoothness()
        
        # Statistical information
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
        
        # Print report
        print(f"\n--- Basic Statistics ---")
        print(f"Total Episodes: {report['summary']['total_episodes']}")
        print(f"Total Steps: {report['summary']['total_steps']}")
        print(f"Average Steps/Episode: {report['summary']['avg_steps_per_episode']:.1f}")
        
        print(f"\n--- Key Metrics ---")
        print(f"Success Rate (SR): {sr:.3f} ({sr*100:.1f}%)")
        print(f"Mean Track Belief Error (MTBE): {mtbe_overall:.4f}")
        print(f"Action Smoothness (AS): {as_overall:.4f}")
        
        if len(mtbe_episodes) > 1:
            print(f"\n--- Episode Details ---")
            print(f"MTBE Standard Deviation: {np.std(mtbe_episodes):.4f}")
            print(f"AS Standard Deviation: {np.std(as_episodes):.4f}")
            print(f"Best MTBE: {np.min(mtbe_episodes):.4f}")
            print(f"Worst MTBE: {np.max(mtbe_episodes):.4f}")
        
        # Save report
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\nReport saved to: {save_path}")
        
        return report
    
    def plot_metrics(self, save_dir: str = None):
        """
        Plot metric visualization charts
        
        Parameters:
        -----------
        save_dir : str, optional
            Chart save directory
        """
        if not self.episodes_data:
            print("No data available for plotting")
            return
        
        # Calculate metrics
        _, mtbe_episodes = self.calculate_mtbe()
        _, as_episodes = self.calculate_action_smoothness()
        
        # Create charts
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. MTBE per episode
        axes[0, 0].plot(range(1, len(mtbe_episodes)+1), mtbe_episodes, 'b-o')
        axes[0, 0].set_title('Mean Track Belief Error per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('MTBE')
        axes[0, 0].grid(True)
        
        # 2. Action Smoothness per episode
        axes[0, 1].plot(range(1, len(as_episodes)+1), as_episodes, 'r-o')
        axes[0, 1].set_title('Action Smoothness per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Action Smoothness')
        axes[0, 1].grid(True)
        
        # 3. Error distribution histogram
        all_errors = []
        for episode in self.episodes_data:
            for step in episode['data']:
                try:
                    target_pos = np.array(step['targets'])
                    belief_raw = step['belief_targets']
                    
                    # Handle belief_targets that may contain lists (e.g., fix_depth as [5, 15])
                    belief_pos = []
                    for i, val in enumerate(belief_raw):
                        if isinstance(val, (list, tuple, np.ndarray)):
                            # If it's a list/array, take the first element or mean
                            if len(val) > 0:
                                belief_pos.append(float(val[0]) if isinstance(val[0], (int, float)) else float(np.mean(val)))
                            else:
                                belief_pos.append(0.0)
                        else:
                            belief_pos.append(float(val))
                    belief_pos = np.array(belief_pos)
                    
                    # Ensure same length
                    min_len = min(len(target_pos), len(belief_pos))
                    target_pos = target_pos[:min_len]
                    belief_pos = belief_pos[:min_len]
                    
                    error = np.linalg.norm(target_pos - belief_pos)
                    all_errors.append(error)
                except:
                    continue
        
        axes[1, 0].hist(all_errors, bins=30, alpha=0.7, color='green')
        axes[1, 0].set_title('Belief Error Distribution')
        axes[1, 0].set_xlabel('Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # 4. Steps distribution
        steps_per_episode = [len(ep['data']) for ep in self.episodes_data]
        axes[1, 1].bar(range(1, len(steps_per_episode)+1), steps_per_episode, alpha=0.7, color='orange')
        axes[1, 1].set_title('Steps per Episode')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir)
            # If save_dir is a directory, append filename; if it's already a file path, use it directly
            if save_path.is_dir() or (not save_path.suffix):
                save_path = save_path / "benchmark_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        else:
            # If no save directory specified, save to current directory
            save_path = Path(self.data_dir) / "benchmark_metrics.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
        
        # Don't display charts as might be running in headless environment
        # plt.show()

def main():
    """Main function - usage example"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark analysis tool')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory path containing JSON files')
    parser.add_argument('--save_report', type=str, default=None,
                       help='Report save path')
    parser.add_argument('--save_plots', type=str, default=None,
                       help='Chart save directory')
    parser.add_argument('--min_steps', type=int, default=0,
                       help='Minimum number of steps required for an episode to be included (default: 0, no filtering)')
    
    args = parser.parse_args()
    
    # Create analyzer and run analysis
    analyzer = BenchmarkAnalyzer(args.data_dir, min_steps=args.min_steps)
    report = analyzer.generate_report(save_path=args.save_report)
    
    if args.save_plots:
        analyzer.plot_metrics(save_dir=args.save_plots)

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Use main() function to handle command line arguments
    main()
