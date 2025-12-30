"""
运行benchmark评估的独立脚本
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
from datetime import datetime
from stable_baselines3 import SAC, PPO
import auv_env
from config_loader import load_config

def evaluate_rl_and_save(env, model_path, alg_config, num_episodes=100, save_prefix="rl", save_dir=None):
    """
    Run reinforcement learning algorithm evaluation and save info for each step
    """
    policy_name = alg_config['policy_hparams']['policy']
    if policy_name == 'SAC' or policy_name == 'CustomSACPolicy':
        model = SAC.load(model_path, device='cuda', env=env,
                         custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    elif policy_name == 'PPO':
        model = PPO.load(model_path, device='cuda', env=env,
                         custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    else:
        raise ValueError(f"Unsupported policy: {policy_name}")
    
    print(f"Starting RL algorithm evaluation for {num_episodes} episodes...")
    # allow custom save_dir; fallback to timestamped default
    base_dir = save_dir or f"log/benchmark/RL/{datetime.now().strftime('%m%d_%H')}"
    
    for episode in range(num_episodes):
        print(f"RL Episode {episode + 1}/{num_episodes}")
        
        # Store all info for this episode
        episode_infos = []
        
        obs, info = env.reset()
        step = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # Add step information
            info['step'] = step
            info['reward'] = float(reward)
            
            # Extract additional information from environment for analysis
            try:
                world = env.world
                # Add action
                if isinstance(action, np.ndarray):
                    info['action'] = action.tolist()
                else:
                    info['action'] = action
                
                # Add agent position (2D for v1, 3D for 3D versions)
                if hasattr(world, 'agent') and hasattr(world.agent, 'est_state'):
                    agent_pos = world.agent.est_state.vec[:3].tolist() if len(world.agent.est_state.vec) >= 3 else world.agent.est_state.vec[:2].tolist()
                    info['agent_pos'] = agent_pos
                
                # Add target position
                if hasattr(world, 'targets') and len(world.targets) > 0:
                    target_pos = world.targets[0].state.vec[:3].tolist() if len(world.targets[0].state.vec) >= 3 else world.targets[0].state.vec[:2].tolist()
                    info['targets'] = target_pos
                
                # Add belief target position
                # Handle 2D vs 3D environments correctly
                if hasattr(world, 'belief_targets') and len(world.belief_targets) > 0:
                    belief_state = world.belief_targets[0].state
                    belief_state_dim = len(belief_state)
                    
                    if belief_state_dim == 4:
                        # 2D environment: state is [x, y, vx, vy]
                        # Use fix_depth for z coordinate
                        fix_depth = getattr(world, 'fix_depth', -5.0)  # Default to -5 if not available
                        belief_pos = [belief_state[0], belief_state[1], fix_depth]
                    elif belief_state_dim >= 6:
                        # 3D environment: state is [x, y, z, vx, vy, vz]
                        belief_pos = belief_state[:3].tolist()
                    else:
                        # Fallback: try to get what we can
                        belief_pos = belief_state[:min(3, belief_state_dim)].tolist()
                        if len(belief_pos) < 3:
                            fix_depth = getattr(world, 'fix_depth', -5.0)
                            belief_pos.append(fix_depth)
                    
                    info['belief_targets'] = belief_pos
                
                # Add collision information
                if hasattr(world, 'is_col'):
                    info['is_collision'] = world.is_col
                
                info['done'] = terminated or truncated
            except Exception as e:
                # If extraction fails, continue without these fields
                if step == 1:  # Only print warning once per episode
                    print(f"  Warning: Could not extract additional info: {e}")
            
            episode_infos.append(info)
            
            if terminated or truncated:
                break

        os.makedirs(base_dir, exist_ok=True)
        filename = f"{base_dir}/{save_prefix}_episode_{episode}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(episode_infos, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved {len(episode_infos)} steps of data to {filename}")
    
    env.close()
    print(f"\nRL algorithm evaluation completed! Results saved to {base_dir}")
    return base_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RL benchmark evaluation')
    parser.add_argument('--rl_model_path', type=str, 
                       default='log/AUVTracking_v1/LQR/CustomSACPolicy/rl_model_2549898_steps.zip',
                       help='Path to RL model')
    parser.add_argument('--env_config', type=str, 
                       default='configs/envs/v1_config.yml',
                       help='Environment configuration file')
    parser.add_argument('--alg_config', type=str,
                       default='configs/algorithm/sac_v1.yml',
                       help='Algorithm configuration file')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save benchmark results (default: log/benchmark/RL/<timestamp>)')
    
    args = parser.parse_args()
    
    env_config = load_config(args.env_config)
    alg_config = load_config(args.alg_config)
    
    env = auv_env.make(env_config['name'], config=env_config, eval=False, 
                       t_steps=env_config.get('t_steps', 1000), show_viewport=False)
    
    print("Evaluating reinforcement learning algorithm...")
    save_dir = evaluate_rl_and_save(env, args.rl_model_path, alg_config, args.num_episodes, save_dir=args.save_dir)
    
    print(f"\nTo analyze the results, run:")
    print(f"python examples/benchmark/benchmark_analyzer.py --data_dir {save_dir} --save_report {save_dir}/report.json --save_plots {save_dir}")

