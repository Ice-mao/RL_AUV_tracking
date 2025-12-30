import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import numpy as np
from datetime import datetime
from stable_baselines3 import SAC, PPO
import auv_env
from config_loader import load_config

def evaluate_rl_and_save(env, model_path, alg_config, num_episodes=5, save_prefix="rl", save_dir=None):
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
    base_dir = save_dir or f"log/benchmark/RL/{datetime.now().strftime('%m%d_%H')}"
    
    for episode in range(num_episodes):
        print(f"RL Episode {episode + 1}")
        
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
                print(f"Warning: Could not extract additional info: {e}")
            
            episode_infos.append(info)
            
            if terminated or truncated:
                break

        os.makedirs(base_dir, exist_ok=True)
        filename = f"{base_dir}/{save_prefix}_episode_{episode}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(episode_infos, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved {len(episode_infos)} steps of data to {filename}")
    
    env.close()
    print("RL algorithm evaluation completed!")

def evaluate_greedy_and_save(env, num_episodes=5, save_prefix="greedy"):
    """
    Run Greedy algorithm evaluation and save info for each step
    """

    try:
        from auv_baseline.greedy import Greedy
        greedy = Greedy(env.unwrapped.world, N=100)
    except ImportError:
        print("Cannot import Greedy algorithm, please ensure auv_baseline.greedy module exists")
        return
    
    print(f"Starting Greedy algorithm evaluation for {num_episodes} episodes...")
    timestamp = datetime.now().strftime("%m%d_%H")
    for episode in range(num_episodes):
        print(f"Greedy Episode {episode + 1}")
        
        # Store all info for this episode
        episode_infos = []
        
        obs, info = env.reset()
        step = 0
        
        while True:
            # Use Greedy algorithm to get action
            action = greedy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # Add step information
            info['step'] = step
            info['reward'] = float(reward)
            
            # Add Greedy-specific information
            if hasattr(greedy.world, 'belief_targets') and len(greedy.world.belief_targets) > 0:
                info['greedy_cov_det'] = float(np.linalg.det(greedy.world.belief_targets[0].cov))
                info['greedy_cov_trace'] = float(np.trace(greedy.world.belief_targets[0].cov))
            
            episode_infos.append(info)
            
            if terminated or truncated:
                break
        
        save_dir = f"log/benchmark/greedy/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/{save_prefix}_episode_{episode}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(episode_infos, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved {len(episode_infos)} steps of data to {filename}")
    
    env.close()
    print("Greedy algorithm evaluation completed!")

def evaluate_diffusion_and_save(model_path, env_config_path, num_episodes=5, save_prefix="diffusion"):
    """
    Run Diffusion Policy algorithm evaluation and save info for each step
    """
    import torch
    import dill
    import hydra
    from diffusion_policy.workspace.base_workspace import BaseWorkspace
    
    # Load configuration
    env_config = load_config(env_config_path)
    
    # env = auv_env.make(env_config['name'], config=env_config, eval=True, 
    #                    t_steps=env_config.get('t_steps', 1000), show_viewport=True)
    
    # Load Diffusion Policy model
    print(f"Loading Diffusion Policy model: {model_path}")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    payload = torch.load(open(model_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get policy
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    policy.to(device)
    policy.eval()
    print(f"✓ Diffusion Policy model loaded successfully! Device: {device}")
        
    
    print(f"Starting Diffusion Policy algorithm evaluation for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        print(f"Diffusion Episode {episode + 1}")
        
        # Store all info for this episode
        episode_infos = []
        
        obs, info = env.reset()
        if policy is not None:
            policy.reset()  # Reset policy state
        step = 0
        obs_dict = {}

        while True:
            # Prepare observation data
            obs_dict = {}
            
            # Process image observations
            if 'images' in obs:
                images = np.array(obs['images'])  # Should be [5, 3, 224, 224]
                obs_dict['camera_image'] = torch.from_numpy(images).float().to(device)
                # Add batch dimension and T dimension, if observation step length is not 0, need to pass in corresponding length obs
                obs_dict['camera_image'] = obs_dict['camera_image'].unsqueeze(0).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                action_dict = policy.predict_action(obs_dict)
            
            # Extract action
            action = action_dict['action'].detach().cpu().numpy()
            action = action[0, 0, :]  # Remove batch dimension, take first time step
                
            
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # Add step information
            info['step'] = step
            info['episode'] = episode + 1
            info['algorithm'] = 'Diffusion'
            info['reward'] = float(reward)
            
            episode_infos.append(info)
            
            if terminated or truncated:
                break
        
        # Episode ended, save all info to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"log/json/diffusion/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/{save_prefix}_episode_{episode+1}_data.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(episode_infos, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved {len(episode_infos)} steps of data to {filename}")
    
    env.close()
    print("Diffusion Policy algorithm evaluation completed!")


if __name__ == "__main__":
    import argparse
    # python evaluate.py --mode rl --rl_model_path your_model.zip
    # python evaluate.py --mode greedy
    # python evaluate.py --mode diffusion --diffusion_model_path your_diffusion_model.pth"
    parser = argparse.ArgumentParser(description='Evaluate performance of different algorithms')
    parser.add_argument('--mode', type=str, choices=['rl', 'greedy', 'diffusion'], default='rl')
    parser.add_argument('--rl_model_path', type=str, default='log/AUVTracking3D_v0/LQR/SAC/08-31_18/rl_model_1800000_steps.zip')
    parser.add_argument('--diffusion_model_path', type=str, 
                       default="data/outputs/19.53.52_train_diffusion_unet_image_track_image/checkpoints/epoch=0050-val_loss=0.228.ckpt")
    parser.add_argument('--env_config', type=str, 
                       default="configs/envs/3d_v1_config.yml") # 3d_v0_config_openwater
    parser.add_argument('--alg_config', type=str,
                       default="configs/algorithm/sac_3d_v0.yml")
    
    parser.add_argument('--num_episodes', type=int, default=3,
                       help='Number of evaluation episodes')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save benchmark results (default: log/benchmark/RL/<timestamp>)')
    
    args = parser.parse_args()
    
    env_config = load_config(args.env_config)
    alg_config = load_config(args.alg_config)
    
    env = auv_env.make(env_config['name'], config=env_config, eval=False, 
                       t_steps=env_config.get('t_steps', 1000), show_viewport=False)
    
    if args.mode == 'rl':
        print("Evaluating reinforcement learning algorithm...")
        evaluate_rl_and_save(env, args.rl_model_path, alg_config, args.num_episodes, save_dir=args.save_dir)
    elif args.mode == 'greedy':
        print("Evaluating Greedy algorithm...")
        evaluate_greedy_and_save(env, args.num_episodes)
    elif args.mode == 'diffusion':
        print("Evaluating Diffusion Policy algorithm...")
        evaluate_diffusion_and_save(args.diffusion_model_path, args.env_config, args.num_episodes)