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

def evaluate_rl_and_save(env, model_path, alg_config, num_episodes=5, save_prefix="rl"):
    """
    Run reinforcement learning algorithm evaluation and save info for each step
    """
    policy_name = alg_config['policy_hparams']['policy']
    if policy_name == 'SAC':
        model = SAC.load(model_path, device='cuda', env=env)
    elif policy_name == 'PPO':
        model = PPO.load(model_path, device='cuda', env=env)
    else:
        raise ValueError(f"Unsupported policy: {policy_name}")
    
    print(f"Starting RL algorithm evaluation for {num_episodes} episodes...")
    timestamp = datetime.now().strftime("%m%d_%H")
    
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
            episode_infos.append(info)
            
            if terminated or truncated:
                break

        save_dir = f"log/benchmark/RL/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/{save_prefix}_episode_{episode}.json"
        
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
    
    args = parser.parse_args()
    
    env_config = load_config(args.env_config)
    alg_config = load_config(args.alg_config)
    
    env = auv_env.make(env_config['name'], config=env_config, eval=False, 
                       t_steps=env_config.get('t_steps', 1000), show_viewport=False)
    
    if args.mode == 'rl':
        print("Evaluating reinforcement learning algorithm...")
        evaluate_rl_and_save(env, args.rl_model_path, alg_config, args.num_episodes)
    elif args.mode == 'greedy':
        print("Evaluating Greedy algorithm...")
        evaluate_greedy_and_save(env, args.num_episodes)
    elif args.mode == 'diffusion':
        print("Evaluating Diffusion Policy algorithm...")
        evaluate_diffusion_and_save(args.diffusion_model_path, args.env_config, args.num_episodes)