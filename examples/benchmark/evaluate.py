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
    运行强化学习算法评估并保存每一步的info
    """
    policy_name = alg_config['policy_hparams']['policy']
    if policy_name == 'SAC':
        model = SAC.load(model_path, device='cuda', env=env)
    elif policy_name == 'PPO':
        model = PPO.load(model_path, device='cuda', env=env)
    else:
        raise ValueError(f"不支持的策略: {policy_name}")
    
    print(f"开始RL算法评估 {num_episodes} 个回合...")
    timestamp = datetime.now().strftime("%m%d_%H")
    
    for episode in range(num_episodes):
        print(f"RL回合 {episode + 1}")
        
        # 存储这个episode的所有info
        episode_infos = []
        
        obs, info = env.reset()
        step = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # 添加步骤信息
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
        
        print(f"  已保存 {len(episode_infos)} 步数据到 {filename}")
    
    env.close()
    print("RL算法评估完成!")

def evaluate_greedy_and_save(env, num_episodes=5, save_prefix="greedy"):
    """
    运行Greedy算法评估并保存每一步的info
    """

    try:
        from auv_baseline.greedy import Greedy
        greedy = Greedy(env.unwrapped.world, N=100)
    except ImportError:
        print("无法导入Greedy算法，请确保auv_baseline.greedy模块存在")
        return
    
    print(f"开始Greedy算法评估 {num_episodes} 个回合...")
    timestamp = datetime.now().strftime("%m%d_%H")
    for episode in range(num_episodes):
        print(f"Greedy回合 {episode + 1}")
        
        # 存储这个episode的所有info
        episode_infos = []
        
        obs, info = env.reset()
        step = 0
        
        while True:
            # 使用Greedy算法获取动作
            action = greedy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # 添加步骤信息
            info['step'] = step
            info['reward'] = float(reward)
            
            # 添加Greedy特有的信息
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
        
        print(f"  已保存 {len(episode_infos)} 步数据到 {filename}")
    
    env.close()
    print("Greedy算法评估完成!")

def evaluate_diffusion_and_save(model_path, env_config_path, num_episodes=5, save_prefix="diffusion"):
    """
    运行Diffusion Policy算法评估并保存每一步的info
    注意：这是一个模板实现，具体的动作执行流程需要根据实际的diffusion policy模型调整
    """
    # 加载配置
    env_config = load_config(env_config_path)
    
    env = auv_env.make(env_config['name'], config=env_config, eval=True, 
                       t_steps=env_config.get('t_steps', 1000), show_viewport=True)
    
    # TODO: 加载Diffusion Policy模型
    # 这里需要根据实际的diffusion policy实现来调整
    print("注意：Diffusion Policy评估当前使用随机动作，需要根据实际模型调整")
    
    print(f"开始Diffusion Policy算法评估 {num_episodes} 个回合...")
    
    for episode in range(num_episodes):
        print(f"Diffusion回合 {episode + 1}")
        
        # 存储这个episode的所有info
        episode_infos = []
        
        obs, info = env.reset()
        step = 0
        
        while True:
            # TODO: 使用Diffusion Policy模型获取动作
            # 目前使用随机动作作为占位符
            action = env.action_space.sample()
            
            # 如果有实际的diffusion policy模型，替换上面的代码：
            # action = diffusion_model.predict(obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # 添加步骤信息
            info['step'] = step
            info['episode'] = episode + 1
            info['algorithm'] = 'Diffusion'
            info['reward'] = float(reward)
            
            # 添加Diffusion特有的信息（如果有的话）
            info['diffusion_note'] = 'Using random actions - replace with actual diffusion policy'
            
            episode_infos.append(info)
            
            if terminated or truncated:
                break
        
        # episode结束，保存所有info到文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"log/json/diffusion/{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{save_dir}/{save_prefix}_episode_{episode+1}_data.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(episode_infos, f, indent=2, ensure_ascii=False)
        
        print(f"  已保存 {len(episode_infos)} 步数据到 {filename}")
    
    env.close()
    print("Diffusion Policy算法评估完成!")


if __name__ == "__main__":
    import argparse
    # python evaluate.py --mode rl --rl_model_path your_model.zip
    # python evaluate.py --mode greedy
    # python evaluate.py --mode diffusion --diffusion_model_path your_diffusion_model.pth"
    parser = argparse.ArgumentParser(description='评估不同算法的性能')
    parser.add_argument('--mode', type=str, choices=['rl', 'greedy', 'diffusion'], default='rl')
    parser.add_argument('--rl_model_path', type=str, default='log/AUVTracking3D_v0/LQR/SAC/08-31_18/rl_model_1800000_steps.zip')
    parser.add_argument('--diffusion_model_path', type=str, 
                       default="path/to/diffusion/model.pth")
    parser.add_argument('--env_config', type=str, 
                       default="configs/envs/3d_v0_config.yml")
    parser.add_argument('--alg_config', type=str,
                       default="configs/algorithm/sac_3d_v0.yml")
    
    parser.add_argument('--num_episodes', type=int, default=3,
                       help='评估回合数')
    
    args = parser.parse_args()
    
    env_config = load_config(args.env_config)
    alg_config = load_config(args.alg_config)
    
    env = auv_env.make(env_config['name'], config=env_config, eval=False, 
                       t_steps=env_config.get('t_steps', 1000), show_viewport=False)
    
    if args.mode == 'rl':
        print("评估强化学习算法...")
        evaluate_rl_and_save(env, args.rl_model_path, alg_config, args.num_episodes)
    elif args.mode == 'greedy':
        print("评估Greedy算法...")
        evaluate_greedy_and_save(env, args.num_episodes)
    elif args.mode == 'diffusion':
        print("评估Diffusion Policy算法...")
        evaluate_diffusion_and_save(env, args.diffusion_model_path, args.num_episodes)