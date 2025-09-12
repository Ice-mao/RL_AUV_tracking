import argparse
import datetime
import os
import pathlib
import sys
import torch
import numpy as np
import auv_env
import csv
import importlib

from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecEnv, DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
#
import auv_env
import torch
import numpy as np
from numpy import linalg as LA
import csv
import argparse
from auv_track_launcher.common.callbacks import SaveOnBestTrainingRewardCallback
from config_loader import load_config

# tools
import os
import datetime

current_time = datetime.datetime.now()
time_string = current_time.strftime('%m-%d_%H')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--choice', choices=['0', '1', '2', '3', '4'], help='0:train; 1:keep train; 2:eval; 3:test; 4:cal',
                    default='0')
parser.add_argument('--env_config', type=str, required=True, help='Path to the environment configuration file.')
parser.add_argument('--alg_config', type=str, required=True, help='Path to the algorithm configuration file.')
parser.add_argument('--eval', 
                    action='store_true')
parser.add_argument('--show_viewport',
                    action='store_true')
args = parser.parse_args()


def make_env(
        env_config: str,
        num_train_envs: int,
        monitor_dir: str,
) -> VecEnv:
    # train_envs = SubprocVecEnv([lambda: gym.make(task, config=config) for _ in range(num_train_envs)], )
    train_envs = SubprocVecEnv([lambda: auv_env.make(env_config['name'],
                                config=env_config,
                                eval=args.eval, t_steps=env_config.get('t_steps', 200),
                                show_viewport=args.show_viewport) for _ in range(num_train_envs)], )
    env = VecMonitor(train_envs, monitor_dir)
    # env = auv_env.make("AUVTracking_v0",
    #                             config=env_config,
    #                             eval=False, t_steps=200,
    #                             show_viewport=True) 
    return env


def make_callback(
        log_dir: str,
        alg_config: dict,
) -> CallbackList:
    # Use get to provide a default value if the key doesn't exist
    nb_envs = alg_config['training']['nb_envs']
    callback = SaveOnBestTrainingRewardCallback(check_freq=alg_config['training']['log_freq'], log_dir=log_dir, save_path=log_dir)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(alg_config['training']['save_freq'] // nb_envs, 1),
        save_path=log_dir,
        name_prefix="rl_model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    callback = CallbackList([callback, checkpoint_callback])
    return callback

def create_action_noise(config):
    if config['agent']['controller']=='PID':
        if config['agent']['controller_config']['PID']['action_dim'] == 2:
            noise = NormalActionNoise(np.array([0.0, 0.0]), np.array([0.05, 0.05]))
        elif config['agent']['controller_config']['PID']['action_dim'] == 3:
            noise = NormalActionNoise(np.array([0.0, 0.0, 0.0]), np.array([0.05, 0.05, 0.05]))
    elif config['agent']['controller']=='LQR':
        if config['agent']['controller_config']['LQR']['action_dim'] == 3:
            noise = NormalActionNoise(np.array([0.0, 0.0, 0.0]), np.array([0.05, 0.05, 0.03]))
        elif config['agent']['controller_config']['LQR']['action_dim'] == 4:
            noise = NormalActionNoise(np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.05, 0.05, 0.05, 0.03]))
    else:
        noise = None
    return noise

def learn(env, log_dir, env_config, alg_config):
    # callback
    callback = make_callback(log_dir, alg_config)
    policy_params = alg_config['policy_hparams']
    training_params = alg_config['training']

    if policy_params['policy'] == 'SAC':
        # policy network
        features_extractor_class, fe_kwargs = get_features_extractor(alg_config)
        policy_kwargs = dict(
            net_arch=alg_config['policy']['net_arch'],
        )
        if features_extractor_class:
            policy_kwargs['features_extractor_class'] = features_extractor_class
            policy_kwargs['features_extractor_kwargs'] = fe_kwargs
            policy_type = 'MultiInputPolicy'
        else:
            policy_type = 'MlpPolicy'

        # action noise 
        action_noise = create_action_noise(env_config)
        
        # replay buffer
        if policy_params['replay_buffer']['type'] == 'HerReplayBuffer':
            buffer_type = "HerReplayBuffer"
            buffer_kwargs = dict(
                n_sampled_goal=policy_params['replay_buffer']['her_kwargs']['n_sampled_goal'],
                goal_selection_strategy=policy_params['replay_buffer']['her_kwargs']['goal_selection_strategy']
            )
            model = SAC(policy_type, env, verbose=1,
                    learning_rate=policy_params['lr'],
                    buffer_size=policy_params['buffer_size'],
                    learning_starts=policy_params['start_timesteps'],
                    batch_size=policy_params['batch_size'],
                    tau=policy_params['tau'],
                    gamma=policy_params['gamma'],
                    train_freq=1,
                    gradient_steps=1,
                    action_noise=action_noise,
                    target_update_interval=10,
                    policy_kwargs=policy_kwargs,
                    replay_buffer_class=buffer_type,
                    replay_buffer_kwargs=buffer_kwargs,
                    tensorboard_log=log_dir,
                    device=training_params['device']
                    )
        else:
            model = SAC(policy_type, env, verbose=1,
                        learning_rate=policy_params['lr'],
                        buffer_size=policy_params['buffer_size'],
                        learning_starts=policy_params['start_timesteps'],
                        batch_size=policy_params['batch_size'],
                        tau=policy_params['tau'],
                        gamma=policy_params['gamma'],
                        train_freq=1,
                        gradient_steps=1,
                        action_noise=action_noise,
                        target_update_interval=10,
                        policy_kwargs=policy_kwargs,
                        # replay_buffer_class=policy_params['replay_buffer']['type'],
                        # replay_buffer_kwargs=buffer_kwargs,
                        tensorboard_log=log_dir,
                        device=training_params['device']
                        )
        
        model.learn(total_timesteps=training_params['timesteps'], tb_log_name="first_run", log_interval=5, callback=callback)
        model.save(os.path.join(log_dir, 'final_model'))
        
    elif policy_params['policy'] == 'PPO':
        features_extractor_class, fe_kwargs = get_features_extractor(alg_config)
        policy_kwargs = dict(
            net_arch=alg_config['policy']['net_arch'],
        )
        if features_extractor_class:
            policy_kwargs['features_extractor_class'] = features_extractor_class
            policy_kwargs['features_extractor_kwargs'] = fe_kwargs
            policy_type = 'MultiInputPolicy'
        else:
            policy_type = 'MlpPolicy'
        model = PPO(policy_type, env, verbose=1,
                    learning_rate=policy_params['lr'],
                    batch_size=training_params['batch_size'],
                    n_epochs=10,
                    n_steps=policy_params['n_steps'],
                    gae_lambda=policy_params['gae_lambda'],
                    clip_range=policy_params['eps_clip'],
                    clip_range_vf=policy_params['value_clip'],
                    ent_coef=policy_params['ent_coef'],
                    vf_coef=policy_params['vf_coef'],
                    max_grad_norm=policy_params['max_grad_norm'],
                    normalize_advantage=policy_params['norm_adv'],
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=log_dir,
                    device=training_params['device']
            )
        model.learn(total_timesteps=training_params['timesteps'], tb_log_name="first_run", log_interval=1, callback=callback)
        model.save(os.path.join(log_dir, 'final_model'))


def keep_learn(env, log_dir, model_name, config):
    training_params = config['training']
    policy_params = config['policy_hparams']
    device = torch.device(training_params['device'])
    callback = make_callback(log_dir, config)

    if policy_params['policy'] == 'SAC':
        model = SAC.load(model_name, device=device, env=env,
                         custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    elif policy_params['policy'] == 'PPO':
        model = PPO.load(model_name, device=device, env=env,
                         custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    else:
        raise ValueError(f"Unknown policy {policy_params['policy']}")

    model.learn(total_timesteps=training_params['timesteps'], tb_log_name="second_run", reset_num_timesteps=False,
                log_interval=5, callback=callback)
    model.save(os.path.join(log_dir, 'final_model'))


def evaluate(model_name: str, env_config: dict, alg_config: dict):
    """
    2
    :param model_name:
    :return:
    """
    env = auv_env.make(env_config['name'],
                        config=env_config,
                        eval=True, t_steps=200,
                        show_viewport=True) 
    policy_name = alg_config['policy_hparams']['policy']

    if policy_name == 'SAC':
        model = SAC.load(model_name, device='cuda', env=env,
                         custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    elif policy_name == 'PPO':
        model = PPO.load(model_name, device='cuda', env=env,
                         custom_objects={'observation_space': env.observation_space, 'action_space': env.action_space})
    else:
        raise ValueError(f"Unknown policy {policy_name}")
    
    import json
    collected_obs = [] 
    obs, _ = env.reset()
    for _ in range(50000):
        action, _ = model.predict(obs, deterministic=True)
        print(action)
        obs, reward, done, _, inf = env.step(action)
        collected_obs.append({'obs':obs})
        if len(collected_obs) == 1000:
            with open("observations.json", "w") as f:
                json.dump(collected_obs, f, indent=2)


def eval_greedy(model_dir, config: dict):
    """
    4
    :param model_dir:
    :return:
    """
    eval_config = load_config(os.path.join(ROOT_DIR, 'configs/eval_tracking_config.yml'))
    env_name = config['env']['name']

    env = auv_env.make(env_name,
                       config=config,
                       render=config['render'],
                       record=args.record,
                       directory=model_dir,
                       num_targets=config['env']['target_num'],
                       eval=True,
                       is_training=False,
                       t_steps=config['agent']['max_episode_step']
                       )
    from auv_baseline.greedy import Greedy
    # Assuming the environment may be wrapped, we access the underlying environment's world.
    greedy = Greedy(env.unwrapped.world)

    for i in range(10):
        # init the eval data
        prior_data = []  # sigma t+1
        posterior_data = []  # sigma t+1|t
        observed = []
        is_col = []

        obs, _ = env.reset()
        for _ in range(args.max_episode_step):
            action = greedy.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            prior_data.append(greedy.world.belief_targets[0].cov.flatten())
            posterior_data.append(greedy.world.belief_targets[0].cov.flatten())
            is_col.append(info['is_col'])
            observed.append(info['observed'])
            if done:
                break

    with open(model_dir + '/greedy_l.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(prior_data)
    with open(model_dir + '/greedy_q.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(posterior_data)
    with open(model_dir + '/greedy_observed.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(observed)
    with open(model_dir + '/greedy_is_col.csv', 'w') as f:
        write = csv.writer(f)
        write.writerows(is_col)

def get_features_extractor(config):
    if 'features_extractor' in config['policy']:
        fe_config = config['policy']['features_extractor']
        module = importlib.import_module(fe_config['module_path'])
        fe_class = getattr(module, fe_config['class_name'])
        return fe_class, fe_config.get('kwargs', {})
    return None, {}

if __name__ == '__main__':
    args = parser.parse_args()

    # Load base configuration from env config
    env_config = load_config(args.env_config)
    alg_config = load_config(args.alg_config)
    
    choice = int(args.choice)
    # Create a unique log directory
    policy_name = alg_config['policy_hparams']['policy']

    if choice == 0:  # Train
        log_dir = os.path.join(alg_config['training']['log_dir'], env_config['name'],
                                env_config['agent']['controller'], policy_name, time_string)
        os.makedirs(log_dir, exist_ok=True)
        env = make_env(env_config, alg_config['training']['nb_envs'], log_dir)
        learn(env, log_dir, env_config, alg_config)
    elif choice == 1:  # Keep training
        model_path = alg_config['training']['resume_path']
        log_dir = os.path.dirname(model_path)
        if not model_path:
            raise ValueError("Resume path must be provided for 'keep training' choice.")
        env = make_env(env_config, alg_config['training']['nb_envs'], log_dir)
        keep_learn(env, log_dir, model_path, alg_config)
    elif choice == 2:  # Evaluate
        model_path = alg_config['training']['resume_path']
        if not model_path:
            raise ValueError("Resume path must be provided for 'evaluate' choice.")
        evaluate(model_path, env_config, alg_config)
    elif choice == 3:  # Calibrate/Eval Greedy
        model_dir = os.path.dirname(alg_config['training']['resume_path'])
        eval_greedy(model_dir, env_config)
