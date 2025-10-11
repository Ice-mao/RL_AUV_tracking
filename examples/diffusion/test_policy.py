import torch
import os
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from torch import nn
from auv_track_launcher.networks.diffusion_vision_encoder import Encoder
from diffusers import DDPMScheduler, DDPMPipeline
import torchvision.transforms.functional as F

def create_model():
    vision_encoder = Encoder(num_channels=[512, 256])
    model = ConditionalUnet1D(
        input_dim=3,
        global_cond_dim=256*5,
    )
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'model': model
    })
    return nets

def load_model(model_path, pt_file):
    nets = create_model()
    state_dict = torch.load(os.path.join(model_path, pt_file))
    nets.load_state_dict(state_dict)
    return nets

def create_pipeline(nets, device="cuda"):
    nets = nets.to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    def inference_fn(image_input, num_inference_steps=50):
        """
        从图像输入生成动作序列
        
        Args:
            image_input
            num_inference_steps
            
        Returns:
            predicted_actions
        """
        nets.eval()
        with torch.no_grad():
            image_features = nets['vision_encoder'](image_input)
            
            batch_size = image_input.shape[0]
            action_shape = (batch_size, 16, 3)
            noisy_action = torch.randn(action_shape).to(device)
            
            scheduler = DDPMScheduler(num_train_timesteps=1000)
            scheduler.set_timesteps(num_inference_steps)
            
            for t in scheduler.timesteps:
                model_output = nets['model'](
                    noisy_action, 
                    timestep=t, 
                    encoder_hidden_states=image_features
                ).sample
                
                noisy_action = scheduler.step(model_output, t, noisy_action).prev_sample
            
            return noisy_action
    
    return inference_fn

from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecEnv
import gymnasium as gym
import auv_env
import numpy as np
from auv_env.envs.tools import ImageBuffer
def main():
    # model init
    model_path = "/home/dell-t3660tow/data/remote_server/diff_test/auv_tracking_diffusion_policy_0414_1136"  # 模型保存路径
    device = "cuda"
    net = load_model(model_path, "unet_ema").to(device)
    scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    scheduler.set_timesteps(200)
    obs_horizon = 5
    pred_horizon=12
    
    image_buffer = ImageBuffer(5, (3, 224, 224), time_gap=0.5)

    env = gym.make('v1-Student-sample')
    obs, _ = env.reset()
    image_buffer.add_image(obs['images'], 0.0)

    for _ in range(500):
        image = np.stack(image_buffer.get_buffer())
        obs_tensor = torch.from_numpy(image).float().to(device)
        obs_tensor= F.resize(obs_tensor, (128, 128))
        obs_tensor = F.normalize(
            obs_tensor, 
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        true_images = obs_tensor[-obs_horizon:].unsqueeze(0)
        image_features = net['vision_encoder'](true_images)

        noise = torch.randn((1, pred_horizon, 3), device=device)
        input = noise

        for t in scheduler.timesteps:
            with torch.no_grad():
                noisy_residual = net['model'](noise, t, encoder_hidden_states=image_features).sample
            previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
            input = previous_noisy_sample
            
        input[:, :, 0] = 0.5 * input[:, :, 0] + 0.5
        action = input[0, 0, :].detach().cpu().numpy()
        obs, reward, dones, _, inf = env.step(action)

if __name__ == "__main__":
    main()