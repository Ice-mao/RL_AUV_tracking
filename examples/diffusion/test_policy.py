import torch
import os
from auv_track_launcher.networks.unet_1d_condition import UNet1DConditionModel
from torch import nn
from auv_track_launcher.networks.diffusion_vision_encoder import Encoder
from diffusers import DDPMScheduler, DDPMPipeline
import torchvision.transforms.functional as F

# 1. 创建模型架构（与训练时相同）
def create_model():
    vision_encoder = Encoder(num_channels=[512, 256])
    model = UNet1DConditionModel(
        input_dim=3,
        local_cond_dim=None,
        global_cond_dim=256*5,
        cross_attention_dim=512,
        time_embedding_type="fourier",
        flip_sin_to_cos=True,
        freq_shift=0,
        block_out_channels=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        act_fn="mish",
        cond_predict_scale=False,
    )
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'model': model
    })
    return nets

# 2. 加载训练好的权重
def load_model(model_path, pt_file):
    nets = create_model()
    state_dict = torch.load(os.path.join(model_path, pt_file))
    nets.load_state_dict(state_dict)
    return nets

# 3. 创建推理pipeline
def create_pipeline(nets, device="cuda"):
    nets = nets.to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    # 创建自定义推理函数
    def inference_fn(image_input, num_inference_steps=50):
        """
        从图像输入生成动作序列
        
        Args:
            image_input: 形状为[B, C, H, W]的图像张量
            num_inference_steps: 扩散推理步数
            
        Returns:
            预测的动作序列
        """
        nets.eval()
        with torch.no_grad():
            # 编码图像
            image_features = nets['vision_encoder'](image_input)
            
            # 初始化随机噪声作为起点
            batch_size = image_input.shape[0]
            # 假设动作维度为3，序列长度为16
            action_shape = (batch_size, 16, 3)
            noisy_action = torch.randn(action_shape).to(device)
            
            # 创建采样器
            scheduler = DDPMScheduler(num_train_timesteps=1000)
            scheduler.set_timesteps(num_inference_steps)
            
            # 执行逆向扩散过程
            for t in scheduler.timesteps:
                # 模型预测噪声
                model_output = nets['model'](
                    noisy_action, 
                    timestep=t, 
                    encoder_hidden_states=image_features
                ).sample
                
                # 执行去噪步骤
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
        # print(input)
        obs, reward, dones, _, inf = env.step(action)

if __name__ == "__main__":
    main()