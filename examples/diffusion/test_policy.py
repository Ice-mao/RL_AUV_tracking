import torch
import os
from unet_1d_condition_dev import UNet1DConditionModel
from torch import nn
from auv_track_launcher.networks.diffusion_vision_encoder import Encoder
from diffusers import DDPMScheduler, DDPMPipeline

# 1. 创建模型架构（与训练时相同）
def create_model():
    vision_encoder = Encoder(num_channels=[512, 256])
    model = UNet1DConditionModel(
        input_dim=3,
        local_cond_dim=None,
        global_cond_dim=1024,
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

def main():
    model_path = "auv_tracking_diffusion_policy"  # 模型保存路径
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    net = load_model(model_path, "unet_ema").to(device)
    
    scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    # model = DDPMPipeline(unet=net["model"], scheduler=scheduler)
    scheduler.set_timesteps(200)

    import datasets
    from auv_track_launcher.dataset.holoocean_image_dataset import HoloOceanImageDataset
    dataset_name = "/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/trajs_dam/"
    dataset_0 = datasets.load_from_disk(dataset_name+"traj_1")
    dataset_1 = datasets.load_from_disk(dataset_name+"traj_2")
    dataset_2 = datasets.load_from_disk(dataset_name+"traj_3")
    _dataset = datasets.concatenate_datasets([dataset_0, dataset_1, dataset_2])
    dataset = HoloOceanImageDataset(_dataset,
                                    horizon=16,
                                    obs_horizon=4,
                                    pred_horizon=12)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )
    batch = next(iter(train_dataloader))
    true_action = batch["action"].float().to(device)
    true_images = batch["obs"].float().to(device)
    image_features = net['vision_encoder'](true_images)

    pred_horizon=12
    noise = torch.randn((1, pred_horizon, 3), device="cuda")
    input = noise

    for t in scheduler.timesteps:
        with torch.no_grad():
            noisy_residual = net['model'](noise, t, encoder_hidden_states=image_features).sample
        previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = previous_noisy_sample
        
    input[:, :, 0] = 0.5 * input[:, :, 0] + 0.5

    print("Diffusion action:", input)
    print("True action:", true_action)
if __name__ == "__main__":
    main()