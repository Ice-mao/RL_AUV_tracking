# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tianshou_launcher.networks.tcn import TemporalConvNet

from tianshou_launcher import utils


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class EncoderResNet(nn.Module):
    def __init__(self, encoder_dim=64):
        super(EncoderResNet, self).__init__()
        # 加载预训练的 ResNet-50 模型
        resnet = models.resnet18('IMAGENET1K_V1')
        # 去掉最后的全连接层，保留到倒数第二层
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]) # 去掉 fc 层
        self.fc = nn.Linear(in_features=512, out_features=encoder_dim, bias=True)

    def forward(self, x):
        # 输入 x: [batch_size, 3, H, W]
        features = self.feature_extractor(x)  # 输出: [batch_size, 512, 1, 1]
        features_fc_input = features.view(features.size(0), -1)
        output = self.fc(features_fc_input)
        return output  # 展平: [batch_size, 128]


class EncoderTCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, kernel_size=3, device='cuda'):
        super(EncoderTCN, self).__init__()
        layers = []
        for i in range(num_layers):
            in_channels = input_dim if i == 0 else hidden_dim
            layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=(kernel_size - 1)))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
        self.tcn = nn.Sequential(*layers).to(device)
        self.fc = nn.Linear(hidden_dim, output_dim).to(device)

    def forward(self, x):
        # 输入 x: [batch_size, seq_len, input_dim] -> 转换为 [batch_size, input_dim, seq_len]
        x = x.permute(0, 2, 1)
        x = self.tcn(x)  # 输出: [batch_size, hidden_dim, seq_len]
        x = torch.mean(x, dim=-1)  # 对时间步求平均: [batch_size, hidden_dim]
        return self.fc(x)  # 输出: [batch_size, output_dim]


class Encoder(nn.Module):
    """

    """

    def __init__(self, N, resnet_output_dim=64, output_dim=256, tcn_hidden_dim=512, tcn_output_dim=256, tcn_layers=2,
                 device='cuda'):
        super(Encoder, self).__init__()
        self.num_images = N
        self.resnets = nn.ModuleList([EncoderResNet().to(device) for _ in range(self.num_images)])
        # self.tcn = EncoderTCN(input_dim=resnet_output_dim*N, hidden_dim=tcn_hidden_dim,
        #                       output_dim=tcn_output_dim, num_layers=tcn_layers, device=device)
        num_channels = [128, 64]
        self.tcn = TemporalConvNet(num_inputs=resnet_output_dim, num_channels=num_channels, kernel_size=2,
                                   dropout=0.2).to(device)
        self.linear = nn.Linear(num_channels[-1], output_dim).to(device)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 输入 images: [batch_size, num_images, 3, H, W]
        batch_size, num_images, C, H, W = images.size()
        assert num_images == self.num_images, "Input number of images must match num_images."

        # 提取每张图像的特征
        features = []
        for i in range(num_images):
            img = images[:, i, :, :, :]  # 取出第 i 张图像: [batch_size, 3, H, W]
            feature = self.resnets[i](img)  # 提取特征: [batch_size, resnet_output_dim]
            features.append(feature)

        # 拼接特征: [batch_size, N, resnet_output_dim]
        features = torch.stack(features, dim=1)
        features = features.permute(0, 2, 1)

        # 输入 TCN 进行时序建模
        encoding = self.tcn(features)  # 输出: [batch_size, tcn_output_dim]
        encoding = encoding.mean(dim=-1)
        output = self.linear(encoding)
        return output


class Actor(nn.Module):
    def __init__(self, encoder_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(encoder_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, encoder_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(encoder_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics


if __name__ == "__main__":
    # 参数设置
    batch_size = 4
    num_images = 5  # n+1 张图片
    image_size = (3, 224, 224)  # 符合 ResNet 的输入要求

    # 随机生成输入数据
    images = torch.rand(batch_size, num_images, *image_size, device='cuda')  # [batch_size, num_images, 3, 224, 224]

    # 初始化网络
    model = Encoder(N=num_images)
    output = model(images)  # 输出编码结果

    print("输出编码结果的形状:", output.shape)  # [batch_size, tcn_output_dim]
