"""
BC Training Script
训练 Behavioral Cloning 策略
"""

import sys
import os
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
from datetime import datetime

from examples.bc.bc_policy import BCPolicy
from examples.bc.bc_dataset import BCDataset


def train_bc(args):
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"bc_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # 保存配置
    with open(output_dir / "config.json", 'w') as f:
        json.dump(vars(args), f, indent=2)

    # 加载数据集
    print(f"Loading dataset from: {args.data_path}")
    train_dataset = BCDataset(args.data_path, val_ratio=args.val_ratio,
                               is_val=False, seed=args.seed,
                               use_augmentation=args.use_augmentation,
                               aug_prob=args.aug_prob)
    val_dataset = BCDataset(args.data_path, val_ratio=args.val_ratio,
                             is_val=True, seed=args.seed)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers,
                               pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)

    # 获取动作统计信息
    action_stats = train_dataset.get_action_stats()
    print(f"Action stats:")
    print(f"  mean: {action_stats['mean']}")
    print(f"  std: {action_stats['std']}")
    print(f"  min: {action_stats['min']}")
    print(f"  max: {action_stats['max']}")

    # 保存动作统计信息
    np.savez(output_dir / "action_stats.npz", **action_stats)

    # 创建模型
    model = BCPolicy(
        action_dim=args.action_dim,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 使用 ReduceLROnPlateau：当验证损失停止下降时降低学习率
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                   min_lr=1e-5, verbose=True)

    # 损失函数
    criterion = nn.MSELoss()

    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    # 早停参数
    early_stop_patience = args.early_stop_patience
    epochs_without_improvement = 0

    for epoch in range(args.num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        for batch in pbar:
            obs = batch['obs'].to(device)
            action = batch['action'].to(device)

            optimizer.zero_grad()
            pred_action = model(obs)
            loss = criterion(pred_action, action)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_steps += 1
            pbar.set_postfix({'loss': loss.item()})

        train_loss /= train_steps
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss = 0.0
        val_steps = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]"):
                obs = batch['obs'].to(device)
                action = batch['action'].to(device)

                pred_action = model(obs)
                loss = criterion(pred_action, action)

                val_loss += loss.item()
                val_steps += 1

        val_loss /= val_steps
        val_losses.append(val_loss)

        # 更新学习率（基于验证损失）
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{args.num_epochs}: "
              f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={current_lr:.6f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'action_stats': action_stats
            }, output_dir / "best_model.pth")
            print(f"  -> Saved best model (val_loss={val_loss:.6f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                print(f"\n早停: 验证损失连续 {early_stop_patience} 轮未改善")
                break

        # 定期保存检查点
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'action_stats': action_stats
            }, output_dir / f"checkpoint_epoch_{epoch+1}.pth")

    # 保存最终模型
    torch.save({
        'epoch': args.num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_losses[-1],
        'action_stats': action_stats
    }, output_dir / "final_model.pth")

    # 保存训练曲线
    np.savez(output_dir / "training_curves.npz",
             train_losses=train_losses,
             val_losses=val_losses)

    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description='Train BC Policy')

    # 数据参数
    parser.add_argument('--data_path', type=str,
                        default='log/sample/3d_auv_data/auv_data_final.zarr',
                        help='数据集路径')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--use_augmentation', action='store_true', default=False,
                        help='启用数据增强（镜像翻转来平衡左右偏差）')
    parser.add_argument('--aug_prob', type=float, default=0.5,
                        help='数据增强概率')

    # 模型参数
    parser.add_argument('--action_dim', type=int, default=4,
                        help='动作维度')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='使用预训练的 ResNet')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                        help='冻结 CNN backbone')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 128],
                        help='MLP 隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='MLP dropout 比例')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')

    # 其他参数
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='训练设备')
    parser.add_argument('--seed', type=int, default=1211,
                        help='随机种子')
    parser.add_argument('--output_dir', type=str, default='log/bc',
                        help='输出目录')
    parser.add_argument('--save_every', type=int, default=20,
                        help='每多少轮保存一次检查点')
    parser.add_argument('--early_stop_patience', type=int, default=30,
                        help='早停耐心值（验证损失连续多少轮不改善则停止）')

    args = parser.parse_args()
    train_bc(args)


if __name__ == '__main__':
    main()
