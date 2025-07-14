"""
AUV数据采集和训练的完整流程示例

展示从数据采集到模型训练的完整工作流程
"""

import numpy as np
import torch
from pathlib import Path
import sys
import os

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from auv_track_launcher.dataset.data_collector import AUVDataCollector
from auv_track_launcher.dataset.auv_tracking_dataset import AUVTrackingDataset


class AUVDataWorkflow:
    """
    AUV数据采集和训练的完整工作流程
    """
    
    def __init__(self, data_dir: str = "./auv_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据采集器
        self.collector = None
        
        # 训练数据集
        self.train_dataset = None
        self.val_dataset = None
        self.normalizer = None

    def phase1_collect_data(self, n_episodes: int = 100):
        """
        阶段1: 数据采集
        
        在这个阶段，使用ReplayBuffer实时收集AUV仿真或实验数据
        """
        print("=== 阶段1: 数据采集 ===")
        
        # 创建数据采集器
        self.collector = AUVDataCollector(
            save_dir=str(self.data_dir / "collected"),
            auto_save_episodes=10
        )
        
        # 模拟数据采集过程
        print(f"开始收集 {n_episodes} 个episodes...")
        
        for episode_idx in range(n_episodes):
            if episode_idx % 10 == 0:
                print(f"进度: {episode_idx}/{n_episodes}")
            
            # 开始新episode
            self.collector.start_new_episode()
            
            # 模拟episode数据收集
            episode_length = np.random.randint(200, 500)
            
            for step in range(episode_length):
                # 模拟观测数据（在实际使用中，这些来自环境）
                obs = self._generate_mock_observation()
                
                # 模拟控制动作（在实际使用中，这些来自控制器）
                action = self._generate_mock_action()
                
                # 模拟奖励和结束条件
                reward = np.random.random()
                done = (step == episode_length - 1)
                
                # 收集数据到ReplayBuffer
                self.collector.collect_step_data(obs, action, reward, done)
        
        # 保存收集的数据
        data_file = self.data_dir / "auv_training_data.zarr"
        self.collector.save_to_file(str(data_file))
        
        # 输出采集统计
        stats = self.collector.get_stats()
        print("\n数据采集完成:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return str(data_file)

    def phase2_create_training_dataset(self, data_file: str):
        """
        阶段2: 创建训练数据集
        
        从保存的数据文件创建PyTorch训练数据集
        """
        print("\n=== 阶段2: 创建训练数据集 ===")
        
        # 配置数据集参数
        dataset_config = {
            'data_path': data_file,
            'horizon': 8,           # 预测序列长度
            'pad_before': 2,        # 历史观测
            'pad_after': 0,         # 未来观测
            'val_ratio': 0.15,      # 验证集比例
            'image_size': (224, 224),
            'use_image': True,
            'use_sonar': True,
            'use_imu': True,
        }
        
        try:
            # 创建训练数据集
            self.train_dataset = AUVTrackingDataset(**dataset_config)
            self.val_dataset = self.train_dataset.get_validation_dataset()
            
            # 获取数据标准化器
            self.normalizer = self.train_dataset.get_normalizer()
            
            # 输出数据集信息
            stats = self.train_dataset.get_dataset_stats()
            print("训练数据集创建成功:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            print(f"训练集大小: {len(self.train_dataset)}")
            print(f"验证集大小: {len(self.val_dataset)}")
            
            # 测试数据加载
            if len(self.train_dataset) > 0:
                sample = self.train_dataset[0]
                print("\n样本数据结构:")
                print("观测数据:")
                for key, value in sample['obs'].items():
                    print(f"  {key}: {value.shape}")
                print(f"动作数据: {sample['action'].shape}")
            
            return True
            
        except Exception as e:
            print(f"数据集创建失败: {e}")
            return False

    def phase3_prepare_training(self):
        """
        阶段3: 准备训练
        
        创建数据加载器，保存配置等
        """
        print("\n=== 阶段3: 准备训练 ===")
        
        if self.train_dataset is None:
            print("错误: 训练数据集未创建")
            return False
        
        # 创建PyTorch数据加载器
        from torch.utils.data import DataLoader
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=16,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"训练加载器: {len(train_loader)} batches")
        print(f"验证加载器: {len(val_loader)} batches")
        
        # 保存标准化器
        normalizer_path = self.data_dir / "normalizer.pkl"
        import pickle
        with open(normalizer_path, 'wb') as f:
            pickle.dump(self.normalizer, f)
        print(f"标准化器已保存: {normalizer_path}")
        
        # 测试一个batch
        try:
            batch = next(iter(train_loader))
            print(f"\n测试batch加载:")
            print(f"观测数据:")
            for key, value in batch['obs'].items():
                print(f"  {key}: {value.shape}")
            print(f"动作数据: {batch['action'].shape}")
            
        except Exception as e:
            print(f"Batch加载测试失败: {e}")
            return False
        
        return train_loader, val_loader

    def _generate_mock_observation(self):
        """生成模拟观测数据"""
        return {
            'camera_image': np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8),
            'sonar_data': np.random.uniform(0, 50, 360).astype(np.float32),
            'auv_state': np.random.randn(12).astype(np.float32),
            'target_state': np.random.randn(5).astype(np.float32)
        }

    def _generate_mock_action(self):
        """生成模拟动作数据"""
        return np.random.uniform(-2, 2, 6).astype(np.float32)

    def run_complete_workflow(self, n_episodes: int = 50):
        """
        运行完整的数据工作流程
        """
        print("开始AUV数据采集和训练准备工作流程")
        print("=" * 50)
        
        # 阶段1: 数据采集
        data_file = self.phase1_collect_data(n_episodes)
        
        # 阶段2: 创建训练数据集
        if not self.phase2_create_training_dataset(data_file):
            print("工作流程失败: 数据集创建错误")
            return False
        
        # 阶段3: 准备训练
        result = self.phase3_prepare_training()
        if result is False:
            print("工作流程失败: 训练准备错误")
            return False
        
        train_loader, val_loader = result
        
        print("\n" + "=" * 50)
        print("✅ 完整工作流程执行成功!")
        print("\n现在可以开始模型训练:")
        print("1. 使用 train_loader 和 val_loader 进行训练")
        print("2. 使用 self.normalizer 进行数据标准化")
        print("3. 训练好的模型可以用于AUV控制")
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'normalizer': self.normalizer,
            'data_file': data_file
        }


def demonstrate_data_collection_vs_training():
    """
    演示数据采集和训练的区别
    """
    print("=" * 60)
    print("AUV数据采集 vs 训练数据集 - 概念说明")
    print("=" * 60)
    
    print("""
📊 数据采集阶段 (使用ReplayBuffer):
   目的: 在仿真/实验中实时收集数据
   工具: AUVDataCollector + ReplayBuffer
   输出: .zarr 或 .npz 文件
   特点: 
   - 实时增量式收集
   - 按episode组织数据
   - 支持大规模数据存储
   - 可以中断和恢复采集

🎯 训练阶段 (使用Dataset):
   目的: 为神经网络训练提供数据
   工具: AUVTrackingDataset (继承PyTorch Dataset)
   输入: .zarr 或 .npz 文件
   特点:
   - 按序列采样数据
   - 数据预处理和标准化
   - 支持批量加载
   - 训练/验证集划分

🔄 完整流程:
   仿真环境 → ReplayBuffer → 数据文件 → Dataset → DataLoader → 神经网络
   """)


if __name__ == "__main__":
    # 演示概念
    demonstrate_data_collection_vs_training()
    
    # 运行完整工作流程
    workflow = AUVDataWorkflow(data_dir="./demo_auv_data")
    result = workflow.run_complete_workflow(n_episodes=20)
    
    if result:
        print(f"\n🎉 演示完成! 数据已准备就绪，可以开始训练AUV控制策略。")
