# WorldBase3D 类使用说明

## 概述

`WorldBase3D` 是继承自 `WorldBase` 的 3D 目标跟踪环境基类，专门用于实现 3D 环境下的 AUV 目标跟踪任务。

## 主要特性

### 1. 继承优势
- **代码复用**：继承了 `WorldBase` 的所有基础功能
- **维护性**：基础逻辑统一管理
- **扩展性**：专注于 3D 特有功能的实现

### 2. 3D 特有功能

#### 动作空间扩展
- **LQR 控制**：支持 4D 航路点 (x, y, z, yaw)
- **PID 控制**：支持 3D 速度控制 (vx, vy, vz, yaw_rate)
- **深度控制**：可通过配置启用/禁用深度控制

#### 观测空间扩展
- **3D 测量**：距离 (range)、方位角 (azimuth)、俯仰角 (elevation)
- **3D 噪声模型**：支持 3D 测量噪声协方差矩阵
- **视野限制**：水平 FOV + 垂直 FOV (elevation_fov)

#### 状态空间扩展
- **6D 目标状态**：(x, y, z, vx, vy, vz)
- **3D 边界检查**：包含深度维度的边界检验
- **3D 碰撞检测**：考虑 3D 距离的碰撞检测

## 使用方法

### 1. 创建具体的 3D 环境类

```python
from auv_env.envs.base_3d import WorldBase3D
from gymnasium import spaces
import numpy as np

class MyWorld3D(WorldBase3D):
    def __init__(self, config, map, show):
        super().__init__(config, map, show)
    
    def set_limits(self):
        """设置 3D 环境的限制参数"""
        # 动作维度 (例如: r, theta, depth_change, yaw_change)
        self.action_dim = 4
        
        # 动作空间
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1]), 
            high=np.array([1, 1, 1, 1]), 
            dtype=np.float32
        )
        
        # 动作范围缩放
        self.action_range_scale = [
            self.config['agent']['sensor_r'],  # 最大距离
            np.pi,  # 最大角度
            2.0,    # 最大深度变化
            np.pi/4 # 最大偏航角变化
        ]
        
        # 观测空间 (根据具体需求调整)
        obs_dim = 64*64 + 4*self.num_targets + 6 + self.action_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(obs_dim,), dtype=np.float32
        )
        
        # 3D 目标限制
        self.target_limit_3d = np.array([
            [self.bottom_corner[0], self.top_corner[0]],  # x
            [self.bottom_corner[1], self.top_corner[1]],  # y  
            [self.bottom_corner[2], self.top_corner[2]],  # z
            [-2.0, 2.0],  # vx
            [-2.0, 2.0],  # vy
            [-1.0, 1.0]   # vz
        ])
        
        # 2D 限制 (向后兼容)
        self.target_limit = self.target_limit_3d[:4]  # x,y,vx,vy
    
    def update_every_tick(self, sensors):
        """每个时间步的更新"""
        # 更新图像缓冲区或其他传感器数据
        pass
    
    def get_reward(self, is_col, action):
        """计算 3D 环境的奖励"""
        # 基础碰撞惩罚
        if is_col:
            return -100, True, 0, 0
        
        # 3D 跟踪奖励计算
        reward = 0
        mean_nlogdetcov = 0
        std_nlogdetcov = 0
        
        for i in range(self.num_targets):
            # 基于 3D 距离的奖励
            target_pos = self.targets[i].state.vec[:3]
            agent_pos = self.agent.state.vec[:3]
            distance_3d = np.linalg.norm(target_pos - agent_pos)
            
            # 距离奖励
            if distance_3d < self.config['agent']['sensor_r']:
                reward += 10 / (1 + distance_3d)
            
            # 不确定性惩罚 (基于 3D 协方差)
            cov_det = np.linalg.det(self.belief_targets[i].cov)
            nlogdetcov = -np.log(cov_det + 1e-8)
            reward -= 0.1 * nlogdetcov
            mean_nlogdetcov += nlogdetcov
        
        mean_nlogdetcov /= self.num_targets
        return reward, False, mean_nlogdetcov, std_nlogdetcov
    
    def state_func(self, observed, action):
        """构建 3D 状态向量"""
        state = []
        
        # 网格地图
        state.extend(self.agent.gridMap.to_grayscale_image().flatten())
        
        # 3D 目标信息
        for i in range(self.num_targets):
            # 3D 相对位置
            rel_pos = self.belief_targets[i].state[:3] - self.agent.est_state.vec[:3]
            r = np.linalg.norm(rel_pos)
            alpha = np.arctan2(rel_pos[1], rel_pos[0]) - np.radians(self.agent.est_state.vec[8])
            beta = np.arctan2(rel_pos[2], np.linalg.norm(rel_pos[:2]))
            
            state.extend([r, alpha, beta, 
                         np.log(np.linalg.det(self.belief_targets[i].cov) + 1e-8),
                         float(observed[i])])
        
        # 3D 智能体状态
        state.extend([
            self.agent.state.vec[0],  # x
            self.agent.state.vec[1],  # y  
            self.agent.state.vec[2],  # z
            np.radians(self.agent.state.vec[8]),  # yaw
            self.agent.state.vec[3],  # vx
            self.agent.state.vec[4]   # vy
        ])
        
        # 动作
        if hasattr(action, 'tolist'):
            state.extend(action.tolist())
        else:
            state.extend([action.linear.x, action.linear.y, action.linear.z, action.angular.z])
        
        return np.array(state, dtype=np.float32)
```

### 2. 配置文件调整

在配置文件中添加 3D 相关参数：

```yaml
agent:
  sensor_r: 20.0
  fov: 120  # 水平视野
  elevation_fov: 60  # 垂直视野 (新增)
  sensor_r_sd: 0.2
  sensor_b_sd: 0.1
  sensor_e_sd: 0.1  # 俯仰角噪声 (新增)

env:
  enable_depth_control: true  # 启用深度控制

target:
  target_init_cov_3d: 100  # 3D 初始协方差 (6x6 矩阵的对角线值)
  depth_range_a2t: [-2, 2]  # 智能体到目标的深度范围
  depth_range_t2b: [-1, 1]  # 目标到信念的深度范围
```

### 3. 创建环境实例

```python
from auv_env.envs.base import TargetTrackingBase

class MyTracking3DEnv(TargetTrackingBase):
    def __init__(self, config, map='AUVTracking3D-v0', show_viewport=False):
        super().__init__(MyWorld3D, map, show_viewport, config)

# 使用
config = load_config('config_3d.yaml')
env = MyTracking3DEnv(config)
obs, info = env.reset()

while True:
    action = env.action_space.sample()  # 或使用训练好的策略
    obs, reward, done, truncated, info = env.step(action)
    if done:
        break
```

## 主要改进点

1. **继承结构**：避免代码重复，提高维护性
2. **3D 扩展**：支持深度控制和 3D 观测
3. **灵活配置**：通过配置文件控制 3D 特性
4. **向后兼容**：保持与 2D 版本的接口一致性

## 注意事项

1. 确保配置文件包含所有必要的 3D 参数
2. 根据具体应用调整状态空间和动作空间的维度
3. 3D 环境的计算复杂度更高，注意性能优化
4. 测试时确保 3D 边界检查和碰撞检测正常工作
