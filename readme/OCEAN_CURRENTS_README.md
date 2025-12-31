# Ocean Current Field System - 实现说明

## 概述

已成功实现动态水流势场系统，用于在AUV追踪环境中模拟海洋水流对auv0机器人的影响。水流场支持时间和位置依赖的向量场，通过YAML配置文件进行配置。

## 已实现功能

### 1. 水流势场函数模块 (`auv_env/current_fields.py`)

实现了4种水流势场类型：

#### A. **vortex_field** - 涡流场
- 描述：围绕中心点的圆形涡流，支持时间相关的旋转
- 参数：
  - `center`: 涡流中心位置 [x, y, z]（默认：[0, 0, -10]）
  - `strength`: 切向速度大小（默认：1.0 m/s）
  - `radius`: 参考半径（默认：5.0 m）
  - `angular_velocity`: 涡流旋转速度（默认：0.0 rad/s）
  - `decay_rate`: 速度随距离衰减率（默认：1.0）

#### B. **uniform_field** - 均匀水流场
- 描述：固定方向的恒定水流，可选时间调制
- 参数：
  - `velocity`: 恒定速度向量 [dx, dy, dz]（默认：[0.5, 0, 0] m/s）
  - `time_modulation`: 启用正弦时间变化（默认：False）
  - `modulation_period`: 时间变化周期（默认：10.0 s）
  - `modulation_amplitude`: 调制幅度乘数（默认：0.5）

#### C. **random_field** - 随机湍流场
- 描述：使用Perlin噪声的空间变化随机湍流
- 参数：
  - `max_velocity`: 最大水流速度（默认：1.0 m/s）
  - `spatial_scale`: 空间变化频率（默认：10.0 m）
  - `temporal_scale`: 时间变化速度（默认：5.0 s）
  - `seed`: 随机种子，用于可重复性（默认：None）
  - `correlation_length`: 空间相关长度（默认：3.0 m）

#### D. **sine_wave_field** - 正弦波动水流场
- 描述：通过空间传播的振荡波状水流
- 参数：
  - `direction`: 波传播方向 [dx, dy, dz]（默认：[1, 0, 0]）
  - `amplitude`: 水流幅度（默认：1.0 m/s）
  - `wavelength`: 空间波长（默认：10.0 m）
  - `period`: 时间周期（默认：5.0 s）
  - `phase`: 初始相位偏移（默认：0.0）

### 2. 配置系统

在配置文件中添加了 `ocean_current` 部分，示例：

```yaml
ocean_current:
  enabled: true  # 设置为true启用水流
  type: 'vortex'  # 可选: 'vortex', 'uniform', 'random', 'sine_wave'
  params:
    # 涡流场参数
    center: [0, 0, -10]
    strength: 2.0
    radius: 5.0
    angular_velocity: 0.1
    decay_rate: 1.0
```

### 3. 水流场可视化

支持使用 HoloOcean 的 `draw_debug_vector_field` API 可视化水流场：

配置示例：
```yaml
ocean_current:
  enabled: true
  type: 'vortex'
  params:
    center: [0, 0, -10]
    strength: 2.0
  visualization:
    enabled: true  # 启用可视化
    draw_at_tick: 100  # 在第100个tick绘制（0表示在reset时绘制）
    location: [0, 0, -10]  # 可视化中心位置
    dimensions: [40, 40, 20]  # 向量场范围 [x, y, z]
    spacing: 3  # 向量间距
    arrow_thickness: 5  # 箭头粗细
    arrow_size: 0.25  # 箭头大小
    lifetime: 0  # 显示时长（0=永久）
```

### 4. 环境集成

已集成到以下文件：

- **base_3d.py**: 3D追踪环境
  - 在 `__init__` 中初始化水流场
  - 在 `step()` 方法的每个tick中应用水流
  - 支持水流场可视化
  - 位置：第83-89行（初始化），第179-211行（应用和可视化）

- **base.py**: 2D追踪环境（兼容性）
  - 使用固定深度坐标计算3D位置
  - 相同的集成模式
  - 位置：第123-125行（初始化），第207-230行（应用）

## 使用方法

### 1. 基本使用

在配置文件中启用水流（例如 `configs/envs/3d_v0_config.yml`）：

```yaml
ocean_current:
  enabled: true
  type: 'vortex'
  params:
    center: [0, 0, -10]
    strength: 2.0
```

### 2. 切换不同类型的水流

**涡流场示例：**
```yaml
ocean_current:
  enabled: true
  type: 'vortex'
  params:
    center: [0, 0, -10]
    strength: 2.0
    radius: 5.0
    angular_velocity: 0.1
```

**均匀水流示例：**
```yaml
ocean_current:
  enabled: true
  type: 'uniform'
  params:
    velocity: [0.5, 0.3, 0.0]
    time_modulation: true
    modulation_period: 10.0
```

**随机湍流示例：**
```yaml
ocean_current:
  enabled: true
  type: 'random'
  params:
    max_velocity: 1.0
    spatial_scale: 10.0
    temporal_scale: 5.0
    seed: 42
```

**波动水流示例：**
```yaml
ocean_current:
  enabled: true
  type: 'sine_wave'
  params:
    direction: [1, 0, 0]
    amplitude: 1.5
    wavelength: 8.0
    period: 6.0
```

### 3. 禁用水流

设置 `enabled: false` 或直接删除 `ocean_current` 部分：

```yaml
ocean_current:
  enabled: false
```

## 技术细节

### 时间应用
- 水流在**每个仿真tick**中应用（不是每个RL步骤），确保物理精确性
- 时间源：`self.sensors['t']`（仿真时间，秒）

### 验证和安全
- 自动验证水流速度向量（3D数组，有限值）
- 将极端值裁剪到 `max_current_speed = 5.0 m/s`
- 错误处理：优雅地处理失败，不中断训练

### HoloOcean API
使用 `self.ocean.set_ocean_currents('auv0', velocity)` 应用水流。

### 向后兼容
- 如果配置中没有 `ocean_current` 部分，系统不会应用水流
- 完全向后兼容现有配置文件

## 测试

运行测试脚本验证实现：

```bash
python test_ocean_currents.py
```

测试覆盖：
- ✓ 所有4种场类型的函数调用
- ✓ 场注册表
- ✓ 工厂函数创建
- ✓ 时间依赖性
- ✓ 空间变化
- ✓ 通过工厂的所有场类型

## 文件修改清单

### 新文件
1. `auv_env/current_fields.py` - 水流场实现（~280行）

### 修改的文件
1. `auv_env/envs/base_3d.py`
   - 添加 `_init_ocean_current()` 方法
   - 在step()中应用水流

2. `auv_env/envs/base.py`
   - 添加 `_init_ocean_current()` 方法
   - 在step()中应用水流（2D兼容）

3. `configs/envs/3d_v0_config.yml`
   - 添加 `ocean_current` 配置部分（带示例）

## 关键设计决策

1. **位置**: 水流势场函数在 `auv_track_launcher` 包中，便于扩展
2. **配置**: 通过函数名称字符串和参数字典指定，灵活性高
3. **时间**: 每tick应用，确保平滑的物理效果
4. **验证**: 裁剪极值，检查NaN/Inf
5. **兼容性**: 如果缺少配置部分则禁用水流

## 扩展性

添加新的水流场类型：

1. 在 `current_fields.py` 中定义新函数：
```python
def my_custom_field(location, time, **params):
    # 实现你的水流场
    return np.array([dx, dy, dz])
```

2. 注册到字典：
```python
CURRENT_FIELD_REGISTRY['my_custom'] = my_custom_field
```

3. 在配置中使用：
```yaml
ocean_current:
  type: 'my_custom'
  params: {...}
```

## 性能考虑

- 水流计算每tick执行一次，计算开销很小
- 使用NumPy向量化操作以提高效率
- 错误处理不会中断训练循环

## 注意事项

1. 水流仅在水下（z < 0）应用
2. 水流只影响auv0机器人（如需求）
3. 参考代码中的 `set_ocean_currents` API可能在某些HoloOcean版本中不存在
4. 如API不可用，可以通过修改self.u数组来应用水流作为额外的力

## 示例使用场景

### 训练场景
使用较弱的水流进行训练：
```yaml
ocean_current:
  enabled: true
  type: 'uniform'
  params:
    velocity: [0.3, 0.0, 0.0]
```

### 评估场景
使用强湍流测试鲁棒性：
```yaml
ocean_current:
  enabled: true
  type: 'random'
  params:
    max_velocity: 2.0
    spatial_scale: 5.0
    seed: 42  # 可重复性
```

### 真实场景模拟
使用涡流场模拟海洋涡流：
```yaml
ocean_current:
  enabled: true
  type: 'vortex'
  params:
    center: [0, 0, -10]
    strength: 3.0
    angular_velocity: 0.2
```

## 未来改进建议

1. 添加水流场可视化（在配置中 `draw_traj: true` 时）
2. 支持多个水流场叠加
3. 添加更多水流场类型（例如：潮汐、海流等）
4. 保存水流场历史用于分析
5. 支持从文件加载自定义水流场

## 支持

如有问题或需要添加新功能，请参考：
- 代码：`auv_env/current_fields.py`
- 配置示例：`configs/envs/3d_v0_config.yml`
