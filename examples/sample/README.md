# AUV Tracking - Sample Examples

本文件夹包含了用于演示AUV追踪项目核心功能的示例脚本，主要用于数据采集、模型测试和视频制作。

## 📁 文件结构

```
examples/sample/
├── README.md              # 本说明文件
├── simple_collect.py      # 简单数据采集脚本
├── sample_env_test.py     # 环境和模型测试脚本
├── test_simple.py         # 数据集加载测试脚本
└── make_video.py          # 视频制作工具
```

## 🚀 快速开始

### 环境设置

确保你已经安装了所有必要的依赖并正确配置了环境：

```bash
cd /home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking
```

## 📋 文件功能详解

### 1. `simple_collect.py` - 数据采集脚本

**功能**: 使用随机策略或预训练模型在AUV追踪环境中采集训练数据。

**主要特性**:
- 支持多回合数据采集
- 自动保存采集的数据到zarr格式
- 支持可视化环境运行过程
- 集成数据收集器(AUVCollector)

**使用方法**:
```bash
python examples/sample/simple_collect.py
```

**配置文件**: `configs/envs/v1_config.yml`

**输出**: 采集的数据保存到 `simple_auv_data/` 目录

### 2. `sample_env_test.py` - 环境和模型测试

**功能**: 测试预训练的强化学习模型在AUV追踪环境中的表现。

**主要特性**:
- 加载预训练的SAC模型
- 使用StateOnlyWrapper包装环境
- 可视化模型执行过程
- 支持不同的环境配置

**使用方法**:
```bash
python examples/sample/sample_env_test.py
```

**依赖模型**: 需要预训练的SAC模型文件 (如: `rl_model_2000000_steps.zip`)

**配置文件**: `configs/envs/v1_sample_config.yml`

### 3. `test_simple.py` - 数据集测试脚本

**功能**: 测试和验证采集的数据集是否能正确加载和使用。

**主要特性**:
- 验证zarr数据格式的完整性
- 测试AUVTrackingDataset数据集加载器
- 检查数据维度和结构
- 支持数据集分割(训练/验证)

**使用方法**:
```bash
python examples/sample/test_simple.py
```

**数据路径**: `/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/log/sample/auv_data/auv_data_final.zarr`

### 4. `make_video.py` - 视频制作工具

**功能**: 将采集的帧序列制作成视频文件。

**主要特性**:
- 支持PNG帧序列转换为MP4视频
- 可调节帧率
- 自动排序和处理帧文件

**使用方法**:
```bash
python examples/sample/make_video.py
```

## 🎯 典型使用流程

### 1. 数据采集流程
```bash
# 1. 采集训练数据
python examples/sample/simple_collect.py

# 2. 验证数据集
python examples/sample/test_simple.py

# 3. (可选) 制作演示视频
python examples/sample/make_video.py
```

### 2. 模型测试流程
```bash
# 测试预训练模型
python examples/sample/sample_env_test.py
```

## ⚙️ 配置说明

### 环境配置文件

- **v1_config.yml**: 用于数据采集的标准环境配置
- **v1_sample_config.yml**: 用于模型测试的采样环境配置

主要配置项：
```yaml
name: AUVTracking_v1_sample        # 环境名称
map: 'SimpleUnderwater-Bluerov2_RGB'  # 仿真地图
agent:
  controller: 'PID'               # 控制器类型
scenario:
  size: [40, 40, 20]             # 环境尺寸
  num_targets: 1                 # 目标数量
```

## 📊 数据格式

### 采集的数据结构
```python
{
    'obs': {
        'camera_image': (H, W, C),    # 相机图像
        'state': (state_dim,),        # 状态信息
    },
    'action': (action_dim,),          # 动作数据
}
```

### 支持的数据键
- `camera_image`: RGB相机图像数据
- `state`: AUV和目标的状态信息
- `action`: 控制动作

## 🔧 故障排除

### 常见问题

1. **环境初始化失败**
   - 检查HoloOcean是否正确安装
   - 确认GPU驱动和CUDA版本兼容性

2. **模型加载错误**
   - 验证模型文件路径是否正确
   - 检查模型与环境的动作/观测空间是否匹配

3. **数据保存问题**
   - 确保有足够的磁盘空间
   - 检查文件写入权限

### 调试建议

1. **开启详细日志**:
   ```python
   config['debug'] = True
   ```

2. **可视化调试**:
   ```python
   show_viewport = True
   ```

3. **减少数据量**:
   ```python
   n_episodes = 1  # 先测试单个回合
   ```

## 📝 注意事项

1. **资源消耗**: 数据采集和可视化会消耗较多GPU和内存资源
2. **存储空间**: 图像数据占用空间较大，注意磁盘容量
3. **版本兼容**: 确保所有依赖库版本兼容

## 🤝 贡献

如果你发现任何问题或有改进建议，请提交issue或pull request。

## 📄 相关文档

- [主项目README](../../README.md)
- [环境配置说明](../../configs/README.md)
- [数据集文档](../../auv_track_launcher/dataset/README.md)
