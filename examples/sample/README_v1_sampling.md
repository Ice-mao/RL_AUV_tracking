# 在AUVTracking_v1中使用v0训练的SAC策略进行采样

## 概述

本指南说明如何使用在`AUVTracking_v0`环境中训练的SAC策略，在`AUVTracking_v1`环境中进行episode采样。

## 使用方法

### 1. 基本使用

```bash
cd /home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking

python examples/sample/sample_v1_with_v0_policy.py \
    --model_path log/AUVTracking_v0/PID/SAC/12-20_01/rl_model_999990_steps.zip \
    --env_config configs/envs/v1_config.yml \
    --alg_config configs/algorithm/sac.yml \
    --n_episodes 50 \
    --save_dir log/sample/v1_episodes
```

### 2. 先分析截断合理性再采样

```bash
python examples/sample/sample_v1_with_v0_policy.py \
    --model_path log/AUVTracking_v0/PID/SAC/12-20_01/rl_model_999990_steps.zip \
    --env_config configs/envs/v1_config.yml \
    --alg_config configs/algorithm/sac.yml \
    --analyze \
    --n_episodes 50
```

### 3. 参数说明

- `--model_path`: SAC模型路径（从`sac.yml`的`resume_path`获取）
- `--env_config`: v1环境配置文件路径（默认：`configs/envs/v1_config.yml`）
- `--alg_config`: 算法配置文件路径（默认：`configs/algorithm/sac.yml`）
- `--n_episodes`: 要采样的episode数量（默认：50）
- `--save_dir`: 数据保存目录（默认：`log/sample/v1_episodes`）
- `--min_length`: episode最小长度，小于此值的episode将被舍弃（默认：300）
- `--truncate_tail`: 截断尾部步数，有效episode会舍弃最后这么多步（默认：100）
- `--show_viewport`: 是否显示可视化
- `--analyze`: 先分析episode截断合理性（运行10个测试episodes）
- `--deterministic`: 是否使用确定性策略（默认：True，评估模式）

## Episode截断机制分析

### 1. 环境层面的截断（TimeLimit）

在`auv_env/__init__.py`中，环境被`gym.wrappers.TimeLimit`包装：

```python
env = gym.wrappers.TimeLimit(env0, max_episode_steps=t_steps)
```

- **v1配置中的t_steps**: 1000步（在`v1_config.yml`中设置）
- **截断行为**: 当episode达到1000步时，环境返回`truncated=True`
- **与terminated的区别**:
  - `terminated=True`: 任务自然完成（如成功跟踪、碰撞等）
  - `truncated=True`: 达到最大步数限制，任务被强制结束

### 2. 数据收集层面的截断（AUVCollector）

在`auv_track_launcher/dataset/data_collector.py`中：

- **min_length**: 过滤掉过短的episodes（默认300步）
  - 原因：过短的episodes可能表示任务失败或初始化问题
  - 合理性：✅ 合理，可以过滤掉无效数据

- **truncate_tail**: 截断episode尾部（默认100步）
  - 原因：episode末尾可能包含跟踪效果不好的部分
  - 合理性：⚠️ **需要根据实际情况调整**

### 3. 截断合理性评估

#### ✅ 合理的截断情况：

1. **TimeLimit截断（t_steps=1000）**:
   - 如果平均episode长度在800-1000步之间，说明设置合理
   - 如果大部分episodes在500步内完成，可以考虑减小t_steps以加快训练

2. **min_length过滤（300步）**:
   - 如果大部分有效episodes长度>300步，设置合理
   - 可以过滤掉初始化失败或快速失败的episodes

#### ⚠️ 需要调整的情况：

1. **truncate_tail（100步）可能过于激进**:
   - 如果episode平均长度只有400-500步，截断100步意味着丢失20-25%的数据
   - **建议**: 
     - 如果episode平均长度>800步，100步截断合理（约12.5%）
     - 如果episode平均长度<600步，建议减小到50步或更少
     - 可以通过分析episode末尾的奖励/跟踪误差来验证是否需要截断

2. **t_steps设置**:
   - 如果>50%的episodes被截断，说明t_steps太小
   - 如果<10%的episodes被截断，且平均长度远小于t_steps，可以考虑减小以加快训练

## 当前实现分析

### 当前配置（v1_config.yml）:
- `t_steps: 1000` - episode最大步数

### 当前数据收集设置（示例代码）:
- `min_length: 300` - 最小episode长度
- `truncate_tail: 100` - 尾部截断步数

### 建议的调整策略：

1. **先运行分析模式**:
   ```bash
   python examples/sample/sample_v1_with_v0_policy.py --analyze
   ```
   这会运行10个测试episodes，分析截断比例和平均长度。

2. **根据分析结果调整**:
   - 如果截断比例>50%: 增加`t_steps`到1500或2000
   - 如果平均长度<600步: 减小`truncate_tail`到50步
   - 如果平均长度>800步: 当前`truncate_tail=100`合理

3. **验证截断效果**:
   - 检查被截断的episode末尾的跟踪误差
   - 如果末尾跟踪误差确实较大，截断合理
   - 如果末尾跟踪效果良好，可以减少或取消截断

## 代码关键点

### 1. 观察空间处理

由于v0和v1的观察空间可能不同，需要使用`StateOnlyWrapper`：

```python
wrapped_env = StateOnlyWrapper(env)
model = SAC.load(model_path, device='cuda', env=wrapped_env)
```

这确保模型接收的是state观察（与v0训练时一致）。

### 2. Episode循环

```python
obs, info = env.reset()
while True:
    action, _ = model.predict(obs['state'], deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

- 同时检查`terminated`和`truncated`
- `truncated=True`表示达到最大步数限制
- `terminated=True`表示任务自然完成

### 3. 数据收集

```python
collector.start_episode()
# ... 收集数据 ...
flag = collector.finish_episode()  # 返回True表示episode有效
```

`finish_episode()`会：
1. 检查episode长度是否>=min_length
2. 截断尾部truncate_tail步
3. 保存有效数据

## 常见问题

### Q1: 为什么需要使用StateOnlyWrapper？

A: v0训练的SAC模型期望接收state观察，而v1环境可能返回包含image和state的字典。StateOnlyWrapper提取state部分，确保模型输入格式一致。

### Q2: 如何判断truncate_tail是否合理？

A: 
1. 运行分析模式查看平均episode长度
2. 检查被截断部分的跟踪误差（如果末尾误差大，截断合理）
3. 如果平均长度较短（<600步），建议减小truncate_tail

### Q3: 如果大部分episodes被截断怎么办？

A: 
1. 增加`t_steps`（在v1_config.yml中）
2. 检查策略性能（可能策略在v1环境中表现不佳）
3. 考虑重新训练或微调策略

### Q4: 采样数据保存在哪里？

A: 默认保存在`log/sample/v1_episodes/auv_data_final.zarr`，使用zarr格式存储。

## 示例输出

```
============================================================
加载配置...
环境配置: AUVTracking_v1
最大episode步数 (t_steps): 1000
模型路径: log/AUVTracking_v0/PID/SAC/12-20_01/rl_model_999990_steps.zip
============================================================

创建环境...

加载SAC模型...
✓ 模型加载成功

初始化数据收集器...
  - 最小episode长度: 300
  - 尾部截断步数: 100

开始采样 50 个episodes...
============================================================

Episode 1/50
  Episode截断 (truncated=True) at step 1000 (达到最大步数 1000)
  ✓ Episode有效，长度: 1000 步
...
```

