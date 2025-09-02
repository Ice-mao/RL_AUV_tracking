# 多算法评估工具使用说明

这个工具支持对三种不同的算法进行评估：
1. **强化学习算法** (RL) - SAC/PPO等
2. **贪心算法** (Greedy) - 基于信息增益的贪心策略
3. **扩散策略** (Diffusion Policy) - 扩散模型生成的动作策略

## 使用方法

### 1. 评估单个算法

```bash
# 评估强化学习算法
python examples/benchmark/evaluate.py --mode rl --rl_model_path your_model.zip --num_episodes 5

# 评估Greedy算法
python examples/benchmark/evaluate.py --mode greedy --num_episodes 5

# 评估Diffusion Policy算法（需要实现具体的模型加载）
python examples/benchmark/evaluate.py --mode diffusion --diffusion_model_path your_diffusion_model.pth --num_episodes 5
```

### 2. 评估所有算法

```bash
python examples/benchmark/evaluate.py --mode all --rl_model_path your_rl_model.zip --diffusion_model_path your_diffusion_model.pth --num_episodes 5
```

## 参数说明

- `--mode`: 评估模式，可选值：`rl`, `greedy`, `diffusion`, `all`
- `--rl_model_path`: 强化学习模型文件路径（.zip格式）
- `--diffusion_model_path`: Diffusion Policy模型文件路径
- `--env_config`: 环境配置文件路径，默认为 `configs/envs/3d_v0_config.yml`
- `--alg_config`: 算法配置文件路径，默认为 `configs/algorithm/sac_3d_v0.yml`
- `--num_episodes`: 评估的回合数，默认为3

## 输出文件

每种算法会为每个episode生成一个JSON文件：
- `rl_episode_1_data_20250901_173000.json` - 强化学习算法结果
- `greedy_episode_1_data_20250901_173000.json` - 贪心算法结果  
- `diffusion_episode_1_data_20250901_173000.json` - 扩散策略结果

## 数据格式

每个JSON文件包含该episode所有步骤的信息：
```json
[
  {
    "action": [0.1, 0.2, 0.3, 0.4],
    "is_collision": false,
    "done": false,
    "agent_pos": [1.0, 2.0, 3.0],
    "targets": [4.0, 5.0, 6.0],
    "belief_targets": [4.1, 5.1, 6.1],
    "step": 1,
    "episode": 1,
    "algorithm": "RL",
    "reward": 1.5
  }
]
```

## 配合基准分析使用

生成的数据文件可以直接用于基准分析：

```bash
# 分析某种算法的结果
python examples/benchmark/analyze_results.py --data_dir . 

# 这将分析当前目录下所有的JSON文件并计算性能指标
```

## 扩展Diffusion Policy

当前Diffusion Policy使用随机动作作为占位符。要使用真实的diffusion模型，请修改 `evaluate_diffusion_and_save()` 函数中的以下部分：

```python
# 替换这行：
action = env.action_space.sample()

# 改为：
action = your_diffusion_model.predict(obs)
```

## 注意事项

1. 确保相应的模型文件存在且可访问
2. Greedy算法需要 `auv_baseline.greedy` 模块
3. 环境配置文件需要正确设置
4. 建议在评估前先测试单个episode确保环境正常工作
