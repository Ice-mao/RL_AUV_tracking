# v1环境评估配置文件

## 文件说明

### v1_config_dam.yml
- **地图**: `Dam-Bluerov2_RGB`
- **场景大小**: [100, 100, 30]
- **底部角落**: [-250, -50, -70]
- **固定深度**: -5
- **用途**: 在Dam场景中评估v1环境的模型性能

### v1_config_openwater.yml
- **地图**: `OpenWater-Bluerov2_RGB`
- **场景大小**: [500, 500, 50]
- **底部角落**: [200, 0, -300]
- **固定深度**: -5
- **用途**: 在OpenWater场景中评估v1环境的模型性能

## 使用方法

### 运行Dam场景评估
```bash
python run_benchmark_eval.py \
  --rl_model_path log/AUVTracking_v1/LQR/CustomSACPolicy/rl_model_2549898_steps.zip \
  --env_config configs/envs/eval/v1_config_dam.yml \
  --alg_config configs/algorithm/sac_v1.yml \
  --num_episodes 100
```

### 运行OpenWater场景评估
```bash
python run_benchmark_eval.py \
  --rl_model_path log/AUVTracking_v1/LQR/CustomSACPolicy/rl_model_2549898_steps.zip \
  --env_config configs/envs/eval/v1_config_openwater.yml \
  --alg_config configs/algorithm/sac_v1.yml \
  --num_episodes 100
```

### 分析结果
```bash
# Dam场景分析
python examples/benchmark/benchmark_analyzer.py \
  --data_dir log/benchmark/RL/[dam时间戳目录] \
  --save_report log/benchmark/RL/[dam时间戳目录]/report.json \
  --save_plots log/benchmark/RL/[dam时间戳目录] \
  --min_steps 150

# OpenWater场景分析
python examples/benchmark/benchmark_analyzer.py \
  --data_dir log/benchmark/RL/[openwater时间戳目录] \
  --save_report log/benchmark/RL/[openwater时间戳目录]/report.json \
  --save_plots log/benchmark/RL/[openwater时间戳目录] \
  --min_steps 150
```

## 配置说明

两个配置文件都基于 `v1_config.yml`，主要差异：

1. **地图**: 从 `SimpleUnderwater-Bluerov2_RGB` 改为对应的场景地图
2. **场景大小和位置**: 根据Dam/OpenWater场景调整
3. **其他参数**: 保持与v1_config.yml一致（2D环境配置）

## 注意事项

- v1环境是2D环境，`target_dim: 4`（不是3D的6）
- `fix_depth`是单个值（-5），不是范围
- 控制器配置保持2D格式（LQR action_dim: 3）

