# 基准分析工具

这个工具用于分析AUV跟踪任务的评估结果，计算关键性能指标。

## 支持的指标

1. **Success Rate (SR)**: 成功率
   - 定义：运行了总步长80%以上且没有发生碰撞的episode比例
   
2. **Mean Track Belief Error (MTBE)**: 平均跟踪信念误差
   - 定义：目标真实位置与belief估计位置之间的平均欧几里得距离
   
3. **Action Smoothness (AS)**: 动作平滑度
   - 定义：基于连续动作之间差异计算的平滑度指标，值越高表示越平滑

## 使用方法

### 1. 快速分析
```bash
cd examples/benchmark
python analyze_results.py --data_dir /path/to/your/json/files
```

### 2. 详细分析
```python
from examples.benchmark.benchmark_analyzer import BenchmarkAnalyzer

# 创建分析器
analyzer = BenchmarkAnalyzer("你的JSON文件目录")

# 生成报告
report = analyzer.generate_report(save_path="report.json")

# 绘制图表
analyzer.plot_metrics(save_dir="plots")
```

## 输入文件格式

工具期望的JSON文件格式（每个episode一个文件）：
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
    "reward": 1.5
  },
  ...
]
```

## 输出示例

```
=== 基准分析报告 ===
分析目录: /path/to/data
Episodes数量: 3

--- 基本统计 ---
总Episodes: 3
总步数: 2500
平均步数/Episode: 833.3

--- 关键指标 ---
Success Rate (SR): 0.667 (66.7%)
Mean Track Belief Error (MTBE): 0.3456
Action Smoothness (AS): 0.8234

--- 每Episode详情 ---
MTBE 标准差: 0.0123
AS 标准差: 0.0456
最好MTBE: 0.3200
最差MTBE: 0.3700
```

## 依赖包

```bash
pip install numpy pandas matplotlib
```

## 文件结构

- `benchmark_analyzer.py`: 主要的分析器类
- `analyze_results.py`: 简化的分析脚本
- `README.md`: 使用说明（本文件）
