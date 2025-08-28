# HER (Hindsight Experience Replay) 配置指南

## HER 原理

HER (Hindsight Experience Replay) 是专门为目标导向的强化学习任务设计的技术。

### 核心思想
在许多强化学习任务中（如机器人抓取、导航），智能体很难在早期训练中达到目标，导致奖励稀疏的问题。HER 通过"事后诸葛亮"的方式，将失败的经验重新解释为成功的经验。

**工作原理**：即使智能体没有达到原始目标，我们也可以假设它实际达到的位置就是目标，这样就能从失败中学习。

## HER 参数详解

### 1. `n_sampled_goal`
- **含义**：每个真实转换生成多少个额外的虚拟转换
- **推荐值**：4（经验值，平衡性能和计算开销）
- **示例**：设置为4意味着每收集1个真实经验，就生成4个额外的重新标记的虚拟经验

### 2. `goal_selection_strategy`
目标选择策略，决定如何为重新标记选择新目标：

- **`"future"`** (推荐)：从同一回合的未来状态中随机选择
  - 最常用，效果最好
  - 利用了因果关系：未来状态是当前行为可能达到的
  
- **`"final"`**：使用回合的最终状态作为目标
  - 简单但可能效果较差
  
- **`"episode"`**：从整个回合的状态中随机选择
  - 包含过去和未来状态
  
- **`"random"`**：从整个缓冲区中随机选择目标
  - 可能选择不相关的目标

## 配置示例

### 标准 SAC（默认）
```yaml
replay_buffer:
  type: "ReplayBuffer"
```

### 使用 HER 的 SAC
```yaml
replay_buffer:
  type: "HerReplayBuffer"
  her_kwargs:
    n_sampled_goal: 4
    goal_selection_strategy: "future"
```

## 使用方法

```bash
# 标准 SAC
python train.sh --alg_config configs/algorithm/sac.yml

# 使用 HER 的 SAC
python train.sh --alg_config configs/algorithm/sac_her.yml
```

## 适用场景

HER 特别适合以下类型的任务：
1. **目标导向任务**：有明确目标状态的任务
2. **稀疏奖励环境**：很难获得正奖励的环境
3. **连续控制任务**：如机器人控制、导航等

## 环境要求

使用 HER 的环境通常需要：
1. **MultiInputPolicy**：支持目标观测的策略
2. **Goal-conditioned**：环境需要支持目标条件设置
3. **Compute Reward**：环境需要实现 `compute_reward()` 方法

## 代码实现

核心实现参考 `SB3_trainer.py` 中的 `get_her_replay_buffer_kwargs()` 函数，直接使用 stable-baselines3 的 `HerReplayBuffer` 类。
