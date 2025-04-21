# RL_AUV_tracking

A project for my major related graduation paper.

Using reinforcement learning(RL) to train an agent to tracking the target in the unknown underwater scenario in HoloOcean.

# Quick Start

To run the simulation, first install all dependencies

- HoloOcean==1.0.0
- Stable Baseline3
- pynput
- bezier
- filterpy
- inekf
- scipy
- sb3-contrib
- seaborn
- shapely

if you want to run in my scenario,I give the scenario link below:
https://drive.google.com/drive/folders/1MdT8NMozJARde7zL5kKebBi4WULfv2KC?usp=drive_link

you should copy the folder into /home/'yourname'/.local/share/holoocean/0.5.0/worlds/

Then simply run the script
```
python SB3_learning.py --env TargetTracking1 --map TestMap_AUV --nb_envs 5 --choice 0 --render 0 
```
choice:(0:train 1:keep training 2:eval)

render:(0:false 1:true)

## Env Setup
### world_auv_rgb
obs: dict type, including 
images[(3,224,224)*2] and state[(10)]

action:[(3)]

## Simulation Process

![simulation](config/simulation.png)

## Additional information

If you want to know more details,you should read the code.
:smile: 

Or please keep staying tuning!

## Something mentioned

Just for single target, mutitarget task needs revise the code

(revise the target0 -> target+str(rank))

# Env List
## edtion
v0:traditional work
v1:for RGB work, LQR controller
v2:for RGB、 sonar work, PID controller

We show the basic env can be used and their function:

| 环境ID | 渲染模式 | 功能描述 |
|--------|----------|---------|
| Teacher-v0 | 默认 | state中包含全局定位真值信息 |
| Teacher-v1 | 默认 | state中不包含全局定位真值信息 |
| Teacher-v1-norender | 无渲染 | 可用于服务器快速训练 |
| Teacher-v1-render | 渲染，2D场景展示 | 渲染场景的Teacher-v1，用于可视化演示以及调试 |
| Student-v0 | 默认 | state为图像信息 |
| Student-v0-norender | 无渲染 | 可用于服务器快速训练 |
| Student-v0-sample | 默认 | 学生模型采样环境，用于评估和测试 |
| Student-v0-sample-teacher | 默认 | 将教师策略应用于学生采样环境，用于比较性能 |

## 环境使用建议

- **训练阶段**: 使用 `-norender` 版本环境提高训练速度
- **评估阶段**: 使用标准或 `-render` 版本查看模型表现
- **教师-学生模式**: 先训练 Teacher 模型，然后用于指导 Student 模型学习

## 环境参数设置

使用环境时可以通过以下方式创建:

```python
import gymnasium as gym
import auv_env  # 确保已导入自定义环境包

# 创建环境实例
env = gym.make("Teacher-v1")  # 创建带渲染的教师环境
env_train = gym.make("Teacher-v1-norender")  # 创建不渲染的训练环境
```

## 模型训练命令示例

```bash
# 训练教师模型
python SB3_trainer.py --device cuda --choice 0 --env Teacher-v1 --policy SAC --render 0

# 训练学生模型（基于教师模型）
python SB3_trainer.py --device cuda --choice 0 --env Student-v0 --policy PPO --render 0
```