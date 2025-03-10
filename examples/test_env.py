import holoocean
import numpy as np
import auv_env
import gymnasium as gym

# env = holoocean.make("AUV_RGB")
env = gym.make("Student-v0-sample")


env.reset()
for _ in range(200):
    command = env.action_space.sample()  # 生成随机动作
    state, reward, done, _, info = env.step(command)  # 执行动作
   #  print(state["t"])
    if done:
        break  # 如果环境结束，则退出循环