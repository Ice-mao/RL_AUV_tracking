import holoocean
import numpy as np
import auv_env
import gymnasium as gym

# env = holoocean.make("PierHarbor-Hovering")
env = gym.make("Student-v0-norender")


env.reset()
for _ in range(180):
   state = env.step(command)
   print(state["t"])
