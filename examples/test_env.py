import holoocean
import numpy as np
import auv_env
import gymnasium as gym

# env = holoocean.make("PierHarbor-Hovering")
env = gym.make("Student-v0-norender")
# The hovering AUV takes a command for each thruster
command = np.array([0.5, 0.5, 0.5])

env.reset()
for _ in range(180):
   state = env.step(command)
