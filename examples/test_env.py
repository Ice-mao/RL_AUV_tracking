import holoocean
import numpy as np

env = holoocean.make("PierHarbor-Hovering")

# The hovering AUV takes a command for each thruster
command = np.array([10,10,10,10,0,0,0,0])

for _ in range(180):
   state = env.step(command)
