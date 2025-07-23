import holoocean
import numpy as np

env = holoocean.make("SimpleUnderwater-Bluerov2", show_viewport=False)

# The hovering AUV takes a command for each thruster
command = np.array([10,10,10,10,0,0,0,0])

for _ in range(2000):
   state = env.step(command)
   print(state['t'])
print("Finished!")