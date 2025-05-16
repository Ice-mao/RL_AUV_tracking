if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)

import holoocean
import matplotlib.pyplot as plt
from auv_env.envs.tools import KeyBoardCmd
import numpy as np

env = holoocean.make("AUV_RGB_Dam")
kb_cmd = KeyBoardCmd(force=30)
# The hovering AUV takes a command for each thruster

for i in range(1800000):
    print(i)
    if 'q' in kb_cmd.pressed_keys:
        break
    command = kb_cmd.parse_keys()

    # send to holoocean
    env.act("auv0", command)
    state = env.tick()
    print(state['auv0']['LocationSensor'])