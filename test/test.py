import holoocean
import numpy as np
from pynput import keyboard
from auv_env.envs.tools import KeyBoardCmd

kb_cmd = KeyBoardCmd(force=500)


with holoocean.make("Dam-HoveringCamera") as env:
    while True:
        if 'q' in kb_cmd.pressed_keys:
            break
        command = kb_cmd.parse_keys()

        #send to holoocean
        env.act("auv0", command)
        state = env.tick()
        x = state['PoseSensor'][0][3]
        y = state['PoseSensor'][1][3]
        z = state['PoseSensor'][2][3]
        print("x: ", x, "y: ", y, "z: ", z)