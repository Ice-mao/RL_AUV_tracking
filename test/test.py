import holoocean
import numpy as np
from pynput import keyboard
from auv_env.envs.tools import KeyBoardCmd

kb_cmd = KeyBoardCmd(force=10)


with holoocean.make("AUV_RGB") as env:
    while True:
        if 'q' in kb_cmd.pressed_keys:
            break
        command = kb_cmd.parse_keys()

        #send to holoocean
        env.act("target0", command)
        state = env.tick()