import holoocean
import numpy as np
from pynput import keyboard

pressed_keys = list()
force = 25


def on_press(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.append(key.char)
        pressed_keys = list(set(pressed_keys))


def on_release(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.remove(key.char)


listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()


def parse_keys(keys, val):
    command = np.zeros(8)
    if 'i' in keys:
        command[0:4] += val
    if 'k' in keys:
        command[0:4] -= val
    if 'j' in keys:
        command[[4, 7]] += val
        command[[5, 6]] -= val
    if 'l' in keys:
        command[[4, 7]] -= val
        command[[5, 6]] += val

    if 'w' in keys:
        command[4:8] += val
    if 's' in keys:
        command[4:8] -= val
    if 'a' in keys:
        command[[4, 6]] += val
        command[[5, 7]] -= val
    if 'd' in keys:
        command[[4, 6]] -= val
        command[[5, 7]] += val

    return command


from auv_control import scenario

env = holoocean.make(scenario_cfg=scenario)
while True:
    if 'q' in pressed_keys:
        break
    command = parse_keys(pressed_keys, force)

    # send to holoocean
    env.act("auv0", command)
    state = env.tick()
