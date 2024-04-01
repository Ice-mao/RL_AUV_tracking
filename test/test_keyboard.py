import holoocean
import numpy as np
from pynput import keyboard
import cv2


class KeyBoardCmd:
    """
        实现键盘控制的类
    """

    def __init__(self, force=25):
        self.force = force
        self.pressed_keys = list()
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if hasattr(key, 'char'):
            self.pressed_keys.append(key.char)
            self.pressed_keys = list(set(self.pressed_keys))

    def on_release(self, key):
        if hasattr(key, 'char'):
            self.pressed_keys.remove(key.char)

    def parse_keys(self):
        command = np.zeros(8)
        if 'i' in self.pressed_keys:
            command[0:4] += self.force
        if 'k' in self.pressed_keys:
            command[0:4] -= self.force
        if 'j' in self.pressed_keys:
            command[[4, 7]] += self.force
            command[[5, 6]] -= self.force
        if 'l' in self.pressed_keys:
            command[[4, 7]] -= self.force
            command[[5, 6]] += self.force

        if 'w' in self.pressed_keys:
            command[4:8] += self.force
        if 's' in self.pressed_keys:
            command[4:8] -= self.force
        if 'a' in self.pressed_keys:
            command[[4, 6]] += self.force
            command[[5, 7]] -= self.force
        if 'd' in self.pressed_keys:
            command[[4, 6]] -= self.force
            command[[5, 7]] += self.force
        return command


with holoocean.make("TestMap") as env:
    kb = KeyBoardCmd(force=25)
    for _ in range(20000):
        if 'q' in kb.pressed_keys:
            break
        command = kb.parse_keys()

        # send to holoocean
        env.act("auv0", command)
        state = env.tick()

        if "LeftCamera" in state:
            pixels = state["LeftCamera"]
            cv2.namedWindow("Camera Output")
            cv2.imshow("Camera Output", pixels[:, :, 0:3])
            cv2.waitKey(1)
    cv2.destroyAllWindows()
