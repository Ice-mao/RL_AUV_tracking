import holoocean
import numpy as np
from pynput import keyboard
import cv2

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)


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


from PIL import Image
from torchvision import transforms
from auv_control import scenario

# scenario = "SimpleUnderwater-Bluerov2" # "AUV_RGB"
scenario = "OpenWater-Bluerov2_RGB"  #
# with holoocean.make(scenario_cfg=scenario) as env:
with holoocean.make(scenario) as env:
    kb = KeyBoardCmd(force=25)
    from auv_control import State
    for _ in range(20000):
        if 'q' in kb.pressed_keys:
            break
        command = kb.parse_keys()

        # send to holoocean
        # env.act("auv0", command)
        env.act("target0", command)
        state = env.tick()
        state = state["target0"]
        true_state = State(state)
        print(true_state.vec[:3])
        # if ("rangefinder1" in state):
        #     print("rangefinder1:", state["rangefinder1"])
        # true_state = State(state)
        # print(true_state.vec)
        # print(state['VelocitySensor'])
        # state["PoseSensor"][:3, 3]
        # t = np.diag([1, -1, -1])
        # print(state["PoseSensorNED"][:3, :3] @ t)
        # if "LeftCamera" in state:
        #     pixels = state["LeftCamera"]
        #     cv2.namedWindow("Camera Output")
        #     cv2.imshow("Camera Output", pixels[:, :, 0:3])
        #     cv2.waitKey(1)
        #     image = state["LeftCamera"]
        #     rgb_image = image[:, :, :3]  # 取前 3 个通道 (H, W, 3)
        #     # rgb_image = np.transpose(rgb_image, (2, 0, 1))
        #     # rgb_image = rgb_image.astype(np.float32) / 255.0
        #     pil_image = Image.fromarray(rgb_image)
        #     preprocess = transforms.Compose([
        #         transforms.Resize(256),
        #         transforms.CenterCrop(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     ])
        #     tensor_image = preprocess(pil_image)
        #     image = tensor_image.numpy()
    cv2.destroyAllWindows()
