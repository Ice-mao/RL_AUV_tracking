import holoocean
import numpy as np
from pynput import keyboard
import cv2
import os
import time
from datetime import datetime
from PIL import Image
from torchvision import transforms

# 创建保存图像的目录
SAVE_DIR = "/data/models/YOLO/auv_tracking_dataset/collected_images"
os.makedirs(SAVE_DIR, exist_ok=True)

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
        # 添加保存图像的标志
        self.save_image = False
        
    def on_press(self, key):
        if hasattr(key, 'char'):
            self.pressed_keys.append(key.char)
            self.pressed_keys = list(set(self.pressed_keys))
            
            # 按下'c'键时，设置保存图像标志
            if key.char == 'c':
                self.save_image = True

    def on_release(self, key):
        if hasattr(key, 'char'):
            if key.char in self.pressed_keys:
                self.pressed_keys.remove(key.char)

    def parse_keys(self):
        # 按键处理代码保持不变
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


# 图像计数器，用于生成序列文件名
image_counter = 0

with holoocean.make("AUV_RGB_Dam") as env:
    kb = KeyBoardCmd(force=25)
    save_message_time = 0  # 用于控制保存消息的显示时间
    
    for _ in range(20000):
        if 'q' in kb.pressed_keys:
            break
        command = kb.parse_keys()

        # send to holoocean
        env.act("auv0", command)
        state = env.tick()
        state = state["auv0"]
        
        if "LeftCamera" in state:
            pixels = state["LeftCamera"]
            display_image = pixels[:, :, 0:3].copy()  # 创建副本用于显示
            
            # 显示按键帮助信息
            cv2.putText(display_image, "Press 'c' to capture image", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 当按下'c'键时保存图像
            if kb.save_image:
                # 获取RGB图像
                image = state["LeftCamera"]
                bgr_image = image[:, :, :3]  # 取前3个通道 (H, W, 3)
                
                # 生成文件名 (可以使用时间戳或计数器)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{SAVE_DIR}/image_{image_counter:04d}_{timestamp}.jpg"
                cv2.imwrite(filename, bgr_image)
                # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(filename, bgr_image)

                # pil_image = Image.fromarray(rgb_image.astype('uint8'))
                # pil_image.save(filename)
                
                print(f"图像已保存: {filename}")
                save_message_time = time.time()
                image_counter += 1
                kb.save_image = False  # 重置保存标志
            
            # 显示保存状态消息
            if time.time() - save_message_time < 2.0:  # 显示2秒
                cv2.putText(display_image, "Image Saved!", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.namedWindow("Camera Output")
            cv2.imshow("Camera Output", display_image)
            cv2.waitKey(1)
            
    cv2.destroyAllWindows()