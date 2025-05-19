if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)

import holoocean
import numpy as np
from numpy import linalg as LA
from auv_env import util
from metadata import METADATA
import copy
from ultralytics import YOLO  # 引入YOLO库
from auv_control.control import CmdVel
import cv2

class YOLOTracker:
    """
        YOLOwithLQR
    """

    def __init__(self):
        # 加载YOLO模型
        # self.model = YOLO("yolov8n.pt")  # 使用YOLOv8 nano模型，根据需要替换为合适的模型
        self.model = YOLO("/home/dell-t3660tow/data/RL/RL_AUV_tracking/RL_AUV_tracking/auv_baseline/best.pt")
        self.last_detection = None
        self.action = CmdVel()

        # 设置目标标签 - 球的标签（在COCO数据集中，sports ball的ID为32）
        self.target_class_ids = [0]  # 可以添加更多目标类别ID
        self.target_class_names = ["auv"]  # 对应的类名

        # 可视化设置
        self.show_visualization = True

    def predict(self, obs):  
        display_image = obs.copy()
        results = self.model(obs)

        if len(results) == 0 or len(results[0].boxes) == 0:
            if self.last_detection is not None:
                print("目标丢失，使用上一次的控制策略")
                if self.show_visualization:
                    cv2.putText(display_image, "Target Lost!", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Tracking View", display_image)
                    cv2.waitKey(1)
                return self.last_detection
            else:
                print("未检测到目标")
                if self.show_visualization:
                    cv2.putText(display_image, "No Target Found", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("Tracking View", display_image)
                    cv2.waitKey(1)
                self.action.linear.x = 0.0
                self.action.angular.z = 0.0
                return self.action

        found_target = False
        found_idx = 0
        found_confidence = 0.0
        for i in range(len(results[0].boxes)):
            detection = results[0].boxes[i]
            class_id = int(detection.cls.item())
            class_name = results[0].names[class_id]
            confidence = detection.conf.item()

            if class_id in self.target_class_ids or class_name in self.target_class_names:
                found_target = True
                if confidence > found_confidence:
                    found_confidence = confidence
                    found_idx = i
                
        if found_target:
            detection = results[0].boxes[found_idx]
            confidence = detection.conf.item()
            bbox = detection.xyxy[0].cpu().numpy().astype(np.int32)
            bbox_xywhn = detection.xywhn[0].cpu().numpy()
            
            # 计算边界框中心点
            center_x = bbox_xywhn[0]
            image_height_normal = bbox_xywhn[3]

            # 在图像上绘制边界框和标签
            if self.show_visualization:
                # 绘制矩形框
                cv2.rectangle(display_image, 
                            (bbox[0], bbox[1]), 
                            (bbox[2], bbox[3]), 
                            (0, 255, 0), 2)
                
                # 在框上方添加标签和置信度
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(display_image, label, 
                            (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)
                
                # 绘制中心点
                img_h, img_w = display_image.shape[:2]
                center_x_pixel = int(center_x * img_w)
                center_y_pixel = int(bbox_xywhn[1] * img_h)
                cv2.circle(display_image, (center_x_pixel, center_y_pixel), 
                            5, (255, 0, 0), -1)
                
                # 绘制图像中心线
                cv2.line(display_image, (img_w//2, 0), (img_w//2, img_h), 
                        (0, 0, 255), 1)
                
            # 计算目标中心与图像中心的偏差
            error_x = center_x - 0.5
            error_h = image_height_normal - 0.2
            print(f"error_x: {error_x}, error_h: {error_h}")
            if error_h > 0:
                u_v = 0
            else:
                u_v = -4 * error_h
            u_w = -5 * error_x
            
            self.action.linear.x = u_v
            self.action.angular.z = u_w

        else:
            print("No target")
            if self.show_visualization:
                cv2.putText(display_image, "No Ball Found", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            if self.last_detection is not None:
                # 使用上一次的控制动作
                pass
            else:
                self.action.linear.x = 0.0
                self.action.angular.z = 0.0

        # 存储当前动作用于下一次检测
        self.last_detection = self.action
        
        if self.show_visualization:
            cv2.imshow("Ball Tracking", display_image)
            cv2.waitKey(1)
        return self.action

if __name__ == "__main__":
    import numpy as np
    import holoocean

    from auv_control.control import LQR, PID, CmdVel
    from auv_control import State
    # from plotter import Plotter
    from auv_control import scenario

    # Load in HoloOcean info
    ts = 1 / scenario["ticks_per_sec"]
    # num_ticks = int(num_seconds / ts)

    # Set everything up
    tracker = YOLOTracker()
    controller = PID()
    controller.set_depth_target(-5.0)
    # Run simulation!
    u = np.zeros(8)
    env = holoocean.make("AUV_RGB_Dam_test")
    env.agents['auv0'].teleport(location=[0, 0, -5], rotation=[0.0, 0.0, 0])
    env.spawn_prop('sphere', [5, 0, -5], [0, 0, 0])
    cmd_vel = CmdVel()

    for i in range(20000):
        # Tick environment
        env.act("auv0", u)
        # env.act("target0", np.array([0, 0, 0, 500]))
        sensors = env.tick()
        if 'LeftCamera' in sensors['auv0']:
            obs = sensors['auv0']['LeftCamera']
            obs = obs[:, :, :3]
            cmd_vel = tracker.predict(obs)

        # Pluck true state from sensors
        t = sensors["t"]
        true_state = State(sensors['auv0'])
        # cmd_vel.linear.x = 0.0  # 前进速度 0.5 m/s
        # cmd_vel.angular.z = -0.3  # 绕z轴旋转角速度 0.1 rad/s
        u = controller.u(true_state, cmd_vel)
