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
import copy
from auv_control.control import CmdVel
import cv2

from auv_baseline.SeqTrackv2.tracking.seqtrack_simulation import SeqTrackSimulation

class SeqTracker:
    """
        SeqTracker
    """

    def __init__(self):
        self.model = SeqTrackSimulation('seqtrack', 'seqtrack_b256')
        self.begin_flag = False
        self.action = np.array([0.0, 0.0, 0.0])  # [forward, vertical, yaw]

        self.show_visualization = True
        
        self.gain_vx = 0.001  # v_x gain
        self.gain_wy = 0.0  # w_y gain
        self.gain_vh = 0.0  # v_z gain
        self.target_height = 0.3

    def predict(self, obs):  
        # 创建显示图像的副本，确保与OpenCV兼容
        display_image = obs.copy()
        if display_image.dtype != np.uint8:
            # 如果图像是浮点格式，转换为uint8
            if display_image.max() <= 1.0:
                display_image = (display_image * 255).astype(np.uint8)
            else:
                display_image = display_image.astype(np.uint8)
        
        # 初始化SeqTrack
        if self.begin_flag is False:
            self.begin_flag = True
            self.model.initialize(obs)
            if self.show_visualization:
                cv2.putText(display_image, "SeqTrack Initialized", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("SeqTrack Tracking", display_image)
                cv2.waitKey(1)  # 改为1ms等待，避免阻塞
            self.action = np.array([0.0, 0.0, 0.0])  # [forward, vertical, yaw]
            return self.action

        bbox = self.model.track(obs)
        
        if bbox is None or len(bbox) == 0:
            assert "no bbox"
        
        x, y, w, h = [int(v) for v in bbox]
        
        # 计算边界框的四个角点
        x_min, y_min = x, y
        x_max, y_max = x + w, y + h
        
        # 计算图像尺寸
        img_h, img_w = display_image.shape[:2]
        
        # 计算归一化的边界框参数
        center_x_norm = (x_min + x_max) / 2.0 / img_w
        center_y_norm = (y_min + y_max) / 2.0 / img_h
        bbox_height_norm = h / img_h    
        
        if self.show_visualization:
            cv2.rectangle(display_image, 
                        (x_min, y_min), 
                        (x_max, y_max), 
                        (0, 255, 0), 2)
        
        # 计算控制误差
        error_x = center_x_norm - 0.5  # 水平偏差 [-0.5, 0.5]
        error_y = center_y_norm - 0.5  # 垂直偏差 [-0.5, 0.5]
        error_h = bbox_height_norm - self.target_height  # 高度误差

        print(f"SeqTrack - error_x: {error_x:.3f}, error_y: {error_y:.3f}, error_h: {error_h:.3f}")

        if error_h > 0:
            u_forward = 0
        else:
            u_forward = -self.gain_vx * error_h  # 目标太小，前进
        
        u_yaw = -self.gain_wy * error_x
        u_vertical = -self.gain_vh * error_y
        self.action = np.array([u_forward, u_yaw, u_vertical])
        
        if self.show_visualization:
            cv2.imshow("SeqTrack Tracking", display_image)
            cv2.waitKey(1)  # 改为1ms等待，避免阻塞
            
        return self.action

    def get_action(self, obs):
        """
        Alias for predict to be consistent with SB3 model API.
        """
        return self.predict(obs)

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
    tracker = SeqTracker()
    controller = PID(robo_type='BlueROV2')
    # controller.set_depth_target(-5.0)
    # Run simulation!
    u = np.zeros(8)
    env = holoocean.make("SimpleUnderwater-Bluerov2_RGB")
    env.agents['auv0'].teleport(location=[0, 0, -5], rotation=[0.0, 0.0, 0])
    env.agents['target0'].teleport(location=[3, 0, -5], rotation=[0.0, 0.0, 0])
    # env.spawn_prop('sphere', [5, 0, -5], [0, 0, 0])
    cmd_vel = CmdVel()
    action = None
    from auv_env.envs.tools import KeyBoardCmd
    kb_cmd = KeyBoardCmd(force=10)

    for i in range(20000):
        # Tick environment
        if 'q' in kb_cmd.pressed_keys:
            break
        command = kb_cmd.parse_keys()

        env.act("auv0", u)
        env.act("target0", command)
        # env.act("target0", np.array([0, 0, 0, 500]))
        sensors = {}
        for _ in range(100):
            sensors.update(env.tick())
        sensors.update(env.tick())
        if 'LeftCamera' in sensors['auv0']:
            obs = sensors['auv0']['LeftCamera']
            obs = obs[:, :, :3]
            action = tracker.predict(obs)
            print(action)
        if action is not None:
            cmd_vel.linear.x = action[0]
            cmd_vel.angular.z = action[1]
            cmd_vel.linear.z = action[2]
        # Pluck true state from sensors
        t = sensors["t"]
        true_state = State(sensors['auv0'])
        # cmd_vel.linear.x = 0.0  # 前进速度 0.5 m/s
        # cmd_vel.angular.z = -0.3  # 绕z轴旋转角速度 0.1 rad/s
        u = controller.u(true_state, cmd_vel)
