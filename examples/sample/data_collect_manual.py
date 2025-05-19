if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)

import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2
import math
from tqdm import tqdm
import holoocean
from pynput import keyboard
from auv_control.control import LQR, PID, CmdVel
from auv_control import State

#### GET SONAR CONFIG
# scenario = "AUV_RGB_PH_sonar"
scenario = "AUV_RGB_OpenWater_sonar"
# scenario = "AUV_RGB_Dam_sonar"
# scenario = "OpenWater-HoveringImagingSonar"
config = holoocean.packagemanager.get_scenario(scenario)
config = config['agents'][0]['sensors'][-1]["configuration"]
azi = config['Azimuth']
minR = config['RangeMin']
maxR = config['RangeMax']
binsR = config['RangeBins']
binsA = config['AzimuthBins']

#### GET PLOT READY
plt.ion()
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 5))
ax.set_theta_zero_location("N")
ax.set_thetamin(-azi / 2)
ax.set_thetamax(azi / 2)

theta = np.linspace(-azi / 2, azi / 2, binsA) * np.pi / 180
r = np.linspace(minR, maxR, binsR)
T, R = np.meshgrid(theta, r)
z = np.zeros_like(T)

plt.grid(False)
plot = ax.pcolormesh(T, R, z, cmap='gray', shading='auto', vmin=0, vmax=1)
plt.tight_layout()
fig.canvas.draw()
fig.canvas.flush_events()

class KeyBoardCmd:
    """
        # 实现键盘控制的类
        for example:
        kb_cmd = KeyBoardCmd(force=10)
        if 'q' in kb_cmd.pressed_keys:
            break
        command = kb_cmd.parse_keys()

        ## send to holoocean
        env.act("auv0", command)
        state = env.tick()
    """

    def __init__(self, force=25):
        self.force = force
        self.command_agent = CmdVel()
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

    def parse_keys(self, force=None):
        command_target = np.zeros((8))
        # command_agent = CmdVel()
        if force != None:
            self.force = force

        if 'w' in self.pressed_keys:
            command_target[4:8] += self.force
        if 's' in self.pressed_keys:
            command_target[4:8] -= self.force
        if 'a' in self.pressed_keys:
            command_target[[4, 7]] += self.force / 2
            command_target[[5, 6]] -= self.force / 2
        if 'd' in self.pressed_keys:
            command_target[[4, 7]] -= self.force / 2
            command_target[[5, 6]] += self.force / 2

        if 'i' in self.pressed_keys:
            self.command_agent.linear.x += 0.01
        if 'k' in self.pressed_keys:
            self.command_agent.linear.x -= 0.01
        if 'j' in self.pressed_keys:
            self.command_agent.angular.z += 0.005
        if 'l' in self.pressed_keys:
            self.command_agent.angular.z -= 0.005
        if 'o' in self.pressed_keys:
            self.command_agent.linear.x = 0
            self.command_agent.angular.z = 0
        # if 'i' in self.pressed_keys:
        #     command_agent.linear.x += 0.02
        # if 'k' in self.pressed_keys:
        #     command_agent.linear.x -= 0.02
        # if 'j' in self.pressed_keys:
        #     command_agent.angular.z += 0.01
        # if 'l' in self.pressed_keys:
        #     command_agent.angular.z -= 0.01
        return command_target, self.command_agent

def collect_dataset(output_dir, max_steps=200):
    """
    Args:
        env: 环境
        policy: 策略
        output_dir: 输出目录
        max_steps: 最大步数
    """
    # 创建保存数据的目录
    left_camera_dir = os.path.join(output_dir, "left_camera")
    right_camera_dir = os.path.join(output_dir, "right_camera")
    sonar_dir = os.path.join(output_dir, "sonar")
    state_dir = os.path.join(output_dir, "state")
    action_dir = os.path.join(output_dir, "action")
    
    os.makedirs(left_camera_dir, exist_ok=True)
    os.makedirs(right_camera_dir, exist_ok=True)
    os.makedirs(sonar_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(action_dir, exist_ok=True)
    
    kb_cmd = KeyBoardCmd(force=20)
    controller = PID()
    controller.set_depth_target(-295.0) # set target depth
    left_image = None
    right_image = None
    sonar_image = None
    state = None
    action = None
    
    # weather = holoocean.weather.WeatherController
    # weather.set_weather(weather_type="rain")
    with holoocean.make(scenario) as env:
        state = env.reset()
        for i in range(100):
            env.tick()
        print("init")
        while True:  # 主循环
            if 'q' in kb_cmd.pressed_keys:
                break
            if 'u' in kb_cmd.pressed_keys:
                for step in tqdm(range(max_steps), desc="do the episode"): #TODO 还需要修改成每0.1s进行采集
                    flag_begin = False
                    for i in range(10):
                        command_target, command_agent = kb_cmd.parse_keys()
                        if flag_begin == False:
                            flag_begin = True
                            command_agent_till = CmdVel()
                            command_agent_till.linear.x = command_agent.linear.x
                            command_agent_till.angular.z = command_agent.angular.z
                        true_state = State(state['auv0'])
                        print(f"当前前向速度: {command_agent_till.linear.x}, 当前角速度: {command_agent_till.angular.z}")
                        u = controller.u(true_state, command_agent_till)
                        # send to holoocean
                        env.act("auv0", u)
                        env.act("target0", command_target)
                        
                        state = env.tick()
                        # print(state['auv0']['LocationSensor'])
                        if 'LeftCamera' in state['auv0']:
                            left_image = state['auv0']['LeftCamera']
                        if 'RightCamera' in state['auv0']:
                            right_image = state['auv0']['RightCamera']
                        if 'ImagingSonar' in state['auv0']:
                            s = state['auv0']['ImagingSonar']  # (512, 512)
                            plot.set_array(s.ravel())
                            fig.canvas.draw()
                            fig.canvas.flush_events()
                            sonar_image = s

                    # get the data collection
                    left_img = left_image[:, :, :3]
                    cv2.imwrite(os.path.join(left_camera_dir, f"step_{step:03d}.jpg"), left_img)
                    
                    right_img = right_image[:, :, :3]
                    cv2.imwrite(os.path.join(right_camera_dir, f"step_{step:03d}.jpg"), right_img)
                    
                    sonar_img = (sonar_image * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(sonar_dir, f"step_{step:03d}.png"), sonar_img)
                    
                    # state保存
                    x = state['auv0']['LocationSensor'][:2]-state['target0']['LocationSensor'][:2]
                    obs_vec = np.sqrt(x[0]**2 + x[1]**2)
                    np.save(os.path.join(state_dir, f"step_{step:03d}.npy"), obs_vec)

                    # action保存j
                    action = np.array([command_agent_till.linear.x, command_agent_till.angular.z])
                    np.save(os.path.join(action_dir, f"step_{step:03d}.npy"), action)
                break
            else:
                command_target, command_agent = kb_cmd.parse_keys()
                true_state = State(state['auv0'])
                # print(f"{command_agent.linear.x},{command_agent.angular.z}")
                u = controller.u(true_state, command_agent)
                env.act("auv0", u)
                env.act("target0", command_target)
                state = env.tick()
                print(state['auv0']['LocationSensor'])
                if 'ImagingSonar' in state.get('auv0', {}):
                    s = state['auv0']['ImagingSonar']
                    plot.set_array(s.ravel())
                    fig.canvas.draw()
                    fig.canvas.flush_events()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="收集原始数据并保存为不同文件格式")
    parser.add_argument('--output', '-o',
                        default="/home/dell-t3660tow/data/log/sample/trajs_manual/openwater/traj_11",
                        help='输出目录路径')
    parser.add_argument('--max_steps', '-e', type=int, default=1000,
                        help='最大步数')
    args = parser.parse_args()
    
    output_path = args.output
    
    # 导入必要的库
    from stable_baselines3 import SAC
    import gymnasium as gym
    import auv_env
    from auv_env.wrappers.obs_wrapper import TeachObsWrapper
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    # env = gym.make(args.env)
    # expert_env = TeachObsWrapper(DummyVecEnv([lambda: gym.make(args.env) for _ in range(1)]))
        
    # 收集数据
    os.makedirs(output_path, exist_ok=True)
    collect_dataset(output_path, max_steps=args.max_steps)
