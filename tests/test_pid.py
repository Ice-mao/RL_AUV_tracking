import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

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
controller = PID(robo_type="BlueROV2")
# controller.set_depth_target(-10.0)
# Run simulation!
u = np.zeros(8)
# env = holoocean.make(scenario_cfg=scenario)
# env = holoocean.make("AUV_RGB_Dam_sonar")
env = holoocean.make("SimpleUnderwater-Bluerov2")
env.agents['auv0'].teleport(location=[0, 0, -5], rotation=[0.0, 0.0, 0])
# env.spawn_prop('sphere', [5, 0, -5], [0, 0, 0])
cmd_vel = CmdVel()

for i in range(20000):
    # Tick environment
    env.act("auv0", u)
    sensors = env.tick()

    # Pluck true state from sensors
    t = sensors["t"]
    true_state = State(sensors['auv0'])
    current_lin_x = true_state.vec[3]  # 前向速度 (body frame)
    current_lin_z = true_state.vec[5]  # 垂直速度 (body frame)
    current_ang_z = true_state.vec[11]
    print(f"时间t, {t}, 当前前向速度: {current_lin_x}, 当前垂直速度: {current_lin_z}, 当前角速度: {current_ang_z}")
    cmd_vel.linear.x = 0 # 前进速度 0.5 m/s
    cmd_vel.angular.z = 0.2 # 绕z轴旋转角速度 0.1 rad/s
    cmd_vel.linear.z = 0.8 # 垂直速度 -0.1 m/s
    u = controller.u(true_state, cmd_vel)