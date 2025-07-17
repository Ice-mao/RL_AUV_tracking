import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import holoocean

from auv_control.control import PID, CmdVel
from auv_control import State
from auv_control import scenario

# Load in HoloOcean info
ts = 1 / scenario["ticks_per_sec"]

# Set everything up
controller = PID(robo_type="BlueROV2")
# Run simulation!
u = np.zeros(8)
env = holoocean.make("SimpleUnderwater-Bluerov2")
env.agents['auv0'].teleport(location=[0, 0, -5], rotation=[0.0, 0.0, 0])
cmd_vel = CmdVel()

print("测试BlueROV2专用PID控制器")
print("目标: 前进速度 0.0 m/s, 角速度 0.1 rad/s")
print("=" * 50)

for i in range(2000):  # 缩短测试时间
    # Tick environment
    env.act("auv0", u)
    sensors = env.tick()

    # Pluck true state from sensors
    t = sensors["t"]
    true_state = State(sensors['auv0'])
    body_vel = true_state.body_velocity
    body_ang_vel = true_state.body_angular_velocity
    current_lin_x = body_vel[0]  # 前向速度 (body frame)
    current_lin_z = body_vel[2]  # 垂直速度 (body frame)
    current_ang_z = body_ang_vel[2] 

    # 设置控制目标
    cmd_vel.linear.x = 0.0  # 前进速度 0.0 m/s
    cmd_vel.linear.z = 0.1 # 垂直速度 -0.1 m/s
    cmd_vel.angular.z = 0.0 # 绕z轴旋转角速度 0.1 rad/s
    
    # 计算控制输出
    u = controller.u(true_state, cmd_vel)
    
    # 每100步输出一次信息
    if i % 100 == 0:
        lin_x_error = cmd_vel.linear.x - current_lin_x
        ang_z_error = cmd_vel.angular.z - current_ang_z
        lin_z_error = cmd_vel.linear.z - current_lin_z
        max_thrust = np.max(np.abs(u))
        
        print(f"步骤 {i:4d} (t={t:.2f}s):")
        print(f"  前向速度: {current_lin_x:.3f} -> {cmd_vel.linear.x:.3f} (误差: {lin_x_error:.3f})")
        print(f"  角速度:   {current_ang_z:.3f} -> {cmd_vel.angular.z:.3f} (误差: {ang_z_error:.3f})")
        print(f"  垂直速度: {current_lin_z:.3f} -> {cmd_vel.linear.z:.3f} (误差: {lin_z_error:.3f})")
        print(f"  最大推力: {max_thrust:.1f}N")
        print()

print("\n测试完成！BlueROV2 PID控制器基于真实规格参数调优。")
