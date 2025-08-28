import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import holoocean

from auv_control.control import LQR
from auv_control import State
# from plotter import Plotter
from auv_control import scenario

# Load in HoloOcean info
ts = 1 / scenario["ticks_per_sec"]
# num_ticks = int(num_seconds / ts)

# Set everything up
controller = LQR(l_p=50, robo_type="BlueROV2")
# observer = InEKF()
# if route == "rrt":
#     planner = RRT(num_seconds)
# # elif route == "RL":
# #     return
# else:
#     planner = Traj(route, num_seconds)
# if plot:
#     plotter = Plotter(["True", "Estimated", "Desired"])

# Run simulation!
u = np.zeros(8)
env = holoocean.make("SimpleUnderwater-Bluerov2")
# planner.draw_traj(env, num_seconds)

# for i in tqdm(range(num_ticks)):
env.draw_point([-5, 3, -5], color=[0,255,0], thickness=20, lifetime=0)
env.draw_line([-5, 3, -5], [0, 0, -5], color=[0,0,255], thickness=20, lifetime=0)
env.draw_line([5, 3, -5], [-5, 3, -5], color=[0,0,255], thickness=20, lifetime=0)
env.draw_line([5, -3, -5], [5, 3, -5], color=[0,0,255], thickness=20, lifetime=0)
# env.spawn_prop('sphere', [5, -3, -5], [0, 0, 0])
env.agents['auv0'].teleport(location=[0, 0, -5], rotation=[0.0, 0.0, 0])
for i in range(20000):
    # Tick environment
    env.act("auv0", u)
    sensors = env.tick()

    # Pluck true state from sensors
    t = sensors["t"]
    true_state = State(sensors['auv0'])

    # Estimate State
    # est_state = observer.tick(sensors, ts)

    # Path planner
    # des_state = planner.tick(t)

    # Autopilot Commands
    depth = -5
    # if t < 3000:
    #     vel = [5 ,0, 0]
    #     w = 0.5
    #     yaw = 0
    # # elif 3 <= t < 6:
    # #     vel = [0,1,0]
    # #     yaw = 0
    # # elif 6 <= t < 9:
    # #     vel = [0,0,1]
    # #     yaw = 0
    # des_state = State(np.array([0.00, 0.00, depth, vel[0], vel[1], vel[2], 0.00, 0.00, yaw, - 0.00, - 0.00, w]))
    # # des_state.vec = [-5, 3, -5, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, - 0.00, - 0.00, 0.00]
    # u = controller.u(true_state, des_state)

    if t < 3000:
        pos = [-5, 3, -5]
        yaw = 90
    elif 3 <= t < 6:
        pos = [5, 3, -5]
        yaw = 90
    elif 6 <= t < 9:
        pos = [5, -3, -8]
        yaw = -90
    des_state = State(np.array([pos[0], pos[1], pos[2], 0.00, 0.00, 0.00, 0.0, 90.00, yaw, - 0.00, - 0.00, 0.00]))
    u = controller.u(true_state, des_state)