import numpy as np
import holoocean
from tqdm import tqdm

from auv_control.estimation import InEKF
from auv_control.control import LQR
from auv_control.planning import Traj, RRT
from auv_control import State
from auv_env.tools import Plotter
# from plotter import Plotter
from auv_control import scenario
import argparse
import os

# Load in HoloOcean info
ts = 1 / scenario["ticks_per_sec"]
# num_ticks = int(num_seconds / ts)

# Set everything up
controller = LQR()
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
env = holoocean.make(scenario_cfg=scenario)
# planner.draw_traj(env, num_seconds)

# for i in tqdm(range(num_ticks)):
for _ in range(20000):
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
    des_state = State(np.array([-5, 3, -5, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, - 0.00, - 0.00, 0.00]))
    # des_state.vec = [-5, 3, -5, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, - 0.00, - 0.00, 0.00]
    u = controller.u(true_state, des_state)