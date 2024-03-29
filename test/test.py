import numpy as np
import holoocean
from tqdm import tqdm
import cv2

from auv_control.estimation import InEKF
from auv_control.control import LQR
from auv_control.planning import Traj, RRT
from auv_control import State
from auv_env.tools import Plotter
# from plotter import Plotter
from auv_control import scenario
import argparse

np.set_printoptions(suppress=True, formatter={"float_kind": f"{{:0.2f}}".format})

def main(num_seconds, show, plot, verbose, route):
    # Install simulation environments
    if "Ocean" not in holoocean.installed_packages():
        holoocean.install("Ocean")

    # Load in HoloOcean info
    ts = 1 / scenario["ticks_per_sec"]
    num_ticks = int(num_seconds / ts)

    # Set everything up
    controller = LQR()
    observer = InEKF()
    if route == "rrt":
        planner = RRT(num_seconds)
    # elif route == "RL":
    #     return
    else:
        planner = Traj(route, num_seconds)
    if plot:
        plotter = Plotter(["True", "Estimated", "Desired"])

    # Run simulation!
    u = np.zeros(8)
    with holoocean.make(scenario_cfg=scenario, show_viewport=show, verbose=verbose) as env:
        planner.draw_traj(env, num_seconds)

        for i in tqdm(range(num_ticks)):
            # Tick environment
            env.act("auv0", u)
            sensors = env.tick()

            # Pluck true state from sensors
            t = sensors["t"]
            sensors = sensors['auv0']
            true_state = State(sensors)

            # Estimate State
            est_state = observer.tick(sensors, ts)

            # Path planner
            des_state = planner.tick(t)

            # Autopilot Commands
            u = controller.u(est_state, des_state)

            # Update visualization
            if plot:
                plotter.add_timestep(t, [true_state, est_state, des_state])
                if i % 100 == 0:
                    plotter.update_plots()
            if show:
                if i % 10 == 0:
                    planner.draw_step(env, t, ts*10)

            if 'LeftCamera' in sensors:
                pixels = sensors["LeftCamera"]
                cv2.namedWindow("Camera Output")
                cv2.imshow("Camera Output", pixels[:, :, 0:3])
                cv2.waitKey(1)

            if 'RangeFinderSensor' in sensors:
                range_data = sensors['RangeFinderSensor']
                # 获得数组中的最小值及其索引
                min_value = np.min(range_data)
                angle = (360/24)*np.argmin(range_data)
                print(min_value, angle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run AUV simulation.')
    parser.add_argument('-s', '--show', action='store_true', help='Show viewport')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot data')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print holoocean output')
    parser.add_argument('-n', '--num_seconds', default=100, type=float, help='Length to run simulation for')
    parser.add_argument('-r', '--route', default="rrt", type=str, help='Routing to use (e.g., "rrt", "helix", "wave", "square")')

    args = parser.parse_args()
    main(**vars(args))
    cv2.destroyAllWindows()