import numpy as np
from auv_env.metadata import METADATA
from auv_env.tools import KeyBoardCmd, ImagingSonar, RangeFinder
import auv_env.util as util

from auv_control.estimation import InEKF
from auv_control.control import LQR, SE2PIDController
from auv_control.planning import RRT_2d
from auv_control.state import rot_to_rpy
from auv_control import State
from auv_control import scenario


class Agent(object):
    def __init__(self, dim, sampling_period):
        self.dim = dim
        self.sampling_period = sampling_period
        self.margin = METADATA['margin']
        self.margin2wall = METADATA['margin2wall']

    def range_check(self):
        self.state = np.clip(self.state, self.limit[0], self.limit[1])

    def collision_check(self, pos):
        return self.collision_func(pos[:2])

    def margin_check(self, pos, target_pos):
        return any(np.sqrt(np.sum((pos - target_pos) ** 2, axis=1)) < self.margin)  # no update

    def reset(self, init_state):
        self.state = State(init_state)


class AgentAuv(Agent):
    def __init__(self, dim, sampling_period, sensor):
        Agent.__init__(self, dim, sampling_period)
        # init the control part of Auv
        self.controller = LQR()
        self.observer = InEKF()
        if METADATA['render']:
            self.keyboard = KeyBoardCmd(20)

        # init the sensor part of AUV
        # to see if it is useful
        # self.imagesonar = ImagingSonar(scenario=scenario)  # to see if it is useful
        self.rangefinder = RangeFinder(scenario=scenario)
        self.state = State(sensor)
        self.est_state = State(sensor)
        # self.planner = RRT()

    def reset(self, sensor):
        self.state = State(sensor)
        self.est_state = State(sensor)
        self.vw = [0.0, 0.0]

    def update(self, action_waypoint, depth, sensors):
        self.rangefinder.update(sensors)
        # Estimate State
        self.state = State(sensors)
        self.est_state = self.observer.tick(sensors, self.sampling_period)

        # Path planner
        des_state = State(np.array([action_waypoint[0], action_waypoint[1], depth,
                                    0.00, 0.00, 0.00,
                                    0.00, 0.00, action_waypoint[2],
                                    -0.00, -0.00, 0.00]))

        # Autopilot Commands
        u = self.controller.u(self.est_state, des_state)
        return u


class AgentSphere(Agent):
    """
        use for target
    """

    def __init__(self, dim, sampling_period, sensor, obstacles, fixed_depth, size, bottom_corner, start_time, scene):
        Agent.__init__(self, dim, sampling_period)
        self.size = size
        self.bottom_corner = bottom_corner
        self.fix_depth = fixed_depth
        self.obstacles = obstacles
        self.scene = scene
        # init the control part of Auv
        self.init_pos = sensor['LocationSensor']
        _target = None
        is_end_valid = False
        while not is_end_valid:
            _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
            is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target, self.margin2wall)
        self.target_pos = np.append(_target, self.fix_depth)
        self.planner = RRT_2d(start=self.init_pos, end=self.target_pos, obstacles=self.obstacles, margin=self.margin,
                              fixed_depth=self.fix_depth, num_seconds=30,
                              bottom_corner=self.bottom_corner, size=self.size, start_time=start_time)
        self.controller = SE2PIDController()
        # self.observer = InEKF()
        # init the sensor part of AUV
        # to see if it is useful
        # self.imagesonar = ImagingSonar(scenario=scenario)  # to see if it is useful
        self.vec = []
        self.vec[0:3] = sensor["PoseSensor"][:3, 3]
        self.vec[3:6] = sensor["VelocitySensor"]
        self.vec[6:9] = rot_to_rpy(sensor["PoseSensor"][:3, :3])

    def reset(self, sensor, obstacles, scene, start_time):
        self.obstacles = obstacles
        self.scene = scene  # have a check if is changed
        # init the control part of AUV
        self.init_pos = sensor['LocationSensor']
        _target = None
        is_end_valid = False
        while not is_end_valid:
            _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
            is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target, self.margin2wall)
        self.target_pos = np.append(_target, self.fix_depth)
        self.planner.reset(start=self.init_pos, end=self.target_pos, time=start_time)
        self.controller.reset()
        self.vec = []
        self.vec[0:3] = sensor["PoseSensor"][:3, 3]
        self.vec[3:6] = sensor["VelocitySensor"]
        self.vec[6:9] = rot_to_rpy(sensor["PoseSensor"][:3, :3])

    def update(self, sensor, t):
        # update time
        self.time = t
        # true state
        self.vec[0:3] = sensor["PoseSensor"][:3, 3]
        self.vec[3:6] = sensor["VelocitySensor"]
        self.vec[6:9] = rot_to_rpy(sensor["PoseSensor"][:3, :3])
        true_state = np.array([self.vec[0], self.vec[1], np.radians(self.vec[8])])
        # desire state
        if self.planner.finish_flag == 1:
            is_end_valid = False
            while not is_end_valid:
                _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
                is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
                                                                                                  self.margin2wall)
            self.target_pos = np.append(_target, self.fix_depth)
            self.planner.reset(start=true_state, end=self.target_pos, time=t)
            if METADATA['render']:
                self.planner.draw_traj(self.scene, 30)
        if self.planner.desire_path_num == 0:
            self.scene.agents['target'].teleport(rotation=[0.0, 0.0,
                                                           -np.rad2deg(np.arctan2(
                                                               self.planner.path[1, 1] - self.planner.path[1, 0],
                                                               self.planner.path[0, 1] - self.planner.path[0, 0]))])
        des_state = self.planner.tick(true_state)  # only x, y
        # Autopilot Commands
        u = self.controller.u(true_state, des_state, self.sampling_period)
        return u

    def in_bound(self, pos):
        """
        :param pos:
        :return: True: in area, False: out area
        """
        return not ((pos[0] < self.bottom_corner[0] + self.margin2wall)
                    or (pos[0] > self.size[0] + self.bottom_corner[0] - self.margin2wall)
                    or (pos[1] < self.bottom_corner[1] + self.margin2wall)
                    or (pos[1] > self.size[1] + self.bottom_corner[1] - self.margin2wall))


#########################
# agent model:
def SE2Dynamics(x, dt, u):
    """
    update dynamics function with a control input -- linear, angular velocities
    """
    assert (len(x) == 3)
    tw = dt * u[1]  # tau * w

    # Update the agent state
    if abs(tw) < 0.001:
        diff = np.array([dt * u[0] * np.cos(x[2] + tw / 2),
                         dt * u[0] * np.sin(x[2] + tw / 2),
                         tw])
    else:
        diff = np.array([u[0] / u[1] * (np.sin(x[2] + tw) - np.sin(x[2])),
                         u[0] / u[1] * (np.cos(x[2]) - np.cos(x[2] + tw)),
                         tw])
    new_x = x + diff
    new_x[2] = util.wrap_around(new_x[2])
    return new_x
