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
        self.keyboard = KeyBoardCmd(20)

        # init the sensor part of AUV
        # to see if it is useful
        # self.imagesonar = ImagingSonar(scenario=scenario)  # to see if it is useful
        self.rangefinder = RangeFinder(scenario=scenario)
        self.state = State(sensor)
        # self.planner = RRT()

    def reset(self, sensor):
        self.state = State(sensor)
        self.vw = [0.0, 0.0]

    def update(self, action_vw, sensors):
        self.vw = action_vw

        # Estimate State
        est_state = self.observer.tick(sensors, self.sampling_period)

        # Path planner
        # TODO get a new planner to get des_state from vw
        des_state = self.planner.tick(self.vw)

        # TODO then check the des_state if is_col

        # Autopilot Commands
        u = self.controller.u(est_state, des_state)
        return u

class AgentSphere(Agent):
    """
        use for target
    """
    def __init__(self, dim, sampling_period, sensor, obstacles, fixed_depth, size, bottom_corner, start_time):
        Agent.__init__(self, dim, sampling_period)
        self.size = size
        self.bottom_corner = bottom_corner
        self.fix_depth = fixed_depth
        self.obstacles = obstacles
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

    def reset(self, init_state):
        super().reset(init_state)
        self.vw = [0.0, 0.0]

    def update(self, sensor, t):
        # update time
        self.time = t
        # true state
        self.vec[0:3] = sensor["PoseSensor"][:3, 3]
        self.vec[3:6] = sensor["VelocitySensor"]
        self.vec[6:9] = rot_to_rpy(sensor["PoseSensor"][:3, :3])
        # desire state
        des_state = self.planner.tick(self.time)
        # TODO then check the des_state if is_col
        # Autopilot Commands
        true_state = np.array([self.vec[0], self.vec[1], np.radians(self.vec[8])])
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
