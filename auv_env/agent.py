import numpy as np
from auv_env.metadata import METADATA
from auv_env.tools import KeyBoardCmd, ImagingSonar, RangeFinder
import auv_env.util as util

from auv_control.estimation import InEKF
from auv_control.control import LQR
from auv_control.planning import Traj, RRT, RRT_2d
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
    def __init__(self, dim, sampling_period, sensor, fixed_depth, size, bottom_corner):
        Agent.__init__(self, dim, sampling_period)
        self.size = size
        self.bottom_corner = bottom_corner
        # init the control part of Auv
        self.init_pos = sensor['LocationSensor']
        _target= np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
        is_agent_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target, self.margin2wall)
        # self.planner = RRT(start=self.init_pos, )
        self.planner = RRT_2d(start=self.init_pos, )
        self.controller = LQR()
        self.observer = InEKF()
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

    def update(self, action_vw, sensor):
        self.vw = action_vw
        self.vec[0:3] = sensor["PoseSensor"][:3, 3]
        self.vec[3:6] = sensor["VelocitySensor"]
        self.vec[6:9] = rot_to_rpy(sensor["PoseSensor"][:3, :3])
        # Estimate State
        est_state = self.observer.tick(sensor, self.sampling_period)

        # Path planner
        # TODO get a new planner to get des_state from vw
        des_state = self.planner.tick(self.vw)

        # TODO then check the des_state if is_col

        # Autopilot Commands
        u = self.controller.u(est_state, des_state)
        return u
