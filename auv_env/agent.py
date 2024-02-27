import numpy as np
from auv_env.metadata import METADATA
from auv_env.tools import KeyBoardCmd, ImagingSonar
import auv_env.util as util

from auv_control.estimation import InEKF
from auv_control.control import LQR
from auv_control.planning import Traj, RRT
from auv_control import State
from auv_control import scenario


class Agent(object):
    def __init__(self, dim, sampling_period):
        self.dim = dim
        self.sampling_period = sampling_period
        self.margin = METADATA['margin']

    def range_check(self):
        self.state = np.clip(self.state, self.limit[0], self.limit[1])

    def collision_check(self, pos):
        return self.collision_func(pos[:2])

    def margin_check(self, pos, target_pos):
        return any(np.sqrt(np.sum((pos - target_pos) ** 2, axis=1)) < self.margin)  # no update

    def reset(self, init_state):
        self.state = init_state


class AgentAuv(Agent):
    def __init__(self, dim, sampling_period, init_state):
        Agent.__init__(self, dim, sampling_period)
        # init the control part of Auv
        self.controller = LQR()
        self.observer = InEKF()
        self.keyboard = KeyBoardCmd(10)
        # init the sensor part of AUV
        # to see if it is useful
        # self.imagesonar = ImagingSonar(scenario=scenario)  # to see if it is useful
        self.state = State(init_state)
        # self.planner = RRT()

    def reset(self, init_state):
        super().reset(init_state)
        self.vw = [0.0, 0.0]

    def update(self, control_input=None, margin_pos=None, col=False):
        """
        Parameters:
        ----------
        control_input : list. [linear_velocity, angular_velocity]
        margin_pos : a minimum distance to a target
        """
        if control_input is None:
            control_input = self.policy.get_control(self.state)
        if self.dim == 3:
            new_state = SE2Dynamics(self.state, self.sampling_period, control_input)
        elif self.dim == 5:
            new_state = SE2DynamicsVel(self.state, self.sampling_period, control_input)
        is_col = 0
        if self.collision_check(new_state[:2]):
            is_col = 1
            new_state[:2] = self.state[:2]  # 暂不执行坐标更新
            # control_input = self.vw  # 执行上一次的命令？
            if self.policy is not None:  # 对于target进行策略调整
                corrected_policy = self.policy.collision(new_state)
                control_input = corrected_policy
                if corrected_policy is not None:
                    if self.dim == 3:
                        new_state = SE2Dynamics(self.state, self.sampling_period, corrected_policy)
                    elif self.dim == 5:
                        new_state = SE2DynamicsVel(self.state, self.sampling_period, corrected_policy)

        elif margin_pos is not None:  # 对于agent进行最小距离检测
            if self.margin_check(new_state[:2], margin_pos):  # 如果小于最小距离
                new_state[:2] = self.state[:2]  # 暂不执行坐标更新
                # control_input = self.vw

        self.state = new_state
        self.vw = control_input
        self.range_check()

        return is_col

        def update_state()