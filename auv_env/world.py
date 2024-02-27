import holoocean
import numpy as np
from auv_control.estimation import InEKF
from auv_control.control import LQR
from auv_control.planning import Traj, RRT
from auv_control import State
from tools import Plotter

from auv_env.agent import AgentAuv

class World:
    def __init__(self, scenario, show, verbose, **kwargs):
        # define the entity
        self.ocean = holoocean.make(scenario_cfg=scenario, show_viewport=show, verbose=verbose)
        self.agent = None
        # init the param
        self.sampling_period = 1 / scenario["ticks_per_sec"]
        # Init and tick the scenario
        self.u = np.zeros(8)
        self.ocean.act("auv0", self.u)
        self.sensors = self.ocean.tick()
        self.build_models(sampling_period=self.sampling_period, init_state=self.sensors)

    def build_models(self, sampling_period, init_state, **kwargs):
        # Build a robot
        self.agent = AgentAuv(dim=3, sampling_period=sampling_period, init_state=init_state)

        # Build targets
        self.targetA = np.concatenate((np.concatenate((np.eye(2),
                                                       self.sampling_period * np.eye(2)), axis=1),
                                       [[0, 0, 1, 0], [0, 0, 0, 1]]))
        self.target_noise_cov = self.const_q * np.concatenate((
            np.concatenate((self.sampling_period ** 3 / 3 * np.eye(2),
                            self.sampling_period ** 2 / 2 * np.eye(2)), axis=1),
            np.concatenate((self.sampling_period ** 2 / 2 * np.eye(2),
                            self.sampling_period * np.eye(2)), axis=1)))
        if known_noise:
            self.target_true_noise_sd = self.target_noise_cov
        else:
            self.target_true_noise_sd = self.const_q_true * np.concatenate((
                np.concatenate((self.sampling_period ** 2 / 2 * np.eye(2),
                                self.sampling_period / 2 * np.eye(2)), axis=1),
                np.concatenate((self.sampling_period / 2 * np.eye(2),
                                self.sampling_period * np.eye(2)), axis=1)))

        self.targets = [AgentDoubleInt2D_Nonlinear(self.target_dim,
                                                   self.sampling_period, self.limit['target'],
                                                   lambda x: self.MAP.is_collision(x),
                                                   W=self.target_true_noise_sd, A=self.targetA,
                                                   obs_check_func=lambda x: self.MAP.get_closest_obstacle(
                                                       x, fov=2 * np.pi, r_max=10e2))
                        for _ in range(self.num_targets)]