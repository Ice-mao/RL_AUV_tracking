from gymnasium import spaces, logger

import numpy as np
from numpy import linalg as LA

from auv_env.base import TargetTrackingBase
from auv_env.maps import map_utils
from ttenv.agent_models import *
from ttenv.policies import *
from ttenv.belief_tracker import KFbelief, UKFbelief
from auv_env.metadata import METADATA
import ttenv.util as util

class AuvTrackingEnv(TargetTrackingBase):
    def __init__(self, num_targets=1, map_name='empty', is_training=True, known_noise=True, **kwargs):
        TargetTrackingBase.__init__(self, num_targets=num_targets, map_name=map_name,
                                    is_training=is_training, known_noise=known_noise, **kwargs)
        self.id = 'AuvTracking'
        self.target_dim = 4
        self.target_init_vel = np.array(METADATA['target_init_vel'])

        # Set limits.
        self.set_limits(target_speed_limit=METADATA['target_speed_limit'])

        # Build an agent, targets, and beliefs.
        self.build_models(const_q=METADATA['const_q'], known_noise=known_noise)

    def reset(self, **kwargs):
        # Always set the limits first.
        if 'target_speed_limit' in kwargs:
            self.set_limits(target_speed_limit=kwargs['target_speed_limit'])

        if 'const_q' in kwargs:
            self.build_models(const_q=kwargs['const_q'])

        # Reset the agent, targets, and beliefs with sampled initial positions.
        init_pose = super().reset(**kwargs)
        self.agent.reset(init_pose['agent'])
        for i in range(self.num_targets):
            self.belief_targets[i].reset(
                init_state=np.concatenate((init_pose['belief_targets'][i][:2], np.zeros(2))),
                init_cov=self.target_init_cov)
            self.targets[i].reset(np.concatenate((init_pose['targets'][i][:2], self.target_init_vel)))

        # The targets are observed by the agent (z_0) and the beliefs are updated (b_0).
        observed = self.observe_and_update_belief()

        # Predict the target for the next step, b_1|0.
        for i in range(self.num_targets):
            self.belief_targets[i].predict()

        observed = [True]  # 初始化的时候输入先验
        # Compute the RL state.
        self.state_func([0.0, 0.0], observed)
        return self.state, 0

    def state_func(self, action_vw, observed):
        # Find the closest obstacle coordinate.
        obstacles_pt = self.MAP.get_closest_obstacle(self.agent.state)
        if obstacles_pt is None:
            obstacles_pt = (self.sensor_r, np.pi)

        self.state = []
        for i in range(self.num_targets):
            r_b, alpha_b = util.relative_distance_polar(self.belief_targets[i].state[:2],
                                                        xy_base=self.agent.state[:2],
                                                        theta_base=self.agent.state[2])
            r_dot_b, alpha_dot_b = util.relative_velocity_polar(
                self.belief_targets[i].state[:2],
                self.belief_targets[i].state[2:],
                self.agent.state[:2], self.agent.state[2],
                action_vw[0], action_vw[1])
            self.state.extend([r_b, alpha_b, r_dot_b, alpha_dot_b,
                               np.log(LA.det(self.belief_targets[i].cov)),
                               float(observed[i])])
        self.state.extend([obstacles_pt[0], obstacles_pt[1]])
        self.state = np.array(self.state)

        # Update the visit map when there is any target not observed for the evaluation purpose.
        if self.MAP.visit_map is not None:
            self.MAP.update_visit_freq_map(self.agent.state, 1.0, observed=bool(np.mean(observed)))

    def set_limits(self, target_speed_limit=None):
        self.num_target_dep_vars = 6
        self.num_target_indep_vars = 2

        if target_speed_limit is None:
            self.target_speed_limit = np.random.choice([1.0, 3.0])
        else:
            self.target_speed_limit = target_speed_limit
        rel_speed_limit = self.target_speed_limit + METADATA['action_v'][0]  # Maximum relative speed

        self.limit = {}  # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.MAP.mapmin, [-np.pi])), np.concatenate((self.MAP.mapmax, [np.pi]))]
        self.limit['target'] = [np.concatenate((self.MAP.mapmin, [-self.target_speed_limit, -self.target_speed_limit])),
                                np.concatenate((self.MAP.mapmax, [self.target_speed_limit, self.target_speed_limit]))]
        self.limit['state'] = [np.concatenate(
            ([0.0, -np.pi, -rel_speed_limit, -10 * np.pi, -50.0, 0.0] * self.num_targets, [0.0, -np.pi])),
            np.concatenate((
                [600.0, np.pi, rel_speed_limit, 10 * np.pi, 50.0, 2.0] * self.num_targets,
                [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float32)
        assert (len(self.limit['state'][0]) == (
                self.num_target_dep_vars * self.num_targets + self.num_target_indep_vars))

    def build_models(self, const_q=None, known_noise=True, **kwargs):
        if const_q is None:
            self.const_q = np.random.choice([0.001, 0.1, 1.0])
        else:
            self.const_q = const_q

        # Build a robot
        self.agent = AgentSE2(dim=3, sampling_period=self.sampling_period, limit=self.limit['agent'],
                              collision_func=lambda x: self.MAP.is_collision(x))
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
        self.belief_targets = [KFbelief(dim=self.target_dim,
                                        limit=self.limit['target'], A=self.targetA,
                                        W=self.target_noise_cov,
                                        obs_noise_func=self.observation_noise,
                                        collision_func=lambda x: self.MAP.is_collision(x))
                               for _ in range(self.num_targets)]