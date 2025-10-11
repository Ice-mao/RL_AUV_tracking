import holoocean
import numpy as np
from numpy import linalg as LA
from auv_control.estimation import KFbelief, UKFbelief
from auv_control.control import LQR
from auv_control.planning import Traj, RRT
from auv_control import State

from auv_env import util
from auv_env.envs.base import WorldBase
from auv_env.envs.agent import AgentAuv, AgentAuvTarget
from auv_env.envs.obstacle import Obstacle
from auv_env.envs.tools import CameraBuffer

from gymnasium import spaces
import logging
import copy
from collections import deque

class WorldAuvV0(WorldBase):
    """
        state only
    """
    def __init__(self, config, map, show):
        self.obs = {}
        self.reward_queue = deque(maxlen=100)
        self.action_queue = deque(maxlen=5)
        super().__init__(config, map, show)

    def reset(self, seed=None, **kwargs):
        self.reward_queue.clear()
        self.action_queue.clear()
        return super().reset(seed=seed, **kwargs)

    def set_limits(self):
        # action_dim and action_space
        if self.config['agent']['controller'] == 'LQR':
            self.action_dim = self.config['agent']['controller_config']['LQR']['action_dim']
            self.action_space = spaces.Box(low=np.float32(self.config['action_range_low'][0]),
                                       high=np.float32(self.config['action_range_high'][0]),
                                       dtype=np.float32)
            self.action_range_scale = self.config['action_range_scale'][0]
        elif self.config['agent']['controller'] == 'PID':
            self.action_dim =  self.config['agent']['controller_config']['PID']['action_dim']
            self.action_space = spaces.Box(low=np.float32(self.config['action_range_low'][1]),
                            high=np.float32(self.config['action_range_high'][1]),
                            dtype=np.float32)
            self.action_range_scale = self.config['action_range_scale'][1]
        else:
            raise ValueError("Unknown controller type: {}".format(self.config['agent']['controller']))
        
        # target_limit for kf belief
        if self.config['target']['target_dim'] == 2:
            self.target_limit = [self.bottom_corner[:2],self.top_corner[:2]]
        elif self.config['target']['target_dim'] == 4:
            self.target_limit = [np.concatenate((self.bottom_corner[:2], np.array([-3, -3]))),
                                 np.concatenate((self.top_corner[:2], np.array([3, 3])))]
        else:
            raise ValueError("Unknown target dimension: {}".format(self.config['target']['target_dim']))
        
        # observation_space:
        # target distance, angle, covariance determinant value, bool; nearest obstacle distance, angle;
        state_lower_bound = np.concatenate(([0.0, -np.pi, -50.0, 0.0] * self.num_targets,
                                            [0.0, -np.pi]))
        state_upper_bound = np.concatenate(([600.0, np.pi, 50.0, 2.0] * self.num_targets,
                                            [self.config['agent']['sensor_r'], np.pi]))
        self.observation_space = spaces.Box(low=state_lower_bound, high=state_upper_bound, dtype=np.float32)

    def update_every_tick(self, sensors):
        # update extra sensors buffer
        if 'LeftCamera' in sensors['auv0']:
            import cv2
            cv2.imshow("Camera Output", sensors['auv0']['LeftCamera'][:, :, 0:3])
            cv2.waitKey(1)
            # self.image_buffer.add_image(sensors['auv0']['LeftCamera'], sensors['t'])
        pass

    def get_reward(self, is_col, action):
        self.action_queue.append(action)
        reward_param = self.config['reward_param']
        detcov = [LA.det(b_target.cov) for b_target in self.belief_targets]
        r_detcov_mean = - np.mean(np.log(detcov))
        r_detcov_std = - np.std(np.log(detcov))

        r_action_smooth = 0.0
        # if len(self.action_queue) >= 2:
        #     action_diff = np.linalg.norm(np.array(action) - np.array(self.action_queue[-2]))
        #     r_action_smooth = np.exp(-action_diff)

        reward = reward_param["c_mean"] * r_detcov_mean + \
                reward_param["c_std"] * r_detcov_std + \
                reward_param["c_smooth"] * r_action_smooth
        if is_col:
            reward = np.min([0.0, reward]) - reward_param["c_penalty"] * 1.0

        self.reward_queue.append(reward)
        done_by_reward = False
        if len(self.reward_queue) == 100:
            avg_reward = np.mean(self.reward_queue)
            if avg_reward < -3.5:
                done_by_reward = True

        done = is_col or done_by_reward

        if self.config['render']:
            print('reward:', reward)
        return reward, done, r_detcov_mean, r_detcov_std

    def state_func(self, observed, action):
        '''
        Called in the parent class's step method to update self.state
        RL state: [d, alpha, log det(Sigma), observed] * nb_targets, [o_d, o_alpha]
        '''
        # Find the closest obstacle coordinate.
        if self.agent.rangefinder.min_distance < self.config['agent']['sensor_r']:
            obstacles_pt = (self.agent.rangefinder.min_distance, np.radians(self.agent.rangefinder.min_angle))
        else:
            obstacles_pt = (self.config['agent']['sensor_r'], 0)

        state_observation = []
        for i in range(self.num_targets):
            r_b, alpha_b = util.relative_distance_polar(
                self.belief_targets[i].state[:2],
                xy_base=self.agent.est_state.vec[:2],
                theta_base=np.radians(self.agent.est_state.vec[8]))
            state_observation.extend([r_b, alpha_b,
                                      np.log(LA.det(self.belief_targets[i].cov)),
                                      float(observed[i])])
        state_observation.extend(obstacles_pt)
        state_observation = np.array(state_observation)

        self.obs = state_observation
        return self.obs

if __name__ == '__main__':
    pass

