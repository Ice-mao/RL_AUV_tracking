import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import holoocean
import numpy as np
from numpy import linalg as LA
from auv_control.estimation import KFbelief, UKFbelief
from auv_control.control import LQR
from auv_control.planning import Traj, RRT
from auv_control import State

from auv_env import util
from auv_env.envs.base_3d import WorldBase3D
from auv_env.envs.agent import AgentAuv, AgentAuvTarget
from auv_env.envs.obstacle import Obstacle
from auv_env.envs.tools import CameraBuffer

from gymnasium import spaces
import logging
import copy

class WorldAuv3DV0(WorldBase3D):
    """
    3D AUV tracking environment with spherical observations
    """
    def __init__(self, config, map, show):
        self.obs = {}
        super().__init__(config, map, show)

    def reset(self, seed=None, **kwargs):
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
        
        # target_limit for kf belief (3D case)
        # 6D state: [x, y, z, vx, vy, vz]
        self.target_limit = [np.concatenate((self.bottom_corner, np.array([-1, -1, -1]))),
                                np.concatenate((self.top_corner, np.array([1, 1, 1])))]
        
        # 3D observation: [r, theta, gamma, log_det_cov, observed] * nb_targets, [o_r, o_theta, o_gamma]
        state_lower_bound = np.concatenate(([0.0, -np.pi, -np.pi/2, -50.0, 0.0] * self.num_targets,
                                            [0.0, -np.pi, -np.pi/2]))
        state_upper_bound = np.concatenate(([600.0, np.pi, np.pi/2, 50.0, 2.0] * self.num_targets,
                                            [self.config['agent']['sensor_r'], np.pi, np.pi/2]))

        self.observation_space = spaces.Box(low=state_lower_bound, high=state_upper_bound, dtype=np.float32)

    def update_every_tick(self, sensors):
        # update extra sensors buffer
        pass

    def get_reward(self, is_col, action):
        reward_param = self.config['reward_param']
        detcov = [LA.det(b_target.cov) for b_target in self.belief_targets]
        r_detcov_mean = - np.mean(np.log(detcov))
        r_detcov_std = - np.std(np.log(detcov))

        reward = reward_param["c_mean"] * r_detcov_mean + reward_param["c_std"] * r_detcov_std
        if is_col:
            reward = np.min([0.0, reward]) - reward_param["c_penalty"] * 1.0
        if self.config['render']:
            print('reward:', reward)
        return reward, False, r_detcov_mean, r_detcov_std

    def state_func(self, observed, action):
        '''
        在父类的step中调用该函数对self.state进行更新
        For 3D: RL state: [r, theta, gamma, log det(Sigma), observed] * nb_targets, [o_r, o_theta, o_gamma]
        For 2D: RL state: [d, alpha, log det(Sigma), observed] * nb_targets, [o_d, o_alpha]
        '''
        # 3D case: use spherical coordinates for obstacles
        if self.agent.rangefinder.min_distance < self.config['agent']['sensor_r']:
            # In 3D, we need to extend this to include elevation angle
            # For now, assume obstacle is at same height (gamma=0)
            obstacles_pt = (self.agent.rangefinder.min_distance, 
                            np.radians(self.agent.rangefinder.min_angle), 
                            0.0)  # elevation = 0 for ground obstacles
        else:
            obstacles_pt = (self.config['agent']['sensor_r'], 0.0, 0.0)

        state_observation = []
        for i in range(self.num_targets):
            if self.config['target']['target_dim'] == 6:
                # 3D spherical coordinates
                r_b, theta_b, gamma_b = util.relative_distance_spherical(
                    self.belief_targets[i].state[:3],
                    xyz_base=self.agent.est_state.vec[:3],
                    theta_base=np.radians(self.agent.est_state.vec[8]))
                state_observation.extend([r_b, theta_b, gamma_b,
                                          np.log(LA.det(self.belief_targets[i].cov)),
                                          float(observed[i])])
        
        state_observation.extend(obstacles_pt)
        state_observation = np.array(state_observation)
        
        self.obs = state_observation
        return copy.deepcopy(state_observation)
    
if __name__ == '__main__':
    print('test')
    from config_loader import load_config
    config = load_config('configs/envs/3d_v0_config.yml')
    world = WorldAuv3DV0(config=config, map='SimpleUnderwater-Bluerov2', show=True)
    world.reset()


