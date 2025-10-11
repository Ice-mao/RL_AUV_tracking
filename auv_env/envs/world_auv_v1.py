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

class WorldAuvV1(WorldBase):
    """
        different from world:target is also an auv
    """
    def __init__(self, config, map, show):
        self.obs = {}
        self.image_buffer = CameraBuffer(5, (3, 224, 224), time_gap=0.5)
        super().__init__(config, map, show)

    def reset(self, seed=None, **kwargs):
        self.image_buffer.reset()
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
        # target distance, angle, covariance determinant value, bool; agent self-localization;
        state_lower_bound = np.concatenate(([0.0, -np.pi, -50.0, 0.0] * self.num_targets,
                                                            # [self.bottom_corner[0], self.bottom_corner[1], -np.pi])),
                                            [0.0, -np.pi]))
        state_upper_bound = np.concatenate(([600.0, np.pi, 50.0, 2.0] * self.num_targets,
                                                            # [self.top_corner[0], self.top_corner[1], np.pi])),
                                            [self.config['agent']['sensor_r'], np.pi]))

        # target distance, angle, covariance determinant value, bool; agent self-localization;
        # self.limit['state'] = [np.concatenate(([0.0, -np.pi, -50.0, 0.0] * self.num_targets, [0.0, -np.pi])),
        #                        np.concatenate(([600.0, np.pi, 50.0, 2.0] * self.num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Dict({
            "images": spaces.Box(low=0, high=255, shape=(3, 224, 224), dtype=np.float32),
            "state": spaces.Box(low=state_lower_bound, high=state_upper_bound, dtype=np.float32),
        })

    def update_every_tick(self, sensors):
        # update
        if 'LeftCamera' in sensors['auv0']:
            if self.config['render']:
                import cv2
                cv2.imshow("Camera Output", sensors['auv0']['LeftCamera'][:, :, 0:3])
                cv2.waitKey(1)
            self.image_buffer.add_image(sensors['auv0']['LeftCamera'], sensors['t'])

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
                                      float(observed[i])])  # dim:4
        # state_observation.extend([self.agent.state.vec[0], self.agent.state.vec[1],
        #                           np.radians(self.agent.state.vec[8])])  # dim:3
        state_observation.extend(obstacles_pt)
        # state_observation.extend(action.tolist())  # dim:3
        state_observation = np.array(state_observation)

        # images = np.stack(self.image_buffer.get_buffer()[-1])
        images = self.image_buffer.get_buffer()[-1]
        self.obs = {'images': images, 'state': state_observation}
        return copy.deepcopy({'images': images, 'state': state_observation})
        # Update the visit map for the evaluation purpose.
        # if self.MAP.visit_map is not None:
        #     self.MAP.update_visit_freq_map(self.agent.state, 1.0, observed=bool(np.mean(observed)))

    def state_func_images(self):
        return self.obs['images']

    def state_func_state(self):
        return self.obs['state']

if __name__ == '__main__':
    pass

