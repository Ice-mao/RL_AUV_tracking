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
from collections import deque

class WorldAuv3DV1(WorldBase3D):
    """
        different from world:target is also an auv
    """
    def __init__(self, config, map, show):
        self.obs = {}
        self.reward_queue = deque(maxlen=100)
        self.image_buffer = CameraBuffer(5, (3, 224, 224), time_gap=0.1)
        super().__init__(config, map, show)

    def reset(self, seed=None, **kwargs):
        self.image_buffer.reset()
        self.reward_queue.clear()
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
            self.action_dim =  self.config['agent']['controller_config']['PID']['action_dim']
            self.action_space = spaces.Box(low=np.float32(self.config['action_range_low'][1]),
                            high=np.float32(self.config['action_range_high'][1]),
                            dtype=np.float32)
            self.action_range_scale = self.config['action_range_scale'][1]
        
        # target_limit for kf belief
        self.target_limit = [np.concatenate((self.bottom_corner, np.array([-1, -1, -1]))),
                                        np.concatenate((self.top_corner, np.array([1, 1, 1])))]
        # observation_space:
        # target distance、angle、协方差行列式值、bool; agent 自身定位;
        state_lower_bound = np.concatenate(([0.0, -np.pi, -np.pi/2, -50.0, 0.0] * self.num_targets,
                                            [0.0, -np.pi, -np.pi/2]))
        state_upper_bound = np.concatenate(([600.0, np.pi, np.pi/2, 50.0, 2.0] * self.num_targets,
                                            [self.config['agent']['sensor_r'], np.pi, np.pi/2]))
        
        self.observation_space = spaces.Dict({
            # "images": spaces.Box(low=0, high=255, shape=(5, 3, 224, 224), dtype=np.float32),
            "images": spaces.Box(low=0, high=1, shape=(3, 224, 224), dtype=np.float32),
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
        在父类的step中调用该函数对self.state进行更新
        RL state: [d, alpha, log det(Sigma), observed] * nb_targets, [o_d, o_alpha]
        '''
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

        # images = np.stack(self.image_buffer.get_buffer()[-1])
        images = np.array(self.image_buffer.get_buffer()[-1])
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0  # 转换为0-1范围
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

