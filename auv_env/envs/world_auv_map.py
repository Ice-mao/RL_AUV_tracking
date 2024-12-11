import holoocean
import numpy as np
from numpy import linalg as LA
from auv_control.control import LQR

from auv_env import util
from auv_env.envs.base import WorldBase
from auv_env.envs.obstacle import Obstacle
from metadata import METADATA

from gymnasium import spaces
import logging


class WorldAuvMap(WorldBase):
    """
        different from world:target is also an auv
    """

    def __init__(self, map, show, verbose, num_targets, **kwargs):
        super().__init__(map, show, verbose, num_targets, **kwargs)

    def set_limits(self):
        # LIMIT
        self.limit = {}  # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.bottom_corner[:2], [-np.pi])),
                               np.concatenate((self.top_corner[:2], [np.pi]))]
        self.limit['target'] = [np.concatenate((self.bottom_corner[:2], np.array([-3, -3]))),
                                np.concatenate((self.top_corner[:2], np.array([3, 3])))]
        # STATE:
        # target distance、angle、协方差行列式值、bool;agent 自身定位;last action; 声呐图像
        state_lower = np.concatenate((np.concatenate(([0.0, -np.pi, -50.0, 0.0] * self.num_targets,
                                                      [self.bottom_corner[0], self.bottom_corner[1], -np.pi])),
                                      [0.0] * 3))
        grid_lower = [0.0] * (64 * 64)
        lower_bound = np.concatenate((state_lower, grid_lower))

        state_upper = np.concatenate((np.concatenate(([600.0, np.pi, 50.0, 2.0] * self.num_targets,
                                                      [self.top_corner[0], self.top_corner[1], np.pi])),
                                      [1.0] * 3))
        grid_upper = [1.0] * (64 * 64)
        upper_bound = np.concatenate((state_upper, grid_upper))
        self.limit['state'] = [lower_bound, upper_bound]

        # target distance、angle、协方差行列式值、bool;agent 自身定位;
        # self.limit['state'] = [np.concatenate(([0.0, -np.pi, -50.0, 0.0] * self.num_targets, [0.0, -np.pi])),
        #                        np.concatenate(([600.0, np.pi, 50.0, 2.0] * self.num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float64)

    def get_reward(self, is_col, reward_param=METADATA['reward_param']):
        detcov = [LA.det(b_target.cov) for b_target in self.belief_targets]
        r_detcov_mean = - np.mean(np.log(detcov))
        r_detcov_std = - np.std(np.log(detcov))

        reward = reward_param["c_mean"] * r_detcov_mean + reward_param["c_std"] * r_detcov_std
        # reward_w = np.exp(-k_3 * np.abs(np.radians(self.agent.state.vec[8]))) - 1
        reward_w = np.exp(-reward_param["k_3"] * np.abs(self.agent_w)) - 1
        if self.agent_last_u is not None:
            reward_a = np.exp(-reward_param["k_4"] * np.sum(np.abs(self.agent_u - self.agent_last_u))) - 1
        else:
            reward_a = -1
        reward_e = np.exp(-reward_param["k_5"] * np.sum([f_i ** 2 for f_i in self.agent_u])) - 1
        reward = reward + reward_w + reward_a + reward_e
        if is_col:
            reward = np.min([0.0, reward]) - reward_param["c_penalty"] * 1.0
        if METADATA['render']:
            print('reward:', reward, 'reward_w:', reward_w, 'reward_a:', reward_a, 'reward_e:', reward_e)
        return reward, False, r_detcov_mean, r_detcov_std

    def state_func(self, observed, action_waypoint):
        '''
        在父类的step中调用该函数对self.state进行更新
        RL state: [d, alpha, log det(Sigma), observed] * nb_targets, [o_d, o_alpha]
        '''
        # Find the closest obstacle coordinate.
        if self.agent.rangefinder.min_distance < METADATA['agent']['sensor_r']:
            obstacles_pt = (self.agent.rangefinder.min_distance, np.radians(self.agent.rangefinder.angle))
        else:
            obstacles_pt = (METADATA['agent']['sensor_r'], 0)

        state = []
        state.extend(self.agent.gridMap.to_grayscale_image().flatten())  # dim:64*64
        for i in range(self.num_targets):
            r_b, alpha_b = util.relative_distance_polar(
                self.belief_targets[i].state[:2],
                xy_base=self.agent.est_state.vec[:2],
                theta_base=np.radians(self.agent.est_state.vec[8]))
            state.extend([r_b, alpha_b,
                          np.log(LA.det(self.belief_targets[i].cov)),
                          float(observed[i])])  # dim:4
        state.extend([self.agent.state.vec[0], self.agent.state.vec[1],
                      np.radians(self.agent.state.vec[8])])  # dim:3
        # self.state.extend(obstacles_pt)
        state.extend(action_waypoint.tolist())  # dim:3

        state = np.array(state)

        # Update the visit map for the evaluation purpose.
        # if self.MAP.visit_map is not None:
        #     self.MAP.update_visit_freq_map(self.agent.state, 1.0, observed=bool(np.mean(observed)))


if __name__ == '__main__':
    from auv_control import scenario

    print("Test World")
    world = WorldAuvMap(scenario, map='TestMap', show=True, verbose=True, num_targets=1)
    world.reset()
    print(world.size)
    world.targets[0].planner.draw_traj(world.ocean, 30)
    action_range_high = METADATA['action_range_high']
    action_range_low = METADATA['action_range_low']
    action_space = spaces.Box(low=np.float32(action_range_low), high=np.float32(action_range_high)
                              , shape=(3,))  # 6维控制 分别是x y theta 的均值和标准差
    while True:
        for _ in range(100000):
            # if 'q' in world.agent.keyboard.pressed_keys:
            #     break
            # command = world.agent.keyboard.parse_keys()
            action = action_space.sample()
            world.step(action)
            # print(world.agent_init_pos, world.sensors['auv0']['PoseSensor'][:3, 3])
        world.reset()
        world.targets[0].planner.draw_traj(world.ocean, 30)
        # test for camera
        # import cv2
        # if "LeftCamera" in world.sensors['auv0']:
        #     pixels = world.sensors['auv0']["LeftCamera"]
        #     cv2.namedWindow("Camera Output")
        #     cv2.imshow("Camera Output", pixels[:, :, 0:3])
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()
