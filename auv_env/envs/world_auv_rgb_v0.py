import holoocean
import numpy as np
from numpy import linalg as LA
from auv_control.estimation import KFbelief, UKFbelief
from auv_control.control import LQR
from auv_control.planning import Traj, RRT
from auv_control import State

from auv_env import util
from auv_env.envs.base import WorldBase
from auv_env.envs.agent import AgentAuv, AgentSphere, AgentAuvTarget
from auv_env.envs.obstacle import Obstacle
from auv_env.envs.tools import ImageBuffer
from metadata import METADATA

from gymnasium import spaces
import logging
import copy


class WorldAuvRGBV0(WorldBase):
    """
        different from world:target is also an auv
    """

    def __init__(self, map, show, verbose, num_targets, **kwargs):
        self.obs = {}
        self.image_buffer = ImageBuffer(5, (3, 224, 224), time_gap=0.5)
        super().__init__(map, show, verbose, num_targets, **kwargs)
        
    def reset(self):
        self.image_buffer.reset()
        return super().reset()

    def step(self, action_waypoint):
        global_waypoint = np.zeros(3)
        observed = []
        # 归一化展开
        r = action_waypoint[0] * self.action_range_scale[0]
        theta = action_waypoint[1] * self.action_range_scale[1]
        global_waypoint[:2] = util.polar_distance_global(np.array([r, theta]), self.agent.est_state.vec[:2],
                                                         np.radians(self.agent.est_state.vec[8]))
        angle = action_waypoint[2] * self.action_range_scale[2]
        global_waypoint[2] = self.agent.est_state.vec[8] + np.rad2deg(angle)
        self.agent_w = angle / 0.5
        if self.agent_u is not None:
            self.agent_last_u = self.agent_u
        for j in range(50):
            for i in range(self.num_targets):
                target = 'target'+str(i)
                if self.has_discovered[i]:
                    self.target_u = self.targets[i].update(self.sensors[target], self.sensors['t'])
                    self.ocean.act(target, self.target_u)
                else:
                    self.target_u = np.zeros(8)
                    self.ocean.act(target, self.target_u)
            if j == 0:
                self.agent_u = self.u
            self.u = self.agent.update(global_waypoint, self.fix_depth, self.sensors['auv0'])
            self.ocean.act("auv0", self.u)
            sensors = self.ocean.tick()
            # update
            if 'LeftCamera' in sensors['auv0']:
                self.image_buffer.add_image(sensors['auv0']['LeftCamera'], sensors['t'])
            self.sensors['auv0'].update(sensors['auv0'])
            for i in range(self.num_targets):
                target = 'target'+str(i)
                self.sensors[target].update(sensors[target])


        # The targets are observed by the agent (z_t+1) and the beliefs are updated.
        observed = self.observe_and_update_belief()
        is_col = not (self.obstacles.check_obstacle_collision(self.agent.state.vec[:2], self.margin2wall)
                      and self.in_bound(self.agent.state.vec[:2])
                      and np.linalg.norm(self.agent.state.vec[:2] - self.targets[0].state.vec[:2]) > self.margin)

        # Compute a reward from b_t+1|t+1 or b_t+1|t.
        reward, done, mean_nlogdetcov, std_nlogdetcov = self.get_reward(is_col=is_col)
        # Predict the target for the next step, b_t+2|t+1
        for i in range(self.num_targets):
            self.belief_targets[i].predict()

        # Compute the RL state.
        state = self.state_func(observed, action_waypoint)
        self.record_observed = observed
        if METADATA['render']:
            print(is_col, observed[0], reward)
        self.is_col = is_col
        return state, reward, done, 0, {'mean_nlogdetcov': mean_nlogdetcov, 'std_nlogdetcov': std_nlogdetcov}

    def set_limits(self):
        # LIMIT
        self.limit = {}  # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.bottom_corner[:2], [-np.pi])),
                               np.concatenate((self.top_corner[:2], [np.pi]))]
        self.limit['target'] = [np.concatenate((self.bottom_corner[:2], np.array([-3, -3]))),
                                np.concatenate((self.top_corner[:2], np.array([3, 3])))]
        # ACTION:
        self.action_space = spaces.Box(low=np.float32(METADATA['action_range_low']),
                                       high=np.float32(METADATA['action_range_high']),
                                       dtype=np.float32)
        # STATE:
        # target distance、angle、协方差行列式值、bool; agent 自身定位; last action waypoint;
        state_lower_bound = np.concatenate((np.concatenate(([0.0, -np.pi, -50.0, 0.0] * self.num_targets
                                                            , [self.bottom_corner[0], self.bottom_corner[1], -np.pi])),
                                            [0.0, -np.pi, 0.0, 0.0, 0.0]))
        state_upper_bound = np.concatenate((np.concatenate(([600.0, np.pi, 50.0, 2.0] * self.num_targets,
                                                            [self.top_corner[0], self.top_corner[1], np.pi])),
                                            [METADATA['agent']['sensor_r'], np.pi, 1.0, 1.0, 1.0]))

        # target distance、angle、协方差行列式值、bool;agent 自身定位;
        # self.limit['state'] = [np.concatenate(([0.0, -np.pi, -50.0, 0.0] * self.num_targets, [0.0, -np.pi])),
        #                        np.concatenate(([600.0, np.pi, 50.0, 2.0] * self.num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Dict({
            "images": spaces.Box(low=-3, high=3, shape=(5, 3, 224, 224), dtype=np.float32),
            "state": spaces.Box(low=state_lower_bound, high=state_upper_bound, dtype=np.float32),
        })

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
            obstacles_pt = (self.agent.rangefinder.min_distance, np.radians(self.agent.rangefinder.min_angle))
        else:
            obstacles_pt = (METADATA['agent']['sensor_r'], 0)

        state_observation = []
        for i in range(self.num_targets):
            r_b, alpha_b = util.relative_distance_polar(
                self.belief_targets[i].state[:2],
                xy_base=self.agent.est_state.vec[:2],
                theta_base=np.radians(self.agent.est_state.vec[8]))
            state_observation.extend([r_b, alpha_b,
                                      np.log(LA.det(self.belief_targets[i].cov)),
                                      float(observed[i])])  # dim:4
        state_observation.extend([self.agent.state.vec[0], self.agent.state.vec[1],
                                  np.radians(self.agent.state.vec[8])])  # dim:3
        state_observation.extend(obstacles_pt)  # dim:2
        state_observation.extend(action_waypoint.tolist())  # dim:3
        state_observation = np.array(state_observation)

        images = np.stack(self.image_buffer.get_buffer())
        # images = util.image_preprocess(images)
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
    from auv_control import scenario
    #
    # print("Test World")
    # world = WorldAuvMap(scenario, map='TestMap', show=True, verbose=True, num_targets=1)
    # world.reset()
    # print(world.size)
    # world.targets[0].planner.draw_traj(world.ocean, 30)
    # action_range_high = METADATA['action_range_high']
    # action_range_low = METADATA['action_range_low']
    # action_space = spaces.Box(low=np.float32(action_range_low), high=np.float32(action_range_high)
    #                           , shape=(3,))  # 6维控制 分别是x y theta 的均值和标准差
    # while True:
    #     for _ in range(100000):
    #         # if 'q' in world.agent.keyboard.pressed_keys:
    #         #     break
    #         # command = world.agent.keyboard.parse_keys()
    #         action = action_space.sample()
    #         world.step(action)
    #         # print(world.agent_init_pos, world.sensors['auv0']['PoseSensor'][:3, 3])
    #     world.reset()
    #     world.targets[0].planner.draw_traj(world.ocean, 30)
    # test for camera
    # import cv2
    # if "LeftCamera" in world.sensors['auv0']:
    #     pixels = world.sensors['auv0']["LeftCamera"]
    #     cv2.namedWindow("Camera Output")
    #     cv2.imshow("Camera Output", pixels[:, :, 0:3])
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
