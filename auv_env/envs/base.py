"""
Target Tracking Environment Base Model.
"""
from typing import List
from abc import ABC, abstractmethod

import gymnasium as gym
from gymnasium import spaces, logger
from gymnasium.utils import seeding

import numpy as np
from numpy import linalg as LA
import os, copy

import holoocean

from auv_control import scenario
from auv_control.estimation import KFbelief, UKFbelief

from metadata import METADATA

from auv_env.maps import map_utils
import auv_env.util as util
from auv_env.envs.obstacle import Obstacle
from auv_env.envs.agent import AgentAuv, AgentSphere, AgentAuvTarget


class TargetTrackingBase(gym.Env):
    """
        base class for env creation
    """

    def __init__(self, world_class, map="TestMap", num_targets=1, show_viewport=True, verbose=True, is_training=False, **kwargs):
        gym.Env.__init__(self)
        np.random.seed()
        self.state = None
        # init some params
        self.num_targets = num_targets
        self.is_training = is_training
        self.action_space = spaces.Box(low=np.float32(METADATA['action_range_low']),
                                       high=np.float32(METADATA['action_range_high']),
                                       dtype=np.float32)
        # init the scenario
        self.world = world_class(map=map, show=show_viewport, verbose=verbose, num_targets=self.num_targets)
        # # init the action space
        # self.action_space = self.world.action_space
        self.observation_space = self.world.observation_space
        self.reset_num = 0

    def reset(self, **kwargs):
        np.random.seed()
        self.reset_num += 1
        self.has_discovered = [1] * self.num_targets  # if initial state is observed:1
        self.state = []
        return self.world.reset()

    def step(self, action):
        """
        I want to get a waypoint to guidance the learning,and make the action command
        every 0.5s or more time, so how do I calculate the reward?W
        :param action:
        :return:
        """
        return self.world.step(action_waypoint=action)

    def seed(self, seed):
        np.random.seed(seed)


class WorldBase:
    """
        different from world: target is also an auv
        for v0、v1
    """
    def __init__(self, map, show, verbose, num_targets, **kwargs):
        # define the entity
        # self.ocean = holoocean.make(scenario_cfg=scenario, show_viewport=show, verbose=verbose)
        self.map = map
        self.ocean = holoocean.make(self.map, show_viewport=show)
        scenario = holoocean.get_scenario(self.map)
        self.ocean.should_render_viewport(METADATA['render'])
        self.agent = None
        # init the param
        self.sampling_period = 1 / scenario["ticks_per_sec"]  # sample time
        self.task_random = METADATA['env']['task_random']
        self.control_period = METADATA['env']['control_period']
        self.num_targets = num_targets  # num of target
        self.action_range_scale = METADATA['action_range_scale']
        self.noblock = METADATA['env']['noblock']
        self.insight = METADATA['env']['insight'][0]
        if self.insight:
            self.has_discovered = [1] * self.num_targets  # Set to 0 values for your evaluation purpose.
        else:
            self.has_discovered = [0] * self.num_targets  # Set to 0 values for your evaluation purpose.
        # for record
        self.record_cov_posterior = []
        self.record_observed = []

        # Setup environment
        margin = 0.25
        self.size = np.array([METADATA['scenario']['size'][0] - 2 * margin,
                              METADATA['scenario']['size'][1] - 2 * margin,
                              METADATA['scenario']['size'][2]])
        self.bottom_corner = np.array([METADATA['scenario']['bottom_corner'][0] + margin,
                                       METADATA['scenario']['bottom_corner'][1] + margin,
                                       METADATA['scenario']['bottom_corner'][2]])
        self.fix_depth = METADATA['scenario']['fix_depth']
        self.margin = METADATA['env']['margin']
        self.margin2wall = METADATA['env']['margin2wall']
        # self.ocean.draw_box(self.center.tolist(), (self.size / 2).tolist(), color=[0, 0, 255], thickness=30,
        #                     lifetime=0)  # draw the area

        # Setup obstacles
        # rule is obstacles combined will rotate from their own center
        self.obstacles = Obstacle(self.ocean, self.fix_depth)

        # Record for reward obtain(diff from the control period and the sampling period)
        self.agent_w = None
        # for u calculate:when receive the new waypoints
        self.agent_last_u = None
        self.agent_u = None
        self.sensors = {}
        self.set_limits()
        # Cal random  pos of agent and target
        self.reset()

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

    def build_models(self, sampling_period, agent_init_state, target_init_state, time, **kwargs):
        """
        :param sampling_period:
        :param agent_init_state:list [[x,y,z],yaw(theta)]
        :param target_init_state:list [[x,y,z],yaw(theta)]
        :param kwargs:
        :return:
        """
        # Build a robot
        self.agent = AgentAuv(dim=3, sampling_period=sampling_period, sensor=agent_init_state,
                              scenario=self.map)
        self.targets = [AgentAuvTarget(dim=3, sampling_period=sampling_period, sensor=target_init_state, rank=i
                                       , obstacles=self.obstacles, fixed_depth=self.fix_depth, size=self.size,
                                       bottom_corner=self.bottom_corner, start_time=time, scene=self.ocean,
                                       l_p=METADATA['target']['lqr_l_p'])
                        for i in range(self.num_targets)]
        # Build target beliefs.
        if METADATA['target']['random']:
            self.const_q = np.random.choice(METADATA['target']['const_q'][1])
        else:
            self.const_q = METADATA['target']['const_q'][0]
        self.targetA = np.concatenate((np.concatenate((np.eye(2),
                                                       self.control_period * np.eye(2)), axis=1),
                                       [[0, 0, 1, 0], [0, 0, 0, 1]]))
        self.target_noise_cov = self.const_q * np.concatenate((
            np.concatenate((self.control_period ** 3 / 3 * np.eye(2),
                            self.control_period ** 2 / 2 * np.eye(2)), axis=1),
            np.concatenate((self.control_period ** 2 / 2 * np.eye(2),
                            self.control_period * np.eye(2)), axis=1)))
        self.belief_targets = [KFbelief(dim=METADATA['target']['target_dim'],
                                        limit=self.limit['target'], A=self.targetA,
                                        W=self.target_noise_cov,
                                        obs_noise_func=self.observation_noise)
                               for _ in range(self.num_targets)]

    def reset(self, action_dim=3):
        self.ocean.reset()
        if METADATA['render']:
            self.ocean.draw_box(self.center.tolist(), (self.size / 2).tolist(), color=[0, 0, 255], thickness=30,
                                lifetime=0)  # draw the area
        self.obstacles.reset()
        self.obstacles.draw_obstacle()

        if self.task_random:
            self.insight = np.random.choice(METADATA['env']['insight'][1])
        else:
            self.insight = METADATA['env']['insight'][0]
        print("insight is :", self.insight)
        if self.insight:
            self.has_discovered = [1] * self.num_targets  # Set to 0 values for your evaluation purpose.
        else:
            self.has_discovered = [0] * self.num_targets  # Set to 0 values for your evaluation purpose.
        # reset the reward record
        self.agent_w = None
        self.agent_last_state = None
        self.agent_last_u = None
        # reset the random position

        # Cal random pos of agent and target
        self.agent_init_pos = None
        self.agent_init_yaw = None
        self.target_init_pos = None
        self.target_init_yaw = None
        self.agent_init_pos, self.agent_init_yaw, self.target_init_pos, self.target_init_yaw, self.belief_init_pos \
            = self.get_init_pose_random()

        if METADATA['eval_fixed']:
            self.agent_init_pos = np.array([-12.05380736, -17.06450028, -5.])
            self.agent_init_yaw = -0.9176929024434316
            self.target_init_pos = np.array([-8.92739928, -17.99615254, -5.])
            self.target_init_yaw = -0.28961582668513486

        print(self.agent_init_pos, self.agent_init_yaw)
        print(self.target_init_pos, self.target_init_yaw)

        # Set the pos and tick the scenario
        # self.ocean.agents['auv0'].set_physics_state(location=self.agent_init_pos,
        #                                             rotation=[0.0, 0.0, np.rad2deg(self.agent_init_yaw)],
        #                                             velocity=[0.0, 0.0, 0.0],
        #                                             angular_velocity=[0.0, 0.0, 0.0])
        self.ocean.agents['auv0'].teleport(location=self.agent_init_pos,
                                           rotation=[0.0, 0.0, np.rad2deg(self.agent_init_yaw)])
        self.u = np.zeros(8)
        self.ocean.act("auv0", self.u)

        for i in range(self.num_targets):
            target = 'target' + str(i)
            # self.ocean.agents['target'].set_physics_state(location=self.target_init_pos,
            #                                               rotation=[0.0, 0.0, -np.rad2deg(self.target_init_yaw)],
            #                                               velocity=[0.0, 0.0, 0.0],
            #                                               angular_velocity=[0.0, 0.0, 0.0])
            self.ocean.agents[target].teleport(location=self.target_init_pos,
                                               rotation=[0.0, 0.0, np.rad2deg(self.target_init_yaw)])
            self.target_u = np.zeros(8)
            self.ocean.act(target, self.target_u)

        sensors = self.ocean.tick()
        self.sensors.update(sensors)

        self.build_models(sampling_period=self.sampling_period,
                          agent_init_state=self.sensors['auv0'],
                          target_init_state=self.sensors['target0'],  # TODO
                          time=self.sensors['t'])
        # reset model
        self.agent.reset(self.sensors['auv0'])
        for i in range(self.num_targets):
            target = 'target' + str(i)
            self.targets[i].reset(self.sensors[target], obstacles=self.obstacles,
                                  scene=self.ocean, start_time=self.sensors['t'])
            self.belief_targets[i].reset(
                init_state=np.concatenate((self.belief_init_pos[:2], np.zeros(2))),
                # init_state=np.concatenate((np.array([-10, -10]), np.zeros(2))),
                init_cov=METADATA['target']['target_init_cov'])

        # The targets are observed by the agent (z_0) and the beliefs are updated (b_0).
        observed = self.observe_and_update_belief()

        # Predict the target for the next step, b_1|0.
        for i in range(self.num_targets):
            self.belief_targets[i].predict()

        observed = [True]
        # Compute the RL state.
        state = self.state_func(observed, action_waypoint=np.zeros(action_dim))
        info = {'reset_info': 'yes'}
        return state, info

    @property
    def center(self):
        return self.bottom_corner + self.size / 2

    @property
    def top_corner(self):
        return self.bottom_corner + self.size

    def get_init_pose_random(self,
                             lin_dist_range_a2t=METADATA['target']['lin_dist_range_a2t'],
                             ang_dist_range_a2t=METADATA['target']['ang_dist_range_a2t'],
                             lin_dist_range_t2b=METADATA['target']['lin_dist_range_t2b'],
                             ang_dist_range_t2b=METADATA['target']['ang_dist_range_t2b'],
                             blocked=None, ):
        is_agent_valid = False
        print(self.insight)
        while not is_agent_valid:
            init_pose = {}
            np.random.seed()
            # generatr an init pos around the map
            agent_init_pos = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
            agent_init_yaw = np.random.uniform(-np.pi / 2, np.pi / 2)
            # satisfy the in bound and no collision conditions ----> True(is valid)
            # give a more safe init position
            is_agent_valid = self.in_bound(agent_init_pos) and \
                (not hasattr(self, 'obstacles') or 
                    self.obstacles.check_obstacle_collision(agent_init_pos, self.margin2wall + 2))
            if is_agent_valid:
                for i in range(self.num_targets):
                    count = 0
                    is_target_valid, target_init_pos, target_init_yaw = False, np.zeros((2,)), np.zeros((1,))
                    while not is_target_valid:
                        if self.insight:
                            is_target_valid, target_init_pos, target_init_yaw = self.gen_rand_pose(
                                agent_init_pos,
                                agent_init_yaw,
                                lin_dist_range_a2t[0], lin_dist_range_a2t[1],
                                ang_dist_range_a2t[0], ang_dist_range_a2t[1]
                            )
                            if is_target_valid:  # check the blocked condition
                                is_no_blocked = not hasattr(self, "obstacles") or \
                                    self.obstacles.check_obstacle_block(agent_init_pos, target_init_pos, self.margin2wall + 2)
                                is_target_valid = (self.noblock == is_no_blocked)
                        elif not self.insight:
                            target_init_pos = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
                            target_init_yaw = np.random.uniform(-np.pi / 2, np.pi / 2)
                            # confirm is not insight
                            r, alpha = util.relative_distance_polar(target_init_pos, agent_init_pos, agent_init_yaw)
                            if (r > METADATA['agent']['sensor_r'] or np.rad2deg(alpha) > METADATA['agent']['fov'] / 2
                                    or np.rad2deg(alpha) < -METADATA['agent']['fov'] / 2):
                                is_not_insight = True
                            else:
                                is_not_insight = False
                            is_target_valid = (
                                    self.in_bound(target_init_pos) and
                                    (not hasattr(self, 'obstacles') or
                                        self.obstacles.check_obstacle_collision(target_init_pos, self.margin2wall + 2)) and
                                    np.linalg.norm(target_init_pos - agent_init_pos) > self.margin and
                                    is_not_insight)
                        count += 1
                        if count > 100:
                            is_agent_valid = False
                            count = 0
                            break

                    count = 0
                    is_belief_valid, belief_init_pos = False, np.zeros((2,))
                    while not is_belief_valid:
                        if self.insight:
                            is_belief_valid, init_pose_belief, _ = self.gen_rand_pose(
                                target_init_pos[:2], target_init_yaw,
                                lin_dist_range_t2b[0], lin_dist_range_t2b[1],
                                ang_dist_range_t2b[0], ang_dist_range_t2b[1])
                            # if is_belief_valid and (blocked is not None):
                            #     is_no_blocked = self.obstacles.check_obstacle_block(agent_init_pos, target_init_pos,
                            #                                                         self.margin2wall + 2)
                            #     is_belief_valid = (self.noblock == is_no_blocked)
                        elif not self.insight:
                            is_belief_valid = True
                            init_pose_belief = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]

                        count += 1
                        if count > 100:
                            is_agent_valid = False
                            break

        return (np.append(agent_init_pos, self.fix_depth), agent_init_yaw,
                np.append(target_init_pos, self.fix_depth), target_init_yaw,
                init_pose_belief)

    def in_bound(self, pos):
        """
        :param pos:
        :return: True: in area, False: out area
        """
        return not ((pos[0] < self.bottom_corner[0] + self.margin2wall)
                    or (pos[0] > self.size[0] + self.bottom_corner[0] - self.margin2wall)
                    or (pos[1] < self.bottom_corner[1] + self.margin2wall)
                    or (pos[1] > self.size[1] + self.bottom_corner[1] - self.margin2wall))

    def gen_rand_pose(self, frame_xy, frame_theta, min_lin_dist, max_lin_dist,
                      min_ang_dist, max_ang_dist):
        """Genertes random position and yaw.
        Parameters
        --------
        frame_xy, frame_theta : xy and theta coordinate of the frame you want to compute a distance from.
        min_lin_dist : the minimum linear distance from o_xy to a sample point.
        max_lin_dist : the maximum linear distance from o_xy to a sample point.
        min_ang_dist : the minimum angular distance (counter clockwise direction) from c_theta to a sample point.
        max_ang_dist : the maximum angular distance (counter clockwise direction) from c_theta to a sample point.
        """
        if max_ang_dist < min_ang_dist:
            max_ang_dist += 2 * np.pi
        rand_ang = util.wrap_around(np.random.rand() *
                                    (max_ang_dist - min_ang_dist) + min_ang_dist)

        rand_r = np.random.rand() * (max_lin_dist - min_lin_dist) + min_lin_dist
        rand_xy = np.array([rand_r * np.cos(rand_ang), rand_r * np.sin(rand_ang)])
        rand_xy_global = util.transform_2d_inv(rand_xy, frame_theta, np.array(frame_xy))
        is_valid = (self.in_bound(rand_xy_global) and 
            (not hasattr(self, 'obstacles') or 
             self.obstacles.check_obstacle_collision(rand_xy_global, self.margin2wall + 2)))
        # is_valid = (self.in_bound(rand_xy_global) and self.obstacles.check_obstacle_collision(rand_xy_global,
        #                                                                                       self.margin2wall + 2))
        return is_valid, rand_xy_global, util.wrap_around(rand_ang + frame_theta)

    @abstractmethod
    def set_limits(self):
        '''
        you should define your limit due to the cfg
        like self.limit['agent'], self.limit['target'], self.limit['state'], self.observation_space
        :return:
        '''

    def observation_noise(self, z):
        # 测量噪声矩阵，假设独立
        obs_noise_cov = np.array([[METADATA['agent']['sensor_r_sd'] * METADATA['agent']['sensor_r_sd'], 0.0],
                                  [0.0, METADATA['agent']['sensor_b_sd'] * METADATA['agent']['sensor_b_sd']]])
        return obs_noise_cov

    def observation(self, target):
        """
        返回是否观测到目标，以及测量值
        """
        r, alpha = util.relative_distance_polar(target.state.vec[:2],
                                                xy_base=self.agent.state.vec[:2],
                                                theta_base=np.radians(self.agent.state.vec[8]))
        # 判断是否观察到目标
        observed = (r <= METADATA['agent']['sensor_r']) \
                   & (abs(alpha) <= METADATA['agent']['fov'] / 2 / 180 * np.pi) \
                   & self.obstacles.check_obstacle_block(target.state.vec[:2], self.agent.state.vec[:2],
                                                         self.margin)
        z = None
        if observed:
            z = np.array([r, alpha])
            z += np.random.multivariate_normal(np.zeros(2, ), self.observation_noise(z))  # 加入噪声
        return observed, z

    def observe_and_update_belief(self):
        observed = []
        self.record_cov_posterior = []
        for i in range(self.num_targets):
            observation = self.observation(self.targets[i])
            observed.append(observation[0])
            if observation[0]:  # if observed, update the target belief.
                # we use truth
                self.belief_targets[i].update(observation[1],
                                              np.array([self.agent.est_state.vec[0], self.agent.est_state.vec[1],
                                                        np.radians(self.agent.est_state.vec[8])]))
                if not (self.has_discovered[i]):
                    self.has_discovered[i] = 1
            self.record_cov_posterior.append(self.belief_targets[i].cov)
        return observed

    from typing import Dict
    @abstractmethod
    def get_reward(self, is_col: bool, params: Dict[str, float]):
        """
        calulate the reward should return
        :param is_col:
        :param params:
        :return:
        """

    from typing import Union
    @abstractmethod
    def state_func(self, observed, action_waypoint) -> Union[np.ndarray, dict]:
        """
            should define your own state_func when you inherit the child class.
            just an example below
        """
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
        return state
        # Update the visit map for the evaluation purpose.
        # if self.MAP.visit_map is not None:
        #     self.MAP.update_visit_freq_map(self.agent.state, 1.0, observed=bool(np.mean(observed)))


if __name__ == '__main__':
    env = TargetTrackingBase()
    env.world.ocean.should_render_viewport(False)
    obs, _ = env.reset()
    while True:
        action = [1 * np.random.rand(), 1 * np.random.rand(), 0.1 * np.random.rand(),
                  0.01 * np.random.rand(), 0.01 * np.random.rand(), 0.01 * np.random.rand()]
        obs, reward, done, _, inf = env.step(action)
        print(reward)
