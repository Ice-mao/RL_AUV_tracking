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
from auv_control.control import CmdVel
from auv_control.estimation import KFbelief, UKFbelief

from auv_env.maps import map_utils
import auv_env.util as util
from auv_env.envs.obstacle_3d import Obstacle3D
from auv_env.envs.agent import AgentAuv, AgentAuvTarget3D, AgentAuvManual
from auv_env.envs.base import WorldBase

class WorldBase3D:
    """
    3D Target Tracking World - extends WorldBase for 3D tracking capabilities
    Different from 2D world: includes depth control and 3D observations
    """
    
    def __init__(self, config, map, show):
        self.map = map
        self.ocean = holoocean.make(self.map, show_viewport=show)
        scenario = holoocean.get_scenario(self.map)
        self.config = config
        self.agent = None
        # init the param
        self.sampling_period = 1 / scenario["ticks_per_sec"]  # sample time
        self.task_random = self.config['env']['task_random']
        self.control_period = self.config['env']['control_period']
        self.num_targets = self.config['env']['num_targets']  # num of targets
        self.noblock = self.config['env']['noblock']
        self.insight = self.config['env']['insight'][0]
        if self.insight:
            self.has_discovered = [1] * self.num_targets  # Set to 0 values for your evaluation purpose.
        else:
            self.has_discovered = [0] * self.num_targets  # Set to 0 values for your evaluation purpose.
        # for record
        self.record_cov_posterior = []
        self.record_observed = []

        # Setup environment
        margin = 0.25
        self.size = np.array([self.config['scenario']['size'][0] - 2 * margin,
                              self.config['scenario']['size'][1] - 2 * margin,
                              self.config['scenario']['size'][2]])
        self.bottom_corner = np.array([self.config['scenario']['bottom_corner'][0] + margin,
                                       self.config['scenario']['bottom_corner'][1] + margin,
                                       self.config['scenario']['bottom_corner'][2]])
        self.fix_depth = self.config['scenario']['fix_depth']
        self.margin = self.config['env']['margin']
        self.margin2wall = self.config['env']['margin2wall']

        # Setup obstacles
        self.obstacles = Obstacle3D(self.ocean, self.fix_depth, self.config)
        if self.config['agent']['controller'] == 'LQR':
            self.controller = 'LQR'
        elif self.config['agent']['controller'] == 'PID':
            self.controller = 'PID'
        elif self.config['agent']['controller'] == 'KEYBOARD':
            self.controller = 'KEYBOARD'
        self.ticks_per_rl_step = 1/(self.config['agent']['controller_config'][self.controller]['control_frequency'] * self.sampling_period)
        self.sensors = {}
        self.set_limits()

    def step(self, action):
        if self.controller == 'LQR':
            # waypoint for LQR
            global_waypoint = np.zeros(4) # [x, y, z, yaw]
            observed = []
            r = action[0] * self.action_range_scale[0]
            theta = action[1] * self.action_range_scale[1]
            depth = action[2] * self.action_range_scale[2]
            yaw = action[3] * self.action_range_scale[3]
            global_waypoint[:2] = util.polar_distance_global(np.array([r, theta]), self.agent.est_state.vec[:2],
                                                            np.radians(self.agent.est_state.vec[8]))
            global_waypoint[2] = self.agent.est_state.vec[2] + depth
            global_waypoint[3] = self.agent.est_state.vec[8] + np.rad2deg(yaw)
            self.action = global_waypoint

        elif self.controller == 'PID':
            # cmd_vel for PID
            cmd_vel = CmdVel()
            cmd_vel.linear.x = action[0] * self.action_range_scale[0]
            cmd_vel.angular.z = action[1] * self.action_range_scale[1]
            cmd_vel.linear.z = action[2] * self.action_range_scale[2]  # for depth control(unusual)
            self.action = cmd_vel
        else:
            self.action = np.empty(self.action_dim)

        for _ in range(int(self.ticks_per_rl_step)):
            for i in range(self.num_targets):
                target = 'target'+str(i)
                self.target_u = self.targets[i].update(self.sensors[target], self.sensors['t'])
                self.ocean.act(target, self.target_u)
            self.u = self.agent.update(self.action, depth=None, sensors=self.sensors['auv0'])
            self.ocean.act("auv0", self.u)
            sensors = self.ocean.tick()
            # update
            self.sensors['auv0'].update(sensors['auv0'])
            for i in range(self.num_targets):
                target = 'target'+str(i)
                self.sensors[target].update(sensors[target])
            self.update_every_tick(sensors)

        # The targets are observed by the agent (z_t+1) and the beliefs are updated.
        observed = self.observe_and_update_belief()
        is_col = not (self.obstacles.check_obstacle_collision(self.agent.state.vec[:3], self.margin2wall)
                      and self.in_bound(self.agent.state.vec[:3])
                      and np.linalg.norm(self.agent.state.vec[:3] - self.targets[0].state.vec[:3]) > self.margin)

        # Compute a reward from b_t+1|t+1 or b_t+1|t.
        reward, done, mean_nlogdetcov, std_nlogdetcov = self.get_reward(is_col=is_col, action=self.action)
        # Predict the target for the next step, b_t+2|t+1
        for i in range(self.num_targets):
            self.belief_targets[i].predict()

        # Compute the RL state.
        state = self.state_func(observed, action)
        self.record_observed = observed
        if self.config['render']:
            print(is_col, observed[0], reward)
        self.is_col = is_col
        return state, reward, done, 0, {'mean_nlogdetcov': mean_nlogdetcov, 'std_nlogdetcov': std_nlogdetcov}

    def build_models(self, sampling_period, agent_init_state, target_init_state, time):
        """
        :param sampling_period:
        :param agent_init_state:list [[x,y,z],yaw(theta)]
        :param target_init_state:list [[x,y,z],yaw(theta)]
        :return:
        """
        # Build a robot
        self.agent = AgentAuv(dim=3, sampling_period=sampling_period, sensor=agent_init_state,
                              scenario=self.map, config=self.config)
        if self.config['target']['controller'] == 'Manual':
            self.targets = [AgentAuvManual(dim=3, sampling_period=sampling_period, fixed_depth=self.fix_depth,
                    sensor=target_init_state,
                    scene=self.ocean, config=self.config)
                for _ in range(self.num_targets)]
        else:
            self.targets = [AgentAuvTarget3D(dim=3, sampling_period=sampling_period, sensor=target_init_state, rank=i,
                        obstacles=self.obstacles, size=self.size,
                        bottom_corner=self.bottom_corner, start_time=time, scene=self.ocean, scenario=self.map, config=self.config)
                for i in range(self.num_targets)]
        # Build target beliefs.
        if self.config['target']['random']:
            self.const_q = np.random.choice(self.config['target']['const_q'][1])
        else:
            self.const_q = self.config['target']['const_q'][0]
        self.targetA = np.concatenate((
            np.concatenate((np.eye(3), self.control_period * np.eye(3)), axis=1),
            np.concatenate((np.zeros((3,3)), np.eye(3)), axis=1)
        ))
        self.target_noise_cov = self.const_q * np.concatenate((
            np.concatenate((self.control_period ** 3 / 3 * np.eye(3),
                            self.control_period ** 2 / 2 * np.eye(3)), axis=1),
            np.concatenate((self.control_period ** 2 / 2 * np.eye(3),
                            self.control_period * np.eye(3)), axis=1)))
        self.belief_targets = [KFbelief(dim=6,
                                        limit=self.target_limit, A=self.targetA,
                                        W=self.target_noise_cov,
                                        obs_noise_func=self.observation_noise)
                               for _ in range(self.num_targets)]

    def reset(self, seed=None, **kwargs):
        self.ocean.reset()
        if self.config['draw_traj']:
            self.ocean.draw_box(self.center.tolist(), (self.size / 2).tolist(), color=[0, 0, 255], thickness=30,
                                lifetime=0)  # draw the area
        self.obstacles.reset()
        self.obstacles.draw_obstacle()

        if self.task_random:
            self.insight = np.random.choice(self.config['env']['insight'][1])
        else:
            self.insight = self.config['env']['insight'][0]
        print("insight is :", self.insight)
        if self.insight:
            self.has_discovered = [1] * self.num_targets  # Set to 0 values for your evaluation purpose.
        else:
            self.has_discovered = [0] * self.num_targets  # Set to 0 values for your evaluation purpose.
        # reset the reward record
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

        if self.config['eval_fixed']:
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
                          target_init_state=self.sensors['target0'],
                          time=self.sensors['t'])
        # reset model
        self.agent.reset(self.sensors['auv0'])
        for i in range(self.num_targets):
            target = 'target' + str(i)
            self.targets[i].reset(self.sensors[target], obstacles=self.obstacles,
                                  scene=self.ocean, start_time=self.sensors['t'])
            self.belief_targets[i].reset(
                init_state=np.concatenate((self.belief_init_pos, np.zeros(3))),  # 3D: [x,y,z,vx,vy,vz]
                init_cov=self.config['target']['target_init_cov'])

        # The targets are observed by the agent (z_0) and the beliefs are updated (b_0).
        observed = self.observe_and_update_belief()

        # Predict the target for the next step, b_1|0.
        for i in range(self.num_targets):
            self.belief_targets[i].predict()

        observed = [True]
        # Compute the RL state.
        state = self.state_func(observed, action=np.zeros(self.action_dim))
        info = {'reset_info': 'yes'}
        return state, info

    @property
    def center(self):
        return self.bottom_corner + self.size / 2

    @property
    def top_corner(self):
        return self.bottom_corner + self.size

    def get_init_pose_random(self, blocked=None):
        """
        3D version of random pose generation
        Supports depth dimension and vertical FOV checking
        """
        # a:agent, t: target, b: belief
        lin_dist_range_a2t = self.config['target']['lin_dist_range_a2t']
        ang_dist_range_a2t = self.config['target']['ang_dist_range_a2t']
        lin_dist_range_t2b = self.config['target']['lin_dist_range_t2b']
        ang_dist_range_t2b = self.config['target']['ang_dist_range_t2b']
        
        # 3D specific distance ranges
        h_fov = self.config['agent']['h_fov'] * np.pi / 180.0

        is_agent_valid = False
        print(self.insight)
        while not is_agent_valid:
            np.random.seed()
            # Generate 3D init pos around the map
            agent_init_pos = np.random.random((3,)) * self.size + self.bottom_corner
            agent_init_yaw = np.random.uniform(-np.pi / 2, np.pi / 2)
            
            # Check 3D validity: in bound and no collision
            is_agent_valid = self.in_bound(agent_init_pos) and \
                (not hasattr(self, 'obstacles') or 
                    self.obstacles.check_obstacle_collision(agent_init_pos, self.margin2wall + 2)) # TODO
            
            if is_agent_valid:
                for i in range(self.num_targets):
                    count = 0
                    is_target_valid, target_init_pos, target_init_yaw = False, np.zeros((3,)), np.zeros((1,))
                    
                    while not is_target_valid:
                        if self.insight:
                            is_target_valid, target_init_pos, target_init_yaw = self.gen_rand_pose(
                                agent_init_pos,
                                agent_init_yaw,
                                lin_dist_range_a2t[0], lin_dist_range_a2t[1],
                                ang_dist_range_a2t[0], ang_dist_range_a2t[1],
                                h_fov
                            )
                            if is_target_valid:  # check the blocked condition
                                is_no_blocked = not hasattr(self, "obstacles") or \
                                    self.obstacles.check_obstacle_block(agent_init_pos, target_init_pos, self.margin2wall + 2)
                                is_target_valid = (self.noblock == is_no_blocked)
                                
                        elif not self.insight:
                            # Generate random 3D position
                            target_init_pos = np.random.random((3,)) * self.size + self.bottom_corner
                            target_init_yaw = np.random.uniform(-np.pi / 2, np.pi / 2)
                            
                            # Check 3D insight conditions
                            is_not_insight = self.check_not_in_3d_fov(agent_init_pos, target_init_pos, agent_init_yaw)
                            
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
                    is_belief_valid, belief_init_pos = False, np.zeros((3,))
                    while not is_belief_valid:
                        if self.insight:
                            is_belief_valid, belief_init_pos, _ = self.gen_rand_pose(
                                target_init_pos, target_init_yaw,
                                lin_dist_range_t2b[0], lin_dist_range_t2b[1],
                                ang_dist_range_t2b[0], ang_dist_range_t2b[1],
                                h_fov)
                        elif not self.insight:
                            is_belief_valid = True
                            belief_init_pos = np.random.random((3,)) * self.size + self.bottom_corner

                        count += 1
                        if count > 100:
                            is_agent_valid = False
                            break

        return (agent_init_pos, agent_init_yaw, target_init_pos, target_init_yaw, belief_init_pos)
    
    def check_not_in_3d_fov(self, agent_pos, target_pos, agent_yaw):
        """
        Check if target is NOT in agent's 3D field of view
        
        Parameters
        ----------
        agent_pos : array_like, shape (3,)
            Agent 3D position [x, y, z]
        target_pos : array_like, shape (3,)
            Target 3D position [x, y, z]
        agent_yaw : float
            Agent yaw angle in radians
            
        Returns
        -------
        is_not_insight : bool
            True if target is NOT visible (outside FOV or range)
        """
        # Calculate 3D relative position
        rel_pos = target_pos - agent_pos
        r = np.linalg.norm(rel_pos)
        
        # Calculate horizontal angle (azimuth)
        r_horizontal = np.linalg.norm(rel_pos[:2])
        if r_horizontal > 1e-6:  # Avoid division by zero
            alpha = np.arctan2(rel_pos[1], rel_pos[0]) - agent_yaw
            alpha = util.wrap_around(alpha)
        else:
            alpha = 0
        
        # Calculate vertical angle (elevation)
        if r > 1e-6:  # Avoid division by zero
            beta = np.arcsin(rel_pos[2] / r)  # elevation angle
        else:
            beta = 0
        
        # Check range condition
        range_check = r > self.config['agent']['sensor_r']
        
        # Check horizontal FOV condition
        h_fov_rad = self.config['agent']['fov'] / 2 / 180 * np.pi
        horizontal_fov_check = abs(alpha) > h_fov_rad
        
        # Check vertical FOV condition
        v_fov_rad = self.config['agent'].get('h_fov', 60) / 2 / 180 * np.pi  # vertical FOV
        vertical_fov_check = abs(beta) > v_fov_rad
        
        # Target is not in sight if any condition fails
        is_not_insight = range_check or horizontal_fov_check or vertical_fov_check
        
        return is_not_insight

    def in_bound(self, pos):
        """
        3D boundary check including depth dimension
        :param pos: [x, y, z] position
        :return: True: in area, False: out area
        """
        return not ((pos[0] < self.bottom_corner[0] + self.margin2wall)
                           or (pos[0] > self.size[0] + self.bottom_corner[0] - self.margin2wall)
                           or (pos[1] < self.bottom_corner[1] + self.margin2wall)
                           or (pos[1] > self.size[1] + self.bottom_corner[1] - self.margin2wall)
                           or (pos[2] < self.bottom_corner[2] + self.margin2wall)
                           or (pos[2] > self.size[2] + self.bottom_corner[2] - self.margin2wall))

    def gen_rand_pose(self, frame_xyz, frame_theta, min_lin_dist, max_lin_dist,
                      min_ang_dist, max_ang_dist, h_fov):
        """
        Generate random 3D position and yaw with relative height checking.
        
        Parameters
        --------
        frame_xyz : array_like, shape (3,) or (2,)
            xyz coordinate of the frame (supports both 2D and 3D)
        frame_theta : float
            theta coordinate of the frame
        min_lin_dist : float
            minimum linear distance from frame to sample point
        max_lin_dist : float
            maximum linear distance from frame to sample point
        min_ang_dist : float
            minimum angular distance (counter clockwise) from frame_theta
        max_ang_dist : float
            maximum angular distance (counter clockwise) from frame_theta
        h_fov : float
            vertical field of view in radians for height constraint
            
        Returns
        -------
        is_valid : bool
            Whether the generated pose is valid
        rand_xyz_global : array_like, shape (3,)
            Global 3D position
        rand_theta : float
            Global orientation
        """
        if max_ang_dist < min_ang_dist:
            max_ang_dist += 2 * np.pi
            
        # Generate horizontal position (same as original 2D logic)
        rand_ang = util.wrap_around(np.random.rand() * (max_ang_dist - min_ang_dist) + min_ang_dist)
        rand_r = np.random.rand() * (max_lin_dist - min_lin_dist) + min_lin_dist
        rand_xy = np.array([rand_r * np.cos(rand_ang), rand_r * np.sin(rand_ang)])
        
        # Handle both 2D and 3D frame positions
        frame_xy = frame_xyz[:2]
        frame_z = frame_xyz[2]
        
        rand_xy_global = util.transform_2d_inv(rand_xy, frame_theta, np.array(frame_xy))
        
        # Generate height within the vertical FOV constraint
        # Calculate the height range based on horizontal distance and vertical FOV
        max_height_offset = rand_r * np.tan(h_fov / 2)  # relative height constraint
        rand_height_offset = np.random.uniform(-max_height_offset, max_height_offset)
        rand_z_global = frame_z + rand_height_offset
        
        # Combine to 3D position
        rand_xyz_global = np.array([rand_xy_global[0], rand_xy_global[1], rand_z_global])
        
        # Check 3D validity: in bound and no collision
        is_valid = (self.in_bound(rand_xyz_global) and 
                   (not hasattr(self, 'obstacles') or 
                    self.obstacles.check_obstacle_collision(rand_xyz_global, self.margin2wall + 1)))
        
        return is_valid, rand_xyz_global, util.wrap_around(rand_ang + frame_theta)

    def observation_noise(self, z):
        # 测量噪声矩阵3D，假设独立
        obs_noise_cov = np.array([
            [self.config['agent']['sensor_r_sd'] ** 2, 0.0, 0.0],
            [0.0, self.config['agent']['sensor_b_sd'] ** 2, 0.0],
            [0.0, 0.0, self.config['agent']['sensor_e_sd'] ** 2]
        ])
        return obs_noise_cov

    def observation(self, target):
        """
        返回是否观测到目标，以及测量值
        """
        # get the target coordinate in spherical coordinate
        r, alpha, gamma = util.relative_distance_spherical(target.state.vec[:3], self.agent.state.vec[:3],
                                                np.radians(self.agent.state.vec[8]))
        # 判断是否观察到目标
        observed = (r <= self.config['agent']['sensor_r']) \
                   & (abs(alpha) <= self.config['agent']['fov'] / 2 / 180 * np.pi) \
                   & (abs(gamma) <= self.config['agent']['h_fov'] / 2 / 180 * np.pi) \
                   & self.obstacles.check_obstacle_block(target.state.vec[:3], self.agent.state.vec[:3],
                                                         self.margin)
        z = None
        if observed:
            z = np.array([r, alpha, gamma])
            z += np.random.multivariate_normal(np.zeros(3, ), self.observation_noise(z))  # 加入噪声
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
                                              np.concatenate([self.agent.est_state.vec[:3],
                                                            [np.radians(self.agent.est_state.vec[8])]]))
                if not (self.has_discovered[i]):
                    self.has_discovered[i] = 1
            self.record_cov_posterior.append(self.belief_targets[i].cov)
        return observed


    @abstractmethod
    def set_limits(self):
        '''
        you should define :
            self.action_dim (for LQR or PID)
            self.action_space
            self.action_range_scale
            self.observation_space  
            self.target_limit (for kf belief)
        here due to the cfg
        :return:
        '''
    
    @abstractmethod
    def update_every_tick(self, sensors):
        '''
        you should define your own update_every_tick when you inherit the child class.
        such as update extra image buffer
        '''

    from typing import Dict
    @abstractmethod
    def get_reward(self, is_col: bool, action: np.ndarray | CmdVel ):
        """
        calulate the reward should return
        :param is_col:
        :param action: may be used for reward calculation
        :return:
        """

    from typing import Union
    @abstractmethod
    def state_func(self, observed, action) -> Union[np.ndarray, dict]:
        """
            should define your own state_func when you inherit the child class due to the observation.
            just an example below
        """
        # Find the closest obstacle coordinate.
        if self.agent.rangefinder.min_distance < self.config['agent']['sensor_r']:
            obstacles_pt = (self.agent.rangefinder.min_distance, np.radians(self.agent.rangefinder.angle))
        else:
            obstacles_pt = (self.config['agent']['sensor_r'], 0)

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
        state.extend(action.tolist())  # dim:3

        state = np.array(state)
        return state
        # Update the visit map for the evaluation purpose.
        # if self.MAP.visit_map is not None:
        #     self.MAP.update_visit_freq_map(self.agent.state, 1.0, observed=bool(np.mean(observed)))
