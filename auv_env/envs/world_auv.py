import holoocean
import numpy as np
from numpy import linalg as LA
from auv_control.estimation import KFbelief

from auv_env import util
from auv_env.envs.agent import AgentAuv, AgentAuvTarget
from auv_env.envs.obstacle import Obstacle
from metadata import METADATA

from gymnasium import spaces


class World_AUV:
    """
        different from world:target is also an auv
    """

    def __init__(self, map, show, verbose, num_targets, **kwargs):
        super().__init__(map, show, verbose, num_targets, **kwargs)

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
                                       bottom_corner=self.bottom_corner, start_time=time, scene=self.ocean, l_p=METADATA['lqr_l_p'])
                        for i in range(self.num_targets)]
        # Build target beliefs.
        if self.random:
            self.const_q = np.random.choice([0.01, 0.02, 0.05, 0.1, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
        else:
            self.const_q = METADATA['const_q']
        self.targetA = np.concatenate((np.concatenate((np.eye(2),
                                                       self.control_period * np.eye(2)), axis=1),
                                       [[0, 0, 1, 0], [0, 0, 0, 1]]))
        self.target_noise_cov = self.const_q * np.concatenate((
            np.concatenate((self.control_period ** 3 / 3 * np.eye(2),
                            self.control_period ** 2 / 2 * np.eye(2)), axis=1),
            np.concatenate((self.control_period ** 2 / 2 * np.eye(2),
                            self.control_period * np.eye(2)), axis=1)))
        self.belief_targets = [KFbelief(dim=self.target_dim,
                                        limit=self.limit['target'], A=self.targetA,
                                        W=self.target_noise_cov,
                                        obs_noise_func=self.observation_noise)
                               for _ in range(self.num_targets)]

    def step(self, action_waypoint):
        global_waypoint = np.zeros(3)
        observed = []
        # 归一化展开
        r = action_waypoint[0] * self.action_range_scale[0]
        theta = action_waypoint[1] * self.action_range_scale[1] - self.action_range_scale[1] / 2
        global_waypoint[:2] = util.polar_distance_global(np.array([r, theta]), self.agent.est_state.vec[:2],
                                                         np.radians(self.agent.est_state.vec[8]))
        angle = action_waypoint[2] * self.action_range_scale[2] - self.action_range_scale[2] / 2
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
            self.sensors = self.ocean.tick()

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
        self.state_func(observed)
        self.record_observed = observed
        if METADATA['render']:
            print(is_col, observed[0], reward)
        self.is_col = is_col
        return self.state, reward, done, 0, {'mean_nlogdetcov': mean_nlogdetcov, 'std_nlogdetcov': std_nlogdetcov}

    def reset(self):
        self.ocean.reset()
        self.ocean.draw_box(self.center.tolist(), (self.size / 2).tolist(), color=[0, 0, 255], thickness=30,
                            lifetime=0)  # draw the area
        self.obstacles.reset()
        self.obstacles.draw_obstacle()

        if self.task_random:
            self.insight = np.random.choice([True, False])
        else:
            self.insight = METADATA['insight']
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
        self.target_init_cov = METADATA['target_init_cov']
        self.agent_init_pos, self.agent_init_yaw, self.target_init_pos, self.target_init_yaw, self.belief_init_pos\
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
            target = 'target'+str(i)
            # self.ocean.agents['target'].set_physics_state(location=self.target_init_pos,
            #                                               rotation=[0.0, 0.0, -np.rad2deg(self.target_init_yaw)],
            #                                               velocity=[0.0, 0.0, 0.0],
            #                                               angular_velocity=[0.0, 0.0, 0.0])
            self.ocean.agents[target].teleport(location=self.target_init_pos,
                                                 rotation=[0.0, 0.0, np.rad2deg(self.target_init_yaw)])
            self.target_u = np.zeros(8)
            self.ocean.act(target, self.target_u)

        self.sensors = self.ocean.tick()

        self.set_limits()
        self.build_models(sampling_period=self.sampling_period,
                          agent_init_state=self.sensors['auv0'],
                          target_init_state=self.sensors['target0'],    #TODO
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
                init_cov=self.target_init_cov)

        # The targets are observed by the agent (z_0) and the beliefs are updated (b_0).
        observed = self.observe_and_update_belief()

        # Predict the target for the next step, b_1|0.
        for i in range(self.num_targets):
            self.belief_targets[i].predict()

        observed = [True]
        # Compute the RL state.
        self.state_func(observed)
        return self.state, 0

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
            is_agent_valid = self.in_bound(agent_init_pos) and self.obstacles.check_obstacle_collision(agent_init_pos,
                                                                                                       self.margin2wall + 2)
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
                                is_no_blocked = self.obstacles.check_obstacle_block(agent_init_pos, target_init_pos,
                                                                                    self.margin2wall + 2)
                                is_target_valid = (self.noblock == is_no_blocked)
                        elif not self.insight:
                            target_init_pos = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
                            target_init_yaw = np.random.uniform(-np.pi / 2, np.pi / 2)
                            # confirm is not insight
                            r, alpha = util.relative_distance_polar(target_init_pos, agent_init_pos, agent_init_yaw)
                            if (r > METADATA['sensor_r'] or np.rad2deg(alpha) > METADATA['fov']/2
                                    or np.rad2deg(alpha) < -METADATA['fov']/2):
                                is_not_insight = True
                            else:
                                is_not_insight = False
                            is_target_valid = (
                                    self.in_bound(target_init_pos) and
                                    self.obstacles.check_obstacle_collision(target_init_pos, self.margin2wall + 2) and
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
        is_valid = (self.in_bound(rand_xy_global) and self.obstacles.check_obstacle_collision(rand_xy_global,
                                                                                              self.margin2wall + 2))
        return is_valid, rand_xy_global, util.wrap_around(rand_ang + frame_theta)

    def set_limits(self):
        # LIMIT
        self.limit = {}  # 0: low, 1:highs
        self.limit['agent'] = [np.concatenate((self.bottom_corner[:2], [-np.pi])),
                               np.concatenate((self.top_corner[:2], [np.pi]))]
        self.limit['target'] = [np.concatenate((self.bottom_corner[:2], np.array([-3, -3]))),
                                np.concatenate((self.top_corner[:2], np.array([3, 3])))]
        # STATE:
        # target distance、angle、协方差行列式值、bool;agent 自身定位; 声呐图像
        # target distance、angle、协方差行列式值、bool;agent 自身定位; 最近障碍物的位置
        self.limit['state'] = [np.concatenate((np.concatenate(([0.0, -np.pi, -50.0, 0.0] * self.num_targets,
                                                               [self.bottom_corner[0], self.bottom_corner[1], -np.pi])),
                                               [0.0, -np.pi])),
                               np.concatenate((np.concatenate(([600.0, np.pi, 50.0, 2.0] * self.num_targets,
                                                               [self.top_corner[0], self.top_corner[1], np.pi])),
                                               [self.sensor_r, np.pi]))]
        # target distance、angle、协方差行列式值、bool;agent 自身定位;
        # self.limit['state'] = [np.concatenate(([0.0, -np.pi, -50.0, 0.0] * self.num_targets, [0.0, -np.pi])),
        #                        np.concatenate(([600.0, np.pi, 50.0, 2.0] * self.num_targets, [self.sensor_r, np.pi]))]
        self.observation_space = spaces.Box(self.limit['state'][0], self.limit['state'][1], dtype=np.float64)

    def observation_noise(self, z):
        # 测量噪声矩阵，假设独立
        obs_noise_cov = np.array([[self.sensor_r_sd * self.sensor_r_sd, 0.0],
                                  [0.0, self.sensor_b_sd * self.sensor_b_sd]])
        return obs_noise_cov

    def observation(self, target):
        """
        返回是否观测到目标，以及测量值
        """
        r, alpha = util.relative_distance_polar(target.state.vec[:2],
                                                xy_base=self.agent.state.vec[:2],
                                                theta_base=np.radians(self.agent.state.vec[8]))
        # 判断是否观察到目标
        observed = (r <= self.sensor_r) \
                   & (abs(alpha) <= self.fov / 2 / 180 * np.pi) \
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

    def state_func(self, observed):
        '''
        在父类的step中调用该函数对self.state进行更新
        RL state: [d, alpha, log det(Sigma), observed] * nb_targets, [o_d, o_alpha]
        '''
        # Find the closest obstacle coordinate.
        if self.agent.rangefinder.min_distance < self.sensor_r:
            obstacles_pt = (self.agent.rangefinder.min_distance, np.radians(self.agent.rangefinder.angle))
        else:
            obstacles_pt = (self.sensor_r, 0)

        self.state = []
        for i in range(self.num_targets):
            r_b, alpha_b = util.relative_distance_polar(
                self.belief_targets[i].state[:2],
                xy_base=self.agent.est_state.vec[:2],
                theta_base=np.radians(self.agent.est_state.vec[8]))
            self.state.extend([r_b, alpha_b,
                               np.log(LA.det(self.belief_targets[i].cov)),
                               float(observed[i])])
        self.state.extend([self.agent.state.vec[0], self.agent.state.vec[1],
                           np.radians(self.agent.state.vec[8])])
        self.state.extend(obstacles_pt)
        self.state = np.array(self.state)

        # Update the visit map for the evaluation purpose.
        # if self.MAP.visit_map is not None:
        #     self.MAP.update_visit_freq_map(self.agent.state, 1.0, observed=bool(np.mean(observed)))


if __name__ == '__main__':
    from auv_control import scenario

    print("Test World")
    world = World_AUV(scenario, map='TestMap', show=True, verbose=True, num_targets=1)
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
