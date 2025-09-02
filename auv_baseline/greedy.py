import holoocean
import numpy as np
from numpy import linalg as LA
from auv_env import util
import copy


class Greedy:
    """
    Myopic Greedy Viewpoint Selection:
    a one-step next-best-view planner
    支持2D和3D环境
    """

    def __init__(self, world, N=100):
        self.world = world
        self.action_space = world.action_space
        self.N_nbv = N  # select the num of sample points.
        
        # 判断是否为3D环境
        self.is_3d = world.config['target']['target_dim'] == 6
        self.controller = world.config['agent']['controller']
        
        if self.is_3d:
            print(f"初始化3D Greedy算法，控制器: {self.controller}")
        else:
            print(f"初始化2D Greedy算法，控制器: {self.controller}")

    def predict(self, obs):
        """
        Predict the next best viewpoint.
        支持2D和3D环境
        """
        best_reward = -100
        best_viewpoint = None
        
        for i in range(self.N_nbv):
            # 随机采样一个动作/路径点
            random_action_waypoint = self.action_space.sample()
            
            if self.is_3d and self.controller == 'LQR':
                # 3D LQR: action = [x, y, z, yaw]
                global_waypoint = self._compute_3d_waypoint(random_action_waypoint)
            else:
                # 2D: 保持原有逻辑
                global_waypoint = self._compute_2d_waypoint(random_action_waypoint)

            # 观测和更新belief
            self.cov = self.observe_and_update_belief(global_waypoint)
            
            # 碰撞检测
            is_col = self._check_collision(global_waypoint)

            # 计算奖励
            reward, done, mean_nlogdetcov, std_nlogdetcov = self.get_reward(is_col=is_col)

            if reward > best_reward:
                best_reward = reward
                best_viewpoint = random_action_waypoint
                
        return best_viewpoint
    
    def _compute_3d_waypoint(self, action):
        """
        计算3D全局路径点，与base_3d.py中的step函数逻辑一致
        action: [距离, 角度, 深度, 偏航角] 已经是归一化的动作
        """
        # 获取动作范围和缩放，与base_3d.py中的step函数保持一致
        action_scale = self.world.action_range_scale  # LQR的缩放参数
        
        # 反归一化动作 - 与base_3d.py中的逻辑完全一致
        r = action[0] * action_scale[0]          # 距离: [0,1] -> [0, 0.5]
        theta = action[1] * action_scale[1]      # 角度: [-1,1] -> [-pi/2, pi/2] (注意这里不减/2)
        depth = action[2] * action_scale[2]      # 深度: [-1,1] -> [-0.3, 0.3] (注意这里不减/2)  
        angle = action[3] * action_scale[3]      # 偏航角: [-1,1] -> [-pi/20, pi/20] (注意这里不减/2)
        
        # 使用与base_3d.py相同的计算逻辑
        # 当前agent状态
        agent_pos = self.world.agent.state.vec[:2]
        agent_depth = self.world.agent.state.vec[2]
        agent_yaw = np.radians(self.world.agent.state.vec[8])
        
        # 计算目标位置 - 与base_3d.py相同的方法
        target_pos = util.polar_distance_global(np.array([r, theta]), agent_pos, agent_yaw)
        target_depth = agent_depth + depth
        target_yaw = self.world.agent.state.vec[8] + np.rad2deg(angle)
        
        # 返回全局坐标 [x, y, z, yaw_degrees]
        global_waypoint = np.array([target_pos[0], target_pos[1], target_depth, target_yaw])
        
        return global_waypoint
    
    def _compute_2d_waypoint(self, action):
        """
        计算2D全局路径点，与base.py中的step函数逻辑一致
        """
        # 获取动作范围和缩放，与base.py保持一致
        action_scale = self.world.action_range_scale
        
        # 反归一化动作 - 与base.py中的逻辑一致
        r = action[0] * action_scale[0]      # 距离: [0,1] -> [0, 0.5]
        theta = action[1] * action_scale[1]  # 角度: [-1,1] -> [-pi/2, pi/2] (不减/2)
        angle = action[2] * action_scale[2]  # 偏航角: [-1,1] -> [-pi/16, pi/16] (不减/2)
        
        # 使用与base.py相同的计算逻辑
        target_pos = util.polar_distance_global(np.array([r, theta]), 
                                               self.world.agent.state.vec[:2],
                                               np.radians(self.world.agent.state.vec[8]))
        target_yaw = self.world.agent.state.vec[8] + np.rad2deg(angle)
        
        # 返回2D全局坐标 [x, y, yaw_degrees]
        global_waypoint = np.array([target_pos[0], target_pos[1], target_yaw])
        
        return global_waypoint
    
    def _check_collision(self, global_waypoint):
        """
        检查碰撞 (支持2D和3D)
        """
        if self.is_3d:
            # 3D碰撞检测
            return not (self.world.obstacles.check_obstacle_collision(global_waypoint[:3], self.world.margin2wall)
                       and self.world.in_bound(global_waypoint[:3])
                       and np.linalg.norm(global_waypoint[:3] - self.world.targets[0].state.vec[:3]) > self.world.margin + 2)
        else:
            # 2D碰撞检测 (保持原有逻辑)
            return not (self.world.obstacles.check_obstacle_collision(global_waypoint[:2], self.world.margin2wall)
                       and self.world.in_bound(global_waypoint[:2])
                       and np.linalg.norm(global_waypoint[:2] - self.world.targets[0].state.vec[:2]) > self.world.margin + 2)

    def get_action(self, obs):
        """
        Alias for predict to be consistent with SB3 model API.
        """
        return self.predict(obs)

    def observe_and_update_belief(self, global_waypoint):
        """
        观测和更新belief (支持2D和3D)
        """
        observed = []
        cov = self.world.belief_targets[0].cov
        
        for i in range(self.world.num_targets):
            observation = self.observation(self.world.targets[i], global_waypoint)
            observed.append(observation[0])
            
            if observation[0]:  # if observed, update the target belief.
                if self.is_3d:
                    # 3D情况：使用球坐标和3D agent状态
                    agent_state_3d = np.array([
                        global_waypoint[0],   # x
                        global_waypoint[1],   # y  
                        global_waypoint[2],   # z
                        np.radians(global_waypoint[3])  # yaw in radians
                    ])
                    cov, state = self.world.belief_targets[i].greedy_update(
                        observation[1], agent_state_3d)
                else:
                    # 2D情况：保持原有逻辑
                    agent_state_2d = np.array([
                        global_waypoint[0],
                        global_waypoint[1], 
                        np.radians(global_waypoint[2])
                    ])
                    cov, state = self.world.belief_targets[i].greedy_update(
                        observation[1], agent_state_2d)
                        
                if not (self.world.has_discovered[i]):
                    self.world.has_discovered[i] = 1
                    
        return cov

    def observation(self, target, global_waypoint):
        """
        返回是否观测到目标，以及测量值 (支持2D和3D)
        """
        if self.is_3d:
            # 3D球坐标观测
            r, theta, gamma = util.relative_distance_spherical(
                target.state.vec[:3],
                xyz_base=global_waypoint[:3],
                theta_base=np.radians(global_waypoint[3])
            )
            
            # 3D观测判断：距离、水平视场角、垂直视场角
            observed = (r <= self.world.config['agent']['sensor_r']) \
                      & (abs(theta) <= self.world.config['agent']['fov'] / 2 / 180 * np.pi) \
                      & (abs(gamma) <= self.world.config['agent']['h_fov'] / 2 / 180 * np.pi) \
                      & self.world.obstacles.check_obstacle_block(target.state.vec[:3], 
                                                                  global_waypoint[:3],
                                                                  self.world.margin)
            z = None
            if observed:
                z = np.array([r, theta, gamma])
                # 3D观测噪声
                noise_cov = np.diag([self.world.config['agent']['sensor_r_sd']**2, 
                                   self.world.config['agent']['sensor_b_sd']**2,
                                   self.world.config['agent']['sensor_e_sd']**2])
                z += np.random.multivariate_normal(np.zeros(3), noise_cov)
                
        else:
            # 2D极坐标观测 (保持原有逻辑)
            r, alpha = util.relative_distance_polar(target.state.vec[:2],
                                                   xy_base=global_waypoint[:2],
                                                   theta_base=np.radians(global_waypoint[2]))
            
            # 判断是否观察到目标
            observed = (r <= self.world.sensor_r) \
                      & (abs(alpha) <= self.world.fov / 2 / 180 * np.pi) \
                      & self.world.obstacles.check_obstacle_block(target.state.vec[:2], 
                                                                  global_waypoint[:2],
                                                                  self.world.margin)
            z = None
            if observed:
                z = np.array([r, alpha])
                z += np.random.multivariate_normal(np.zeros(2), self.world.observation_noise(z))
                
        return observed, z

    def get_reward(self, is_col):
        reward_param = self.world.config['reward_param']
        c_mean = reward_param['c_mean']
        c_std = reward_param['c_std']
        c_penalty = reward_param['c_penalty']

        detcov = [LA.det(self.cov)]
        r_detcov_mean = - np.mean(np.log(detcov))
        r_detcov_std = - np.std(np.log(detcov))

        reward = c_mean * r_detcov_mean + c_std * r_detcov_std
        # reward_w = np.exp(-k_3 * np.abs(np.radians(self.agent.state.vec[8]))) - 1
        reward = reward
        if is_col:
            reward = np.min([0.0, reward]) - c_penalty * 1.0
        return reward, False, r_detcov_mean, r_detcov_std
