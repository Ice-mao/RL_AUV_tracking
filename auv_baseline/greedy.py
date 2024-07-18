import holoocean
import numpy as np
from numpy import linalg as LA
from auv_env import util
from metadata import METADATA
import copy


class Greedy:
    """
        Myopic Greedy Viewpoint Selection:
        a one-step next-best-view planner
    """

    def __init__(self, env):
        self.action_space = env.action_space
        self.world = env.world
        self.N_nbv = 50  # select the num of sample points.

    def predict(self, obs):
        """
            Predict the next best viewpoint.
        """
        # belief = self.world.belief_targets
        # self.sample_belief = None
        best_reward = -100
        best_viewpoint = None
        for i in range(self.N_nbv):
            # 初始化每个采样对应的belief
            # self.sample_belief = copy.deepcopy(self.world.belief_targets)
            # 随机采样一个动作/路径点
            random_action_waypoint = self.action_space.sample()
            global_waypoint = np.zeros(3)
            # 归一化展开，并得到全局路径点坐标
            r = random_action_waypoint[0] * self.world.action_range_scale[0]
            theta = random_action_waypoint[1] * self.world.action_range_scale[1] - self.world.action_range_scale[1] / 2
            global_waypoint[:2] = util.polar_distance_global(np.array([r, theta]), self.world.agent.state.vec[:2],
                                                             np.radians(self.world.agent.state.vec[8]))
            angle = random_action_waypoint[2] * self.world.action_range_scale[2] - self.world.action_range_scale[2] / 2
            global_waypoint[2] = self.world.agent.state.vec[8] + np.rad2deg(angle)

            # The targets are observed by the agent (z_t+1) and the beliefs are updated.
            self.cov = self.observe_and_update_belief(global_waypoint)
            is_col = not (self.world.obstacles.check_obstacle_collision(global_waypoint[:2], self.world.margin2wall)
                          and self.world.in_bound(global_waypoint[:2])
                          and np.linalg.norm(
                        global_waypoint[:2] - self.world.targets[0].state.vec[:2]) > self.world.margin+2)

            # Compute a reward from b_t+1|t+1 or b_t+1|t.
            reward, done, mean_nlogdetcov, std_nlogdetcov = self.get_reward(is_col=is_col)

            if reward > best_reward:
                best_reward = reward
                best_viewpoint = random_action_waypoint
        return best_viewpoint

    def observe_and_update_belief(self, global_waypoint):
        observed = []
        cov = self.world.belief_targets[0].cov
        for i in range(self.world.num_targets):
            observation = self.observation(self.world.targets[i], global_waypoint)
            observed.append(observation[0])
            if observation[0]:  # if observed, update the target belief.
                cov, state = self.world.belief_targets[i].greedy_update(observation[1],
                                                                        np.array(
                                                                            [global_waypoint[0],
                                                                             global_waypoint[1],
                                                                             np.radians(global_waypoint[2])]))
                if not (self.world.has_discovered[i]):
                    self.world.has_discovered[i] = 1
        return cov

    def observation(self, target, global_waypoint):
        """
        返回是否观测到目标，以及测量值
        """
        r, alpha = util.relative_distance_polar(target.state.vec[:2],
                                                xy_base=global_waypoint[:2],
                                                theta_base=np.radians(global_waypoint[2]))
        # 判断是否观察到目标
        observed = (r <= self.world.sensor_r) \
                   & (abs(alpha) <= self.world.fov / 2 / 180 * np.pi) \
                   & self.world.obstacles.check_obstacle_block(target.state.vec[:2], global_waypoint[:2],
                                                               self.world.margin)
        z = None
        if observed:
            z = np.array([r, alpha])
            z += np.random.multivariate_normal(np.zeros(2, ), self.world.observation_noise(z))  # 加入噪声
        return observed, z

    def get_reward(self, is_col, is_training=True, c_mean=METADATA['c_mean'], c_std=METADATA['c_std'],
                   c_penalty=METADATA['c_penalty'], k_3=METADATA['k_3'], k_4=METADATA['k_4'], k_5=METADATA['k_5']):
        detcov = [LA.det(self.cov)]
        r_detcov_mean = - np.mean(np.log(detcov))
        r_detcov_std = - np.std(np.log(detcov))

        reward = c_mean * r_detcov_mean + c_std * r_detcov_std
        # reward_w = np.exp(-k_3 * np.abs(np.radians(self.agent.state.vec[8]))) - 1
        reward = reward
        if is_col:
            reward = np.min([0.0, reward]) - c_penalty * 1.0
        return reward, False, r_detcov_mean, r_detcov_std
