#!/usr/bin/env python3
# from ig_manager import IGManager
# import rospy
import scipy.io as sio
import numpy as np
from gymnasium import Wrapper

from ttenv.metadata import METADATA

class Ros(Wrapper):
    def __init__(self, env, skip=3):
        super(Ros, self).__init__(env)
        self.node = IGManager()

        self.num_robots = self.node.num_robots
        self.num_targets = self.node.num_targets

        rate = rospy.Rate(5)
        while(node.robot_states[0] is None):  # Sleep Until ODOM is received to ensure things are set up.
            rate.sleep()

    def render(self, traj_num=None):
        if (traj_num is not None) and (traj_num%self.skip==0):
            assert(self.num_robots == 1) # For now
            env = self.env.env
            for i in range(self.num_robots):
                node.publish_robot_wp(env.agent.state, i, z=0.0)

            for i in range(0, num_targets):
                # Get polar coordinate with velocity
                t_theta = np.arctan2(env.targets[i].state[-1], env.targets[i].state[-2])
                b_theta = np.arctan2(env.belief_targets[i].state[-1], env.belief_targets[i].state[-2])

                node.publish_target_wp(np.concatenate((env.targets[i].state[:2], [t_theta])), i, z=0.0)
                node.publish_target_belief(np.concatenate((env.targets[i].state[:2], [b_theta])), env.target_b_cov, i, z=0.0)


class RosLog(object):
    def __init__(self, num_targets, wrapped_num=1, metadata=METADATA):
        self.wrapped_num = wrapped_num
        self.num_targets = num_targets
        self.robots = []
        self.targets = []
        self.belief_targets = []
        self.belief_covs = []
        self.records = metadata

    def log(self, env_i):

        n = 0
        env = env_i
        while(n < self.wrapped_num):
            env = env.env
            n += 1
        self.robots.append([env.agent.state])
        t_state = [np.concatenate((env.targets[i].state[:2],
                                    [np.arctan2(env.targets[i].state[-1], env.targets[i].state[-2])]))
                                    for i in range(self.num_targets)]
        b_state = [np.concatenate((env.belief_targets[i].state[:2],
                                    [np.arctan2(env.belief_targets[i].state[-1], env.belief_targets[i].state[-2])]))
                                    for i in range(self.num_targets)]
        self.targets.append(t_state)
        self.belief_targets.append(b_state)
        self.belief_covs.append([env.belief_targets[i].cov for i in range(env.num_targets)])

    def save(self, path=''):
        self.records['num_robots'] = 1
        self.records['num_targets'] = self.num_targets
        self.records['robots'] = self.robots
        self.records['targets'] = self.targets
        self.records['belief_targets'] = self.belief_targets
        self.records['belief_covs'] = self.belief_covs
        import os, pickle
        pickle.dump(self.records, open(os.path.join(path,'ros_log.pkl'),'wb'), protocol=2)
