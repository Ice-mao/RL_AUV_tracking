import holoocean
import numpy as np
from auv_control.estimation import InEKF
from auv_control.control import LQR
from auv_control.planning import Traj, RRT
from auv_control import State
from tools import Plotter
from auv_env import util
from auv_env.agent import AgentAuv, AgentSphere
from auv_env.obstacle import Obstacle
from auv_env.metadata import METADATA


class World:
    """
    build a 3d scenario in HoloOcean to use the more real engine
    """

    def __init__(self, scenario, show, verbose, num_targets, **kwargs):
        # define the entity
        self.ocean = holoocean.make(scenario_cfg=scenario, show_viewport=show, verbose=verbose)
        self.agent = None
        # init the param
        self.sampling_period = 1 / scenario["ticks_per_sec"]  # sample time
        self.num_targets = num_targets  # num of target

        # Setup environment
        self.size = np.array([45, 45, 25])
        self.bottom_corner = np.array([-22.5, -22.5, -25])
        self.fix_depth = -5
        self.margin = METADATA['margin']
        self.margin2wall = METADATA['margin2wall']
        self.ocean.draw_box(self.center.tolist(), (self.size / 2).tolist(), color=[0, 0, 255], thickness=30,
                            lifetime=0)  # draw the area

        # Setup obstacles
        # rule is obstacles combined will rotate from their own center
        self.obstacles = Obstacle(self.ocean, self.fix_depth)
        self.obstacles.draw_obstacle()

        # Cal random  pos of agent and target
        self.agent_init_pos = None
        self.agent_init_yaw = None
        self.target_init_pos = None
        self.target_init_yaw = None
        self.agent_init_pos, self.agent_init_yaw, self.target_init_pos, self.target_init_yaw = self.get_init_pose_random()
        print(self.agent_init_pos, self.agent_init_yaw)
        print(self.target_init_pos, self.target_init_yaw)
        # Set the pos and tick the scenario
        self.ocean.agents['auv0'].set_physics_state(location=self.agent_init_pos,
                                                    rotation=[0.0, 0.0, np.rad2deg(self.agent_init_yaw)],
                                                    velocity=[0.0, 0.0, 0.0],
                                                    angular_velocity=[0.0, 0.0, 0.0])
        # self.ocean.agents['auv0'].teleport(location=self.agent_init_pos,
        #                                    rotation=[0.0, 0.0, np.rad2deg(self.agent_init_yaw)])
        self.u = np.zeros(8)
        self.ocean.act("auv0", self.u)
        self.ocean.agents['target'].set_physics_state(location=self.target_init_pos,
                                                      rotation=[0.0, 0.0, np.rad2deg(self.target_init_yaw)],
                                                      velocity=[0.0, 0.0, 0.0],
                                                      angular_velocity=[0.0, 0.0, 0.0])
        # self.ocean.agents['target'].teleport(location=self.target_init_pos,
        #                                    rotation=[0.0, 0.0, np.rad2deg(self.target_init_yaw)])
        self.target_u = [0, 0]
        self.ocean.act("target", self.target_u)
        self.sensors = self.ocean.tick()

        self.build_models(sampling_period=self.sampling_period,
                          agent_init_state=self.sensors['auv0']
                          , target_init_state=self.sensors['target'], time=self.sensors['t'])

    def build_models(self, sampling_period, agent_init_state, target_init_state, time, **kwargs):
        """
        :param sampling_period:
        :param agent_init_state:list [[x,y,z],yaw(theta)]
        :param target_init_state:list [[x,y,z],yaw(theta)]
        :param kwargs:
        :return:
        """
        # Build a robot
        self.agent = AgentAuv(dim=3, sampling_period=sampling_period, sensor=agent_init_state)
        self.targets = [AgentSphere(dim=3, sampling_period=sampling_period, sensor=target_init_state
                                    , obstacles=self.obstacles, fixed_depth=self.fix_depth, size=self.size,
                                    bottom_corner=self.bottom_corner, start_time=time)
                        for _ in range(self.num_targets)]

    def step(self, action_vw):
        for target in self.targets:
            self.target_u = target.update(self.sensors['target'], self.sensors['t'])
            self.ocean.act("target", self.target_u)

        # self.u = self.agent.update(action_vw, self.sensors['auv0'])
        # self.ocean.act("auv0", self.u)
        self.sensors = self.ocean.tick()

    def reset(self):
        self.ocean.reset()
        self.ocean.draw_box(self.center.tolist(), (self.size / 2).tolist(), color=[0, 0, 255], thickness=30,
                            lifetime=0)  # draw the area
        self.obstacles.reset()
        self.obstacles.draw_obstacle()
        # reset the random position
        self.agent_init_pos = None
        self.agent_init_yaw = None
        self.target_init_pos = None
        self.target_init_yaw = None
        self.agent_init_pos, self.agent_init_yaw, self.target_init_pos, self.target_init_yaw = self.get_init_pose_random()
        # Set the pos and tick the scenario
        self.ocean.agents['auv0'].set_physics_state(location=[self.agent_init_pos])
        self.u = np.zeros(8)
        self.ocean.act("auv0", self.u)
        self.ocean.agents['target'].set_physics_state(location=[self.target_init_pos])
        self.target_u = [0, 0]
        self.ocean.act("target", self.target_u)
        self.sensors = self.ocean.tick()
        # reset model
        self.agent.reset(self.sensors['auv0'])
        for target in self.targets:
            target.reset(self.sensors['target'])

    @property
    def center(self):
        return self.bottom_corner + self.size / 2

    def get_init_pose_random(self,
                             lin_dist_range_a2t=METADATA['lin_dist_range_a2t'],
                             ang_dist_range_a2t=METADATA['ang_dist_range_a2t'],
                             blocked=None, ):
        is_agent_valid = False
        blocked = False
        if 'blocked' in METADATA:
            blocked = METADATA['blocked']
        while not is_agent_valid:
            init_pose = {}
            # generatr an init pos around the map
            a_init = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
            # satisfy the in bound and no collision conditions ----> True(is valid)
            is_agent_valid = self.in_bound(a_init) and self.obstacles.check_obstacle_collision(a_init, self.margin2wall)
            agent_init_pos = np.array([a_init[0], a_init[1]])
            agent_init_yaw = np.random.uniform(-np.pi/2, np.pi/2)
            for i in range(self.num_targets):
                count, is_target_valid, target_init_pos, target_init_yaw = 0, False, np.zeros((2,)), np.zeros((1,))
                while not is_target_valid:
                    is_target_valid, target_init_pos, target_init_yaw = self.gen_rand_pose(
                        agent_init_pos,
                        agent_init_yaw,
                        lin_dist_range_a2t[0], lin_dist_range_a2t[1],
                        ang_dist_range_a2t[0], ang_dist_range_a2t[1]
                    )

                    if is_target_valid:  # check the blocked condition
                        is_no_blocked = self.obstacles.check_obstacle_block(agent_init_pos, target_init_pos,
                                                                            self.margin)
                        flag = not is_no_blocked
                        is_target_valid = (blocked == flag)
                    count += 1
                    if count > 50:
                        is_agent_valid = False
                        is_target_valid = False
                        count = 0
        return (np.append(agent_init_pos, self.fix_depth), agent_init_yaw,
                np.append(target_init_pos, self.fix_depth), target_init_yaw)

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
        rand_xy = np.array([rand_r * np.cos(rand_ang), rand_r * np.sin(rand_ang)])  # set in HoloOcean,opposite
        rand_xy_global = util.transform_2d_inv(rand_xy, -frame_theta, np.array(frame_xy))  # the same opposite
        is_valid = (self.in_bound(rand_xy_global) and self.obstacles.check_obstacle_collision(rand_xy_global,
                                                                                              self.margin2wall))
        return is_valid, rand_xy_global, util.wrap_around(rand_ang + frame_theta)


if __name__ == '__main__':
    from auv_control import scenario

    print("Test World")
    world = World(scenario, show=True, verbose=True, num_targets=1)
    print(world.size)
    # world.targets[0].planner.draw_traj(world.ocean, 30)
    for _ in range(20000):
        if 'q' in world.agent.keyboard.pressed_keys:
            break
        command = world.agent.keyboard.parse_keys()
        for target in world.targets:
            world.target_u = target.update(world.sensors['target'], world.sensors['t'])
            # world.target_u = [0.1, 1]
            # world.ocean.act("target", world.target_u * 0.01)
            world.ocean.act("target", tuple(x * 0.01 for x in world.target_u))
        # self.u = self.agent.update(action_vw, self.sensors['auv0'])
        world.ocean.act("auv0", command)
        world.sensors = world.ocean.tick()
