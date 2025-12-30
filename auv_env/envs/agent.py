import holoocean
from auv_env.envs.tools import KeyBoardCmd, ImagingSonar, RangeFinder
import auv_env.util as util

from auv_control.estimation import InEKF
from auv_control.control import LQR, PID
from auv_control.planning import RRT_2d, RRT_3d
from auv_control.state import rot_to_rpy
from auv_control import State

from gridmap.grid_map import *


class Agent(object):
    def __init__(self, dim, sampling_period, config):
        self.dim = dim
        self.sampling_period = sampling_period
        self.config = config
        self.margin = self.config['env']['margin']
        self.margin2wall = self.config['env']['margin2wall']

    def range_check(self):
        self.state = np.clip(self.state, self.limit[0], self.limit[1])

    def collision_check(self, pos):
        return self.collision_func(pos[:2])

    def margin_check(self, pos, target_pos):
        return any(np.sqrt(np.sum((pos - target_pos) ** 2, axis=1)) < self.margin)  # no update

    def reset(self, init_state):
        self.state = State(init_state)


class AgentAuv(Agent):
    def __init__(self, dim, sampling_period, sensor, scenario, config):
        Agent.__init__(self, dim, sampling_period, config)
        scenario_cfg = holoocean.get_scenario(scenario)
        robo_type = scenario_cfg['agents'][0]['agent_type']
        # init the control part of Auv
        if self.config['agent']['controller'] == "LQR":
            lqr_config = self.config['agent']['controller_config']['LQR']
            if lqr_config['random_lp']:
                l_p = np.random.choice(lqr_config['l_p'][1])
            else:
                l_p = lqr_config['l_p'][0]
            self.controller = LQR(l_p=l_p, l_v=lqr_config['l_v'],
                                  l_r=lqr_config['l_r'],
                                  r_f=lqr_config['r_f'],
                                  r_t=lqr_config['r_t'],
                                  robo_type=robo_type)
        elif self.config['agent']['controller'] == "PID":
            self.controller = PID(robo_type=robo_type)
        elif self.config['agent']['controller'] == "KEYBOARD":
            self.controller = KeyBoardCmd(force=5)
        else:
            raise ValueError("Unknown controller choice")

        # init the sonar and grid map part
        if self.config['agent']['use_sonar']:
            self.imagingsonar = ImagingSonar(scenario=scenario)  # to see if it is useful
            self.zmax = self.imagingsonar.zmax
        else:
            self.zmax = 8.0

        if self.config['agent']['grid']['use_grid']:
            # count for grid map
            self.count = 0
            self.period = 10
            # probability
            self.P_prior = self.config['agent']['grid']['p_prior']  # Prior occupancy probability
            self.P_occ = self.config['agent']['grid']['p_occ']  # Probability that cell is occupied with total confidence
            self.P_free = self.config['agent']['grid']['p_free']  # Probability that cell is free with total confidence
            self.RESOLUTION = self.config['agent']['grid']['resolution']  # Grid resolution in [m]
            self.grid_map = GridMap(X_lim=np.array([self.config['scenario']['bottom_corner'][0],
                                                   self.config['scenario']['bottom_corner'][0] + self.config['scenario']['size'][0]]),
                                   Y_lim=np.array([self.config['scenario']['bottom_corner'][1],
                                                   self.config['scenario']['bottom_corner'][1] + self.config['scenario']['size'][1]]),
                                   resolution=self.RESOLUTION,
                                   p=self.P_prior)

        # init the sensor part of AUV
        self.rangefinder = RangeFinder(scenario=scenario, config=self.config)
        self.state = State(sensor)
        self.last_state = State(sensor)
        self.est_state = State(sensor)
        self.observer = InEKF(x_init=self.state.vec[0], y_init=self.state.vec[1])


    def reset(self, sensor):
        self.state = State(sensor)
        self.last_state = State(sensor)
        self.est_state = State(sensor)
        self.vw = [0.0, 0.0]

    def update(self, action_waypoint, depth, sensors):
        self.rangefinder.update(sensors)
        # Estimate State
        self.last_state = self.state
        self.state = State(sensors)
        # if you want to eval,use the observer to update
        self.est_state = State(sensors)
        # self.est_state = self.observer.tick(sensors, self.sampling_period)
        # print(self.est_state.vec[0], self.est_state.vec[1])

        if self.config['agent']['use_sonar']:
            if self.count == self.period:
                self.update_gridmap_rangefinder_based(sensors)
                self.count = 0
            else:
                self.count += 1

        if self.config['agent']['controller'] == "LQR":
            if self.config['agent']['controller_config']['LQR']['action_dim'] == 3: 
                des_state = State(np.array([action_waypoint[0], action_waypoint[1], depth,
                                            0.00, 0.00, 0.00,
                                            0.00, 0.00, action_waypoint[2],
                                            -0.00, -0.00, 0.00]))
            elif self.config['agent']['controller_config']['LQR']['action_dim'] == 4:
                des_state = State(np.array([action_waypoint[0], action_waypoint[1], action_waypoint[2],
                                                0.00, 0.00, 0.00,
                                                0.00, 0.00, action_waypoint[3],
                                                -0.00, -0.00, 0.00]))
            u = self.controller.u(self.est_state, des_state)
        elif self.config['agent']['controller'] == "PID":
            u = self.controller.u(self.est_state, action_waypoint)
        elif self.config['agent']['controller'] == "KEYBOARD":
            u = self.controller.parse_keys()
        else:
            raise ValueError("Unknown controller choice")

        return u

    def update_gridmap_rangefinder_based(self, sensors):
        # get the gridmap part
        # get the agent's pose
        x_odom, y_odom = self.est_state.vec[:2]  # x,y in [m]
        theta_odom = np.radians(self.est_state.vec[8])  # rad

        distances_x, distances_y, distances = self.rangefinder.scan([x_odom, y_odom], self.est_state.vec[8])

        ##################### Grid map update section #####################
        # 机器人当前坐标(x1, y1)
        x1, y1 = self.gridMap.discretize(x_odom, y_odom)
        # for image of the grid map
        X2 = []
        Y2 = []

        # 类似lidar的原理更新free space
        for (dist_x, dist_y, dist) in zip(distances_x, distances_y, distances):
            # 障碍物的坐标(x2, y2)
            x2, y2 = self.gridMap.discretize(dist_x, dist_y)
            # draw a discrete line of free pixels, [robot position -> laser hit spot), 确定测量范围内的free space
            for (x_bres, y_bres) in bresenham(self.grid_map, x1, y1, x2, y2):
                self.grid_map.update(x=x_bres, y=y_bres, p=self.P_free)
            # for BGR image of the grid map
            X2.append(x2)
            Y2.append(y2)

        # 更新occ space
        for (dist_x, dist_y, dist) in zip(distances_x, distances_y, distances):
            if dist < self.zmax:
                # 障碍物的坐标(x2, y2)
                x2, y2 = self.grid_map.discretize(dist_x, dist_y)

                # 检测到障碍物，更新occ grid map
                self.grid_map.update(x=x2, y=y2, p=self.P_occ)

        if self.config['render']:
            gray_map = self.grid_map.to_grayscale_image()
            scale_factor = 2
            new_size = (gray_map.shape[1] * scale_factor, gray_map.shape[0] * scale_factor)
            resized_map = cv2.resize(gray_map, new_size, interpolation=cv2.INTER_NEAREST)

            cv2.imshow("Grid map", resized_map)
            cv2.waitKey(1)


class AgentAuvTarget(Agent):
    """
        use for target of HoveringAUV
    """
    def __init__(self, rank, dim,
                  sampling_period, sensor, obstacles, fixed_depth, size, bottom_corner, start_time,
                  scene, scenario, config):
        Agent.__init__(self, dim, sampling_period, config)
        self.rank = rank
        self.size = size
        self.bottom_corner = bottom_corner
        # Normalize fixed depth to scalar
        if np.isscalar(fixed_depth):
            self.fix_depth = float(fixed_depth)
        elif isinstance(fixed_depth, (list, tuple, np.ndarray)):
            self.fix_depth = float(np.mean(fixed_depth))
        else:
            self.fix_depth = -5.0
        self.obstacles = obstacles
        self.scene = scene
        scenario_cfg = holoocean.packagemanager.get_scenario(scenario)
        robo_type = scenario_cfg['agents'][rank]['agent_type']
        # init the controller part of Auv
        if self.config['target']['controller'] == 'LQR':
            lqr_config = self.config['target']['controller_config']['LQR']
            if lqr_config['random_lp']:
                l_p = np.random.choice(lqr_config['l_p'][1])
            else:
                l_p = lqr_config['l_p'][0]
            self.controller = LQR(l_p=l_p, l_v=lqr_config['l_v'],
                                l_r=lqr_config['l_r'],
                                r_f=lqr_config['r_f'],
                                r_t=lqr_config['r_t'],
                                robo_type=robo_type)
        self.state = State(sensor)
        # init planner rrt
        self.planner = RRT_2d(obstacles=self.obstacles, margin=self.margin,
                            fixed_depth=self.fix_depth, num_seconds=30,
                            bottom_corner=self.bottom_corner, size=self.size,
                            render=self.config['render'], draw_flag=self.config['draw_traj'])
        # reset
        # _target = None
        # is_end_valid = False
        # while not is_end_valid:
        #     _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
        #     is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
        #                                                                                       self.margin2wall + 2)
        # self.target_pos = np.append(_target, self.fix_depth)
        # while not self.planner.reset(start=self.init_pos, end=self.target_pos, time=start_time):
        #     # do not generate the correct path
        #     is_end_valid = False
        #     while not is_end_valid:
        #         _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
        #         is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
        #                                                                                           self.margin2wall + 2)
        #     self.target_pos = np.append(_target, self.fix_depth)
        # if self.planner.desire_path_num == 0:
        #     self.scene.agents['target'+str(self.rank)].teleport(rotation=[0.0, 0.0,
        #                                                    -np.rad2deg(np.arctan2(
        #                                                        self.planner.path[1, 1] - self.planner.path[1, 0],
        #                                                        self.planner.path[0, 1] - self.planner.path[0, 0]))])

    def reset(self, sensor, obstacles, scene, start_time):
        self.obstacles = obstacles
        self.scene = scene  # have a check if is changed
        self.state = State(sensor)
        self.init_pos = self.state.vec[:3]
        # reset rrt
        _target = None
        is_end_valid = False
        while not is_end_valid:
            _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
            is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
                                                                                              self.margin2wall + 2)
        if not self.config['eval_fixed']:
            self.target_pos = np.append(_target, self.fix_depth)
        else:
            self.target_pos = np.array([-15,15,-5])
        while not self.planner.reset(start=self.init_pos, end=self.target_pos, time=start_time):
            is_end_valid = False
            while not is_end_valid:
                _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
                is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
                                                                                                  self.margin2wall + 2)
            self.target_pos = np.append(_target, self.fix_depth)
        if self.planner.desire_path_num == 0:
            self.scene.agents['target'+str(self.rank)].teleport(rotation=[0.0, 0.0,
                                                           np.rad2deg(np.arctan2(
                                                               self.planner.path[1, 1] - self.planner.path[1, 0],
                                                               self.planner.path[0, 1] - self.planner.path[0, 0]))])
        if self.config['draw_traj']:
            self.planner.draw_traj(self.scene, 30)

    def update(self, sensors, t):
        # update time
        self.time = t
        # true state
        self.state = State(sensors)
        true_state = np.array([self.state.vec[0], self.state.vec[1],
                               np.radians(self.state.vec[8])])
        # desire state
        if self.planner.finish_flag == 1:
            _target = None
            is_end_valid = False
            while not is_end_valid:
                _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
                is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
                                                                                                  self.margin2wall)
            if not self.config['eval_fixed']:
                self.target_pos = np.append(_target, self.fix_depth)
            else:
                self.target_pos = np.array([-15, 15, -5])
            while not self.planner.reset(start=true_state, end=self.target_pos, time=t):
                is_end_valid = False
                while not is_end_valid:
                    _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
                    is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
                                                                                                      self.margin2wall)
                self.target_pos = np.append(_target, self.fix_depth)
            if self.config['draw_traj']:
                self.planner.draw_traj(self.scene, 30)
        if self.planner.desire_path_num == 0:
            self.scene.agents['target'+str(self.rank)].teleport(rotation=[0.0, 0.0,
                                                           np.rad2deg(np.arctan2(
                                                               self.planner.path[1, 1] - self.planner.path[1, 0],
                                                               self.planner.path[0, 1] - self.planner.path[0, 0]))])
        des_state = self.planner.tick(true_state)  # only x, y
        if self.planner.desire_path_num != 0:
            angle = np.rad2deg(np.arctan2(
                self.planner.path[1, self.planner.desire_path_num] - self.planner.path[1, self.planner.desire_path_num - 1],
                self.planner.path[0, self.planner.desire_path_num] - self.planner.path[0, self.planner.desire_path_num - 1]))
        else:
            angle = 0

        des_state = State(np.array([des_state[0], des_state[1], self.fix_depth,
                                    0.00, 0.00, 0.00,
                                    0.00, 0.00, angle,
                                    -0.00, -0.00, 0.00]))
        u = self.controller.u(self.state, des_state)
        return u

    def in_bound(self, pos):
        """
        :param pos:
        :return: True: in area, False: out area
        """
        return not ((pos[0] < self.bottom_corner[0] + self.margin2wall)
                    or (pos[0] > self.size[0] + self.bottom_corner[0] - self.margin2wall)
                    or (pos[1] < self.bottom_corner[1] + self.margin2wall)
                    or (pos[1] > self.size[1] + self.bottom_corner[1] - self.margin2wall))

class AgentAuvTarget3D(Agent):
    """
        use for 3D motion target of HoveringAUV
    """
    def __init__(self, rank, dim,
                  sampling_period, sensor, obstacles, size, bottom_corner, start_time,
                  scene, scenario, config):
        Agent.__init__(self, dim, sampling_period, config)
        self.rank = rank
        self.size = size
        self.bottom_corner = bottom_corner
        self.obstacles = obstacles
        self.scene = scene
        scenario_cfg = holoocean.packagemanager.get_scenario(scenario)
        robo_type = scenario_cfg['agents'][rank]['agent_type']
        # init the controller part of Auv
        if self.config['target']['controller'] == 'LQR':
            lqr_config = self.config['target']['controller_config']['LQR']
            if lqr_config['random_lp']:
                l_p = np.random.choice(lqr_config['l_p'][1])
            else:
                l_p = lqr_config['l_p'][0]
            self.controller = LQR(l_p=l_p, l_v=lqr_config['l_v'],
                                l_r=lqr_config['l_r'],
                                r_f=lqr_config['r_f'],
                                r_t=lqr_config['r_t'],
                                robo_type=robo_type)
        self.state = State(sensor)
        # init planner rrt
        self.planner = RRT_3d(obstacles=self.obstacles, margin=self.margin,
                            num_seconds=30,
                            bottom_corner=self.bottom_corner, size=self.size,
                            render=self.config['render'], draw_flag=self.config['draw_traj'])

    def __generate_valid_target(self):
        while True:
            target = np.random.random((3,)) * self.size[0:3] + self.bottom_corner[0:3]
            if (self.in_bound(target) and 
                self.obstacles.check_obstacle_collision(target, self.margin2wall)):
                return target
            
    def reset(self, sensor, obstacles, scene, start_time):
        self.obstacles = obstacles
        self.scene = scene  # have a check if is changed
        self.state = State(sensor)
        self.init_pos = self.state.vec[:3]
        
        # 设置初始目标点
        if not self.config['eval_fixed']:
            self.target_pos = self.__generate_valid_target()
        else:
            self.target_pos = np.array([-15, 15, -5])

        while not self.planner.reset(start=self.init_pos, end=self.target_pos, time=start_time):
            self.target_pos = self.__generate_valid_target()

        if self.planner.desire_path_num == 0:
            self.scene.agents['target'+str(self.rank)].teleport(rotation=[0.0, 0.0,
                                                           np.rad2deg(np.arctan2(
                                                               self.planner.path[1, 1] - self.planner.path[1, 0],
                                                               self.planner.path[0, 1] - self.planner.path[0, 0]))])
        if self.config['draw_traj']:
            self.planner.draw_traj(self.scene, 30)

    def update(self, sensors, t):
        # update time
        self.time = t
        # true state
        self.state = State(sensors)
        true_state = self.state.vec[:3]
        # desire state
        if self.planner.finish_flag == 1:
            if not self.config['eval_fixed']:
                self.target_pos = self.__generate_valid_target()
            else:
                self.target_pos = np.array([-15, 15, -5])

            while not self.planner.reset(time=t, start=true_state, end=self.target_pos):
                self.target_pos = self.__generate_valid_target()

            if self.config['draw_traj']:
                self.planner.draw_traj(self.scene, 30)
        
        # if self.planner.desire_path_num == 0:
        #     self.scene.agents['target'+str(self.rank)].teleport(rotation=[0.0, 0.0,
        #                                                    np.rad2deg(np.arctan2(
        #                                                        self.planner.path[1, 1] - self.planner.path[1, 0],
        #                                                        self.planner.path[0, 1] - self.planner.path[0, 0]))])
        
        des_state = self.planner.tick(true_state)
        
        if self.planner.desire_path_num != 0:
            angle = np.rad2deg(np.arctan2(
                self.planner.path[1, self.planner.desire_path_num] - self.planner.path[
                    1, self.planner.desire_path_num - 1],
                self.planner.path[0, self.planner.desire_path_num] - self.planner.path[
                    0, self.planner.desire_path_num - 1]))
        else:
            angle = 0

        des_state = State(np.array([des_state[0], des_state[1], des_state[2],
                                    0.00, 0.00, 0.00,
                                    0.00, 0.00, angle,
                                    -0.00, -0.00, 0.00]))
        u = self.controller.u(self.state, des_state)
        return u

    def in_bound(self, pos):
        """
        :param pos:
        :return: True: in area, False: out area
        """
        return not ((pos[0] < self.bottom_corner[0] + self.margin2wall)
                    or (pos[0] > self.size[0] + self.bottom_corner[0] - self.margin2wall)
                    or (pos[1] < self.bottom_corner[1] + self.margin2wall)
                    or (pos[1] > self.size[1] + self.bottom_corner[1] - self.margin2wall)
                    or (pos[2] < self.bottom_corner[2] + self.margin2wall)
                    or (pos[2] > self.size[2] + self.bottom_corner[2] - self.margin2wall))

class AgentAuvTarget3DRangeFinder(Agent):
    """
    3D target agent with RangeFinder-based obstacle avoidance for random motion
    Uses only RangeFinder sensor data to determine collision-free waypoints and move toward them
    """
    
    def __init__(self, rank, dim, sampling_period, sensor, size, bottom_corner, start_time,
                 scene, scenario, config):
        Agent.__init__(self, dim, sampling_period, config)
        self.rank = rank
        self.scene = scene
        
        scenario_cfg = holoocean.packagemanager.get_scenario(scenario)
        robo_type = scenario_cfg['agents'][1]['agent_type']
        
        if self.config['target']['controller'] == 'Auto':
            lqr_config = self.config['target']['controller_config']['LQR']
            if lqr_config['random_lp']:
                l_p = np.random.choice(lqr_config['l_p'][1])
            else:
                l_p = lqr_config['l_p'][0]
            self.controller = LQR(l_p=l_p, l_v=lqr_config['l_v'],
                                l_r=lqr_config['l_r'],
                                r_f=lqr_config['r_f'],
                                r_t=lqr_config['r_t'],
                                robo_type=robo_type)
        else:
            assert False, "Only Auto controller is supported"
        
        self.state = State(sensor)

        for sensor in scenario_cfg['agents'][1]['sensors']:
            if sensor['sensor_type'] == 'RangeFinderSensor':
                self.LaserMaxDistance = sensor["configuration"]['LaserMaxDistance']
                self.LaserCount = sensor["configuration"]['LaserCount']
                self.LaserAngle = sensor["configuration"]['LaserAngle']
                self.LaserDebug = sensor["configuration"]['LaserDebug']
        self.rangefinder1 = np.zeros((self.LaserCount,))
        self.rangefinder2 = np.zeros((self.LaserCount,))

        # Motion parameters
        self.target_waypoint = None
        self.waypoint_reach_threshold = 0.3

        self.arrive = True

        
        
    def reset(self, sensor, obstacles, scene, start_time):
        self.scene = scene
        self.state = State(sensor)

        self.target_waypoint = None
        self.arrive = True
        
    def update(self, sensors, t):
        self.time = t
        self.state = State(sensors)
        current_pos = self.state.vec[:3]
        if 'rangefinder1' in sensors:
            self.rangefinder1 = sensors['rangefinder1']
        if 'rangefinder2' in sensors:
            self.rangefinder2 = sensors['rangefinder2']
        
        # Generate new waypoint if needed
        if self.target_waypoint is not None:
            self.arrive = np.linalg.norm(current_pos - self.target_waypoint) < self.waypoint_reach_threshold

        if self.arrive:
            # Try to find a collision-free waypoint
            waypoint_found = False
            while waypoint_found is False:
                idx = np.random.randint(0, 2)
                count = np.random.randint(0, 7) - 3
                if idx == 0:
                    if self.rangefinder1[count] > 2.0:
                        waypoint_found = True
                        r = np.random.uniform(0.5, 1.0)
                        z = r * np.sin(np.radians(self.LaserAngle))
                elif idx == 1:
                    if self.rangefinder2[count] > 2.0:
                        waypoint_found = True
                        r = np.random.uniform(0.5, 1.0)
                        z = -r * np.sin(np.radians(self.LaserAngle))

                if waypoint_found == True:
                    theta = count * (360/self.LaserCount)
                    target_pos = util.polar_distance_global(np.array([r, np.radians(theta)]), self.state.vec[:2],
                                                    np.radians(self.state.vec[8]))
                    yaw = np.rad2deg(np.deg2rad(self.state.vec[8] + theta))
                    depth = self.state.vec[2] + z if self.state.vec[2] + z < 0 else self.state.vec[2]
                    self.desired_state_vec = np.array([
                        target_pos[0], target_pos[1], depth, 
                        0 ,0, 0,
                        0.0, 0.0, yaw,
                        0.0, 0.0, 0.0
                    ])
                    self.target_waypoint = np.array([target_pos[0], target_pos[1], depth])
                    print(f"New target waypoint: {self.target_waypoint}")
                    self.scene.draw_point(self.target_waypoint.tolist(), color=[0,255,0], thickness=20, lifetime=1.0)
        
        desired_state = State(self.desired_state_vec)

        # Get control input from LQR controller
        u = self.controller.u(self.state, desired_state)
        
        return u


class AgentAuvTarget2DRangeFinder(Agent):
    """
    2D target agent with RangeFinder-based random motion (x, y, yaw, fixed depth).
    Mirrors the 3D rangefinder target logic but constrained to 2D state (target_dim=4).
    """

    def __init__(self, rank, dim, sampling_period, sensor, fixed_depth, size, bottom_corner,
                 start_time, scene, scenario, config):
        Agent.__init__(self, dim, sampling_period, config)
        self.rank = rank
        self.scene = scene
        self.fix_depth = fixed_depth
        self.size = size
        self.bottom_corner = bottom_corner

        scenario_cfg = holoocean.packagemanager.get_scenario(scenario)
        robo_type = scenario_cfg['agents'][1]['agent_type']

        if self.config['target']['controller'] == 'Auto':
            lqr_config = self.config['target']['controller_config']['LQR']
            if lqr_config['random_lp']:
                l_p = np.random.choice(lqr_config['l_p'][1])
            else:
                l_p = lqr_config['l_p'][0]
            self.controller = LQR(l_p=l_p, l_v=lqr_config['l_v'],
                                  l_r=lqr_config['l_r'],
                                  r_f=lqr_config['r_f'],
                                  r_t=lqr_config['r_t'],
                                  robo_type=robo_type)
        else:
            assert False, "Only Auto controller is supported for 2D RangeFinder target"

        self.state = State(sensor)

        # RangeFinder params (assume target agent has RangeFinderSensor like 3D case)
        for sensor_cfg in scenario_cfg['agents'][1]['sensors']:
            if sensor_cfg['sensor_type'] == 'RangeFinderSensor':
                self.LaserMaxDistance = sensor_cfg["configuration"]['LaserMaxDistance']
                self.LaserCount = sensor_cfg["configuration"]['LaserCount']
                self.LaserAngle = sensor_cfg["configuration"]['LaserAngle']
                self.LaserDebug = sensor_cfg["configuration"]['LaserDebug']
        self.rangefinder1 = np.zeros((self.LaserCount,))
        self.rangefinder2 = np.zeros((self.LaserCount,))

        # Motion parameters
        self.target_waypoint = None
        self.waypoint_reach_threshold = 0.3
        self.arrive = True

        # default desired state
        self.desired_state_vec = sensor if isinstance(sensor, np.ndarray) else np.array(sensor)

    def reset(self, sensor, obstacles, scene, start_time):
        self.scene = scene
        self.state = State(sensor)
        self.target_waypoint = None
        self.arrive = True
        # Initialize desired state as 12-dim zero with fix_depth and current yaw
        desired = np.zeros(12, dtype=float)
        desired[2] = float(self.fix_depth)
        desired[8] = self.state.vec[8] if len(self.state.vec) > 8 else 0.0
        self.desired_state_vec = desired
        # Re-normalize depth in case config passed a list/array
        if np.isscalar(self.fix_depth):
            self.fix_depth = float(self.fix_depth)
        else:
            self.fix_depth = float(np.mean(self.fix_depth))

    def update(self, sensors, t):
        self.time = t
        self.state = State(sensors)
        current_pos = self.state.vec[:2]
        current_yaw = self.state.vec[8] if len(self.state.vec) > 8 else 0.0  # yaw in degrees
        depth_scalar = float(self.fix_depth)

        if 'rangefinder1' in sensors:
            self.rangefinder1 = sensors['rangefinder1']
        if 'rangefinder2' in sensors:
            self.rangefinder2 = sensors['rangefinder2']

        # Check arrival at current waypoint
        if self.target_waypoint is not None:
            self.arrive = np.linalg.norm(current_pos - self.target_waypoint[:2]) < self.waypoint_reach_threshold

        if self.arrive:
            waypoint_found = False
            attempt = 0
            while waypoint_found is False and attempt < 50:
                attempt += 1
                idx = np.random.randint(0, 2)
                count = np.random.randint(0, 7) - 3
                usable = False
                if idx == 0 and self.rangefinder1[count] > 2.0:
                    usable = True
                elif idx == 1 and self.rangefinder2[count] > 2.0:
                    usable = True
                if usable:
                    r = np.random.uniform(0.5, 1.0)
                    theta = count * (360 / self.LaserCount)
                    target_xy = util.polar_distance_global(np.array([r, np.radians(theta)]),
                                                           current_pos,
                                                           np.radians(current_yaw))
                    yaw = np.rad2deg(np.deg2rad(current_yaw + theta))
                    depth = depth_scalar
                    desired = np.zeros(12, dtype=float)
                    desired[0] = target_xy[0]
                    desired[1] = target_xy[1]
                    desired[2] = depth
                    desired[8] = yaw
                    self.desired_state_vec = desired
                    self.target_waypoint = np.array([target_xy[0], target_xy[1], depth])
                    waypoint_found = True
                    self.scene.draw_point(self.target_waypoint.tolist(), color=[0, 255, 0],
                                          thickness=20, lifetime=1.0)

        desired_state = State(self.desired_state_vec)
        u = self.controller.u(self.state, desired_state)
        return u
class AgentAuvManual(Agent):
    """
        use for target manual
    """

    def __init__(self, dim, sampling_period, sensor, fixed_depth, scene, config):
        Agent.__init__(self, dim, sampling_period, config)
        self.fix_depth = fixed_depth
        self.scene = scene
        self.controller = KeyBoardCmd(force=self.config['target']['controller_config']['Manual']['force'])
        self.state = State(sensor)

    def reset(self, *args, **kwargs):
        return 0
    
    def update(self, sensors, t):
         # update time
        self.time = t
        # true state
        self.state = State(sensors)
        command = self.controller.parse_keys()
        return command



#########################
# agent model:
def SE2Dynamics(x, dt, u):
    """
    update dynamics function with a control input -- linear, angular velocities
    """
    assert (len(x) == 3)
    tw = dt * u[1]  # tau * w

    # Update the agent state
    if abs(tw) < 0.001:
        diff = np.array([dt * u[0] * np.cos(x[2] + tw / 2),
                         dt * u[0] * np.sin(x[2] + tw / 2),
                         tw])
    else:
        diff = np.array([u[0] / u[1] * (np.sin(x[2] + tw) - np.sin(x[2])),
                         u[0] / u[1] * (np.cos(x[2]) - np.cos(x[2] + tw)),
                         tw])
    new_x = x + diff
    new_x[2] = util.wrap_around(new_x[2])
    return new_x
