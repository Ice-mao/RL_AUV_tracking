from metadata import METADATA
from auv_env.envs.tools import KeyBoardCmd, ImagingSonar, RangeFinder
import auv_env.util as util

from auv_control.estimation import InEKF
from auv_control.control import LQR, SE2PIDController, PID
from auv_control.planning import RRT_2d
from auv_control.state import rot_to_rpy
from auv_control import State

from gridmap.grid_map import *


class Agent(object):
    def __init__(self, dim, sampling_period):
        self.dim = dim
        self.sampling_period = sampling_period
        self.margin = METADATA['env']['margin']
        self.margin2wall = METADATA['env']['margin2wall']

    def range_check(self):
        self.state = np.clip(self.state, self.limit[0], self.limit[1])

    def collision_check(self, pos):
        return self.collision_func(pos[:2])

    def margin_check(self, pos, target_pos):
        return any(np.sqrt(np.sum((pos - target_pos) ** 2, axis=1)) < self.margin)  # no update

    def reset(self, init_state):
        self.state = State(init_state)


class AgentAuv(Agent):
    def __init__(self, dim, sampling_period, sensor, scenario, controller = "LQR"):
        Agent.__init__(self, dim, sampling_period)
        # init the control part of Auv
        if controller == "LQR":
            if METADATA['agent']['random']:
                l_p = np.random.normal(40, 15)
            else:
                l_p = 50
            self.controller_choice = "LQR"
            self.controller = LQR(l_p=l_p)
        elif controller == "PID":
            self.controller_choice = "PID"
            self.controller = PID()
        else:
            raise ValueError("Unknown controller choice")

        # init the sonar and grid map part
        if METADATA['agent']['use_sonar']:
            self.imagingsonar = ImagingSonar(scenario=scenario)  # to see if it is useful
            self.zmax = self.imagingsonar.zmax
        else:
            self.zmax = 8.0

        if METADATA['agent']['grid']['use_grid']:
            # count for grid map
            self.count = 0
            self.period = 10
            # probability
            self.P_prior = METADATA['agent']['grid']['p_prior']  # Prior occupancy probability
            self.P_occ = METADATA['agent']['grid']['p_occ']  # Probability that cell is occupied with total confidence
            self.P_free = METADATA['agent']['grid']['p_free']  # Probability that cell is free with total confidence
            self.RESOLUTION = METADATA['agent']['grid']['resolution']  # Grid resolution in [m]
            self.gridMap = GridMap(X_lim=np.array([METADATA['scenario']['bottom_corner'][0],
                                                   METADATA['scenario']['bottom_corner'][0] + METADATA['scenario']['size'][0]]),
                                   Y_lim=np.array([METADATA['scenario']['bottom_corner'][1],
                                                   METADATA['scenario']['bottom_corner'][1] + METADATA['scenario']['size'][1]]),
                                   resolution=self.RESOLUTION,
                                   p=self.P_prior)

        # init the sensor part of AUV
        self.rangefinder = RangeFinder(scenario=scenario)
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
        # if METADATA['use_sonar']:
        #     self.update_gridmap(sensors)
        if METADATA['agent']['use_sonar']:
            if self.count == self.period:
                self.update_gridmap_rangefinder_based(sensors)
                self.count = 0
            else:
                self.count += 1

        # # Autopilot Commands
        if self.controller_choice == "LQR":
            des_state = State(np.array([action_waypoint[0], action_waypoint[1], depth,
                                        0.00, 0.00, 0.00,
                                        0.00, 0.00, action_waypoint[2],
                                        -0.00, -0.00, 0.00]))
            u = self.controller.u(self.est_state, des_state)
        elif self.controller_choice == "PID":
            u = self.controller.u(self.est_state, action_waypoint)
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
            for (x_bres, y_bres) in bresenham(self.gridMap, x1, y1, x2, y2):
                self.gridMap.update(x=x_bres, y=y_bres, p=self.P_free)
            # for BGR image of the grid map
            X2.append(x2)
            Y2.append(y2)

        # 更新occ space
        for (dist_x, dist_y, dist) in zip(distances_x, distances_y, distances):
            if dist < self.zmax:
                # 障碍物的坐标(x2, y2)
                x2, y2 = self.gridMap.discretize(dist_x, dist_y)

                # 检测到障碍物，更新occ grid map
                self.gridMap.update(x=x2, y=y2, p=self.P_occ)

        # self.gridMap.cal_entropy()

        if METADATA['render']:
            gray_map = self.gridMap.to_grayscale_image()
            scale_factor = 2
            new_size = (gray_map.shape[1] * scale_factor, gray_map.shape[0] * scale_factor)
            resized_map = cv2.resize(gray_map, new_size, interpolation=cv2.INTER_NEAREST)

            cv2.imshow("Grid map", resized_map)
            cv2.waitKey(1)



class AgentSphere(Agent):
    """
        use for target
    """

    def __init__(self, dim, sampling_period, sensor, obstacles, fixed_depth, size, bottom_corner, start_time, scene):
        Agent.__init__(self, dim, sampling_period)
        self.size = size
        self.bottom_corner = bottom_corner
        self.fix_depth = fixed_depth
        self.obstacles = obstacles
        self.scene = scene
        # init the control part of Auv
        self.init_pos = sensor['LocationSensor']
        _target = None
        is_end_valid = False
        while not is_end_valid:
            _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
            is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
                                                                                              self.margin2wall + 2)
        self.target_pos = np.append(_target, self.fix_depth)
        self.planner = RRT_2d(start=self.init_pos, end=self.target_pos, obstacles=self.obstacles, margin=self.margin,
                              fixed_depth=self.fix_depth, num_seconds=30,
                              bottom_corner=self.bottom_corner, size=self.size, start_time=start_time)
        # reset
        _target = None
        is_end_valid = False
        while not is_end_valid:
            _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
            is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
                                                                                              self.margin2wall + 2)
        self.target_pos = np.append(_target, self.fix_depth)
        while not self.planner.reset(start=self.init_pos, end=self.target_pos, time=start_time):
            # do not generate the correct path
            is_end_valid = False
            while not is_end_valid:
                _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
                is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
                                                                                                  self.margin2wall + 2)
            self.target_pos = np.append(_target, self.fix_depth)
        if self.planner.desire_path_num == 0:
            self.scene.agents['target'].teleport(rotation=[0.0, 0.0,
                                                           -np.rad2deg(np.arctan2(
                                                               self.planner.path[1, 1] - self.planner.path[1, 0],
                                                               self.planner.path[0, 1] - self.planner.path[0, 0]))])
        self.controller = SE2PIDController()

        # init the sensor part of AUV
        # to see if it is useful
        # self.imagesonar = ImagingSonar(scenario=scenario)  # to see if it is useful
        self.vec = []
        self.vec[0:3] = sensor["PoseSensor"][:3, 3]
        self.vec[3:6] = sensor["VelocitySensor"]
        self.vec[6:9] = rot_to_rpy(sensor["PoseSensor"][:3, :3])

    def reset(self, sensor, obstacles, scene, start_time):
        self.obstacles = obstacles
        self.scene = scene  # have a check if is changed
        # init the control part of AUV
        self.init_pos = sensor['LocationSensor']
        _target = None
        is_end_valid = False
        while not is_end_valid:
            _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
            is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
                                                                                              self.margin2wall + 2)
        self.target_pos = np.append(_target, self.fix_depth)
        while not self.planner.reset(start=self.init_pos, end=self.target_pos, time=start_time):
            is_end_valid = False
            while not is_end_valid:
                _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
                is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
                                                                                                  self.margin2wall + 2)
            self.target_pos = np.append(_target, self.fix_depth)
        if self.planner.desire_path_num == 0:
            self.scene.agents['target'].teleport(rotation=[0.0, 0.0,
                                                           -np.rad2deg(np.arctan2(
                                                               self.planner.path[1, 1] - self.planner.path[1, 0],
                                                               self.planner.path[0, 1] - self.planner.path[0, 0]))])
        if METADATA['render']:
            self.planner.draw_traj(self.scene, 30)
        self.controller.reset()
        self.vec = []
        self.vec[0:3] = sensor["PoseSensor"][:3, 3]
        self.vec[3:6] = sensor["VelocitySensor"]
        self.vec[6:9] = rot_to_rpy(sensor["PoseSensor"][:3, :3])

    def update(self, sensor, t):
        # update time
        self.time = t
        # true state
        self.vec[0:3] = sensor["PoseSensor"][:3, 3]
        self.vec[3:6] = sensor["VelocitySensor"]
        self.vec[6:9] = rot_to_rpy(sensor["PoseSensor"][:3, :3])
        true_state = np.array([self.vec[0], self.vec[1], np.radians(self.vec[8])])
        # desire state
        if self.planner.finish_flag == 1:
            _target = None
            is_end_valid = False
            while not is_end_valid:
                _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
                is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
                                                                                                  self.margin2wall)
            self.target_pos = np.append(_target, self.fix_depth)
            while not self.planner.reset(start=true_state, end=self.target_pos, time=t):
                is_end_valid = False
                while not is_end_valid:
                    _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
                    is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
                                                                                                      self.margin2wall)
                self.target_pos = np.append(_target, self.fix_depth)
            if METADATA['render']:
                self.planner.draw_traj(self.scene, 30)
        if self.planner.desire_path_num == 0:
            self.scene.agents['target'].teleport(rotation=[0.0, 0.0,
                                                           -np.rad2deg(np.arctan2(
                                                               self.planner.path[1, 1] - self.planner.path[1, 0],
                                                               self.planner.path[0, 1] - self.planner.path[0, 0]))])
        des_state = self.planner.tick(true_state)  # only x, y
        # Autopilot Commands
        u = self.controller.u(true_state, des_state, self.sampling_period)
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


class AgentAuvTarget(Agent):
    """
        use for target of HoveringAUV
    """

    def __init__(self, rank, dim, sampling_period, sensor, obstacles, fixed_depth, size, bottom_corner, start_time, scene, l_p):
        Agent.__init__(self, dim, sampling_period)
        self.rank = rank
        self.size = size
        self.bottom_corner = bottom_corner
        self.fix_depth = fixed_depth
        self.obstacles = obstacles
        self.scene = scene
        # init the part of Auv
        if METADATA['target']['random']:
            data = [20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, 250]
            self.controller = LQR(l_p=np.random.choice(data), l_v=0.001)
        else:
            self.controller = LQR(l_p=l_p, l_v=0.001)
        self.state = State(sensor)
        # init planner rrt
        self.init_pos = self.state.vec[:3]
        _target = None
        is_end_valid = False
        while not is_end_valid:
            _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
            is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
                                                                                              self.margin2wall + 2)
        self.target_pos = np.append(_target, self.fix_depth)
        self.planner = RRT_2d(start=self.init_pos, end=self.target_pos, obstacles=self.obstacles, margin=self.margin,
                              fixed_depth=self.fix_depth, num_seconds=30,
                              bottom_corner=self.bottom_corner, size=self.size, start_time=start_time)
        # reset
        _target = None
        is_end_valid = False
        while not is_end_valid:
            _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
            is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
                                                                                              self.margin2wall + 2)
        self.target_pos = np.append(_target, self.fix_depth)
        while not self.planner.reset(start=self.init_pos, end=self.target_pos, time=start_time):
            # do not generate the correct path
            is_end_valid = False
            while not is_end_valid:
                _target = np.random.random((2,)) * self.size[0:2] + self.bottom_corner[0:2]
                is_end_valid = self.in_bound(_target) and self.obstacles.check_obstacle_collision(_target,
                                                                                                  self.margin2wall + 2)
            self.target_pos = np.append(_target, self.fix_depth)
        if self.planner.desire_path_num == 0:
            self.scene.agents['target'+str(self.rank)].teleport(rotation=[0.0, 0.0,
                                                           -np.rad2deg(np.arctan2(
                                                               self.planner.path[1, 1] - self.planner.path[1, 0],
                                                               self.planner.path[0, 1] - self.planner.path[0, 0]))])

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
        if not METADATA['eval_fixed']:
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
        if METADATA['render']:
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
            if not METADATA['eval_fixed']:
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
            if METADATA['render']:
                self.planner.draw_traj(self.scene, 30)
        if self.planner.desire_path_num == 0:
            self.scene.agents['target'+str(self.rank)].teleport(rotation=[0.0, 0.0,
                                                           np.rad2deg(np.arctan2(
                                                               self.planner.path[1, 1] - self.planner.path[1, 0],
                                                               self.planner.path[0, 1] - self.planner.path[0, 0]))])
        des_state = self.planner.tick(true_state)  # only x, y
        if self.planner.desire_path_num != 0:
            angle = np.rad2deg(np.arctan2(
                self.planner.path[1, self.planner.desire_path_num] - self.planner.path[
                    1, self.planner.desire_path_num - 1],
                self.planner.path[0, self.planner.desire_path_num] - self.planner.path[
                    0, self.planner.desire_path_num - 1]))
        else:
            angle = 0

        des_state = State(np.array([des_state[0], des_state[1], self.fix_depth,
                                    0.00, 0.00, 0.00,
                                    0.00, 0.00, angle,
                                    -0.00, -0.00, 0.00]))
        # Autopilot Commands
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

class AgentAuvManual(Agent):
    """
        use for target manual
    """

    def __init__(self, dim, sampling_period, sensor, fixed_depth, scene):
        Agent.__init__(self, dim, sampling_period)
        self.fix_depth = fixed_depth
        self.scene = scene
        self.controller = KeyBoardCmd(force=10)
        self.state = State(sensor)

    def reset(self):
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
