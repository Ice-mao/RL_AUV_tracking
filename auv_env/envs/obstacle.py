import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import math

import numpy as np
import copy

from shapely import LineString
from shapely.geometry import Point, Polygon

res = 0.2
obstacles = {
    '0': {'center': [np.array([29.5, 21]) * res], 'scale': [[40, 37, 1]]},
    '1': {'center': [np.array([26.5, 8.5]) * res, np.array([8.5, 24.5]) * res, np.array([26.5, 41.0]) * res],
          'scale': [[48, 14, 1], [12, 18, 1], [48, 15, 1]]},
    '2': {'center': [np.array([26, 43.]) * res], 'scale': [[49, 13, 1]]},
    '3': {'center': [np.array([18.5, 18.5]) * res, np.array([33, 36.5]) * res], 'scale': [[34, 18, 1], [35, 18, 1]]},
    '4': {'center': [np.array([16.5, 9]) * res, np.array([8.5, 32]) * res], 'scale': [[30, 15, 1], [14, 31, 1]]},
    '5': {'center': [np.array([14.5, 26]) * res, np.array([30, 26.5]) * res], 'scale': [[16, 45, 1], [15, 16, 1]]},
    '6': {'center': [np.array([25.5, 8.5]) * res, np.array([42, 32]) * res], 'scale': [[46, 14, 1], [13, 33, 1]]},
    '7': {'center': [np.array([7.5, 10]) * res, np.array([26.5, 25]) * res, np.array([45, 40]) * res],
          'scale': [[10, 17, 1], [48, 13, 1], [11, 17, 1]]},
}


def rotate_point(point, center, angle):
    angle_rad = np.radians(angle)

    # 计算点相对于中心的偏移量
    offset = np.array(point) - np.array(center)

    # 构建旋转矩阵
    # In HoloOcean,x and y is contrary to the Cartesian coordinate system
    rotation_matrix = np.array([[np.sin(angle_rad), np.cos(angle_rad)],
                                [np.cos(angle_rad), -np.sin(angle_rad)]])
    # rotation_matrix = np.array([[np.cos(angle_rad), np.sin(angle_rad)],
    #                             [-np.sin(angle_rad), -np.cos(angle_rad)]])
    # rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
    #                             [np.sin(angle_rad), -np.cos(angle_rad)]])
    # 计算旋转后的偏移量
    new_offset = np.dot(rotation_matrix, offset)

    # 将旋转后的偏移量添加到中心点的坐标上，得到旋转后的点的坐标
    new_point = np.array(center) + new_offset

    return new_point


# def rotate_point(point, center, angle):
#     angle_rad = np.radians(angle)
#
#     # 计算点相对于中心的偏移量
#     offset = np.array(point) - np.array(center)
#
#     # 构建旋转矩阵
#     # In HoloOcean,x and y is contrary to the Cartesian coordinate system
#     rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
#                                 [np.sin(angle_rad), np.cos(angle_rad)]])
#     # rotation_matrix = np.array([[np.sin(angle_rad), np.cos(angle_rad)],
#     #                             [np.cos(angle_rad), -np.sin(angle_rad)]])
#
#     # 计算旋转后的偏移量
#     new_offset = np.dot(rotation_matrix, offset)
#
#     # 将旋转后的偏移量添加到中心点的坐标上，得到旋转后的点的坐标
#     new_point = np.array(center) + new_offset
#
#     return new_point

def polygon_2_points(center, scale, _res, rotate_center, rotate_angle, global_offset):
    # tmp = rotate_point(np.array([center[0] + scale[0] / 2 * _res, center[1] - scale[1] / 2 * _res])
    #              , rotate_center, rotate_angle)
    points = [rotate_point(np.array([center[0] + scale[0] / 2 * _res, center[1] - scale[1] / 2 * _res])
                           , rotate_center, rotate_angle
                           ) + np.array(global_offset),
              rotate_point(np.array([center[0] + scale[0] / 2 * _res, center[1] + scale[1] / 2 * _res])
                           , rotate_center, rotate_angle
                           ) + np.array(global_offset),
              rotate_point(np.array([center[0] - scale[0] / 2 * _res, center[1] + scale[1] / 2 * _res])
                           , rotate_center, rotate_angle
                           ) + np.array(global_offset),
              rotate_point(np.array([center[0] - scale[0] / 2 * _res, center[1] - scale[1] / 2 * _res])
                           , rotate_center, rotate_angle
                           ) + np.array(global_offset)
              ]

    return points


class Obstacle:
    def __init__(self, env, fix_depth, config):
        self.env = env
        self.fix_depth = fix_depth
        self.config = config
        self.num_obstacles = 4
        self.res = 0.2  # m remeber to * with scale
        self.sub_center = [25 * self.res, 25 * self.res]  # m sub obstacle rotate center
        self.sub_coordinates = [np.array([20, 25]) * self.res, np.array([-70, 25]) * self.res,
                                np.array([-70, -65]) * self.res, np.array([20, -65]) * self.res]  # m
        np.random.seed()
        self.chosen_idx = np.random.choice(len(obstacles), self.num_obstacles, replace=True)
        print(self.chosen_idx)
        self.rot_angs = [np.random.choice(np.arange(-10, 10, 1) / 10. * 180) for _ in range(self.num_obstacles)]
        self.polygons = []  # ready for collision detection
        print('finish obstacles')

    def reset(self):
        np.random.seed()
        if not self.config['eval_fixed']:
            self.chosen_idx = np.random.choice(len(obstacles), self.num_obstacles, replace=False)
            self.rot_angs = [np.random.choice(np.arange(-10, 10, 1) / 10. * 180) for _ in range(self.num_obstacles)]
        else:
            self.chosen_idx = np.array([4, 5, 0, 7])
            self.rot_angs = np.array([36.0, -72.0, -144.0, 125.99999999999999])
        # DEBUG
        # self.chosen_idx = np.array([1,2,3,4])
        # self.rot_angs = [45 for _ in range(self.num_obstacles)]
        print(self.chosen_idx)

        self.polygons = []  # ready for collision detection

    def draw_obstacle(self):
        for i in range(self.num_obstacles):
            obstacle = obstacles[str(self.chosen_idx[i])]  # extract the obstacle to decompose
            for j in range(len(obstacle['center'])):
                # get the collision object
                points = polygon_2_points(obstacle['center'][j], obstacle['scale'][j], self.res, self.sub_center,
                                          self.rot_angs[i], self.sub_coordinates[i])
                # drew the obstacle's boundary
                # need the -1 factor, because the coordinates of line is opposite
                if self.config['debug']:
                    self.env.draw_line([points[0][0], points[0][1], self.fix_depth],
                                       [points[1][0], points[1][1], self.fix_depth],
                                       thickness=5.0, lifetime=0.0)
                    self.env.draw_line([points[1][0], points[1][1], self.fix_depth],
                                       [points[2][0], points[2][1], self.fix_depth],
                                       thickness=5.0, lifetime=0.0)
                    self.env.draw_line([points[2][0], points[2][1], self.fix_depth],
                                       [points[3][0], points[3][1], self.fix_depth],
                                       thickness=5.0, lifetime=0.0)
                    self.env.draw_line([points[3][0], points[3][1], self.fix_depth],
                                       [points[0][0], points[0][1], self.fix_depth],
                                       thickness=5.0, lifetime=0.0)
                self.polygons.append(Polygon(points))

                # 使用边框点的中心作为3D障碍物的位置
                loc_center = rotate_point(obstacle['center'][j], self.sub_center, self.rot_angs[i])
                loc = loc_center + np.array(self.sub_coordinates[i])
                loc = np.append(loc, self.fix_depth)
                
                # 使用与边框相同的坐标，不进行任何翻转
                # loc[0] *= -1  # 注释掉x坐标翻转
                # loc[1] *= -1  # 注释掉y坐标翻转
                _scale = [obstacle['scale'][j][0] * self.res, obstacle['scale'][j][1] * self.res,
                          obstacle['scale'][j][2] * 3]
                self.env.spawn_prop(prop_type="box", scale=_scale, location=loc.tolist(),
                                    rotation=[np.tan(np.radians(self.rot_angs[i])), 1, 0],  # it's annoy to be pitch?
                                    # rotation=[0, 1, 0],
                                    material='gold')

    def check_obstacle_collision(self, point, margin):
        """
        :param point: the target point
        :param margin: the distance with obstacle
        :return:True: safe, False: collision
        idea:get a bounding box of the point to check collision
        """
        points = [Point([point[0] + margin, point[1] + margin]),
                  Point([point[0] - margin, point[1] + margin]),
                  Point([point[0] - margin, point[1] - margin]),
                  Point([point[0] + margin, point[1] - margin])]
        circle_polygon = Polygon(points)

        for polygon in self.polygons:
            if circle_polygon.intersects(polygon) or circle_polygon.within(polygon):
                return False  # collision
        return True

    def check_obstacle_block(self, point1, point2, margin=1):
        """
        check if any obstacles blocked between the two points
        :param point1:
        :param point2:
        :return:False:blocked, True: no blocked
        """
        for _ in range(20):
            theta = np.random.uniform(-np.pi, np.pi)
            point1[:2] + np.array([margin * np.cos(theta), -margin * np.sin(theta)])
            theta = np.random.uniform(-np.pi, np.pi)
            point2[:2] + np.array([margin * np.cos(theta), -margin * np.sin(theta)])
            # 定义线段
            line = LineString([point1, point2])

            for polygon in self.polygons:
                if line.intersects(polygon):
                    return False  # blocked
        return True
    
if __name__ == "__main__":
    import holoocean
    import numpy as np
    import time

    with holoocean.make("SimpleUnderwater-Bluerov2") as env:
        obstacle = Obstacle(env, fix_depth=-5, config={'eval_fixed': False, 'render': True})
        obstacle.reset()
        obstacle.draw_obstacle()
        # # Define the properties of the box to be spawned
        # scale = [3, 1, 2]
        # location = [10, 0, -5]
        
        # # angle from 0 to 360 degrees, changing over time
        # rot_angs = 30/180 * 3.1415
        
        # # HoloOcean uses Roll, Pitch, Yaw in degrees for rotation
        # # rotation = [np.tan(np.radians(rot_angs)), 1, 0]
        # rotation = [0,0, rot_angs]  # Use this if you want to keep the original yaw angle

        # env.draw_line([0,0,-5], [10, 10, -5], thickness=5.0, lifetime=0.0)
        # env.spawn_prop(prop_type="box", 
        #                 scale=scale, 
        #                 location=location,
        #                 rotation=rotation,
        #                 material='gold')
        
        # print(f"Spawned box at location={location}, rotation (roll, pitch, yaw)={rotation}")
        
        # Tick the environment to update
        while True:
            env.tick()
