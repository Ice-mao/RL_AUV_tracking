import math

import numpy as np

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

    # 计算旋转后的偏移量
    new_offset = np.dot(rotation_matrix, offset)

    # 将旋转后的偏移量添加到中心点的坐标上，得到旋转后的点的坐标
    new_point = np.array(center) + new_offset

    return new_point


class Obstacle:
    def __init__(self, env, fix_depth):
        self.env = env
        self.fix_depth = fix_depth
        self.num_obstacles = 4
        self.res = 0.2  # m remeber to * with scale
        self.sub_center = [25*self.res, 25*self.res]  # m subobstacle rotate center
        self.sub_coordinates = [np.array([45, -45]) * self.res, np.array([45, 45]) * self.res,
                                np.array([-45, -45]) * self.res, np.array([-45, 45]) * self.res]  # m
        # self.chosen_idx = np.random.choice(len(obstacles), self.num_obstacles, replace=True)
        self.chosen_idx = np.array([4,5,6,7])
        self.rot_angs = [np.random.choice(np.arange(-10, 10, 1) / 10. * 180) for _ in range(self.num_obstacles)]

    def draw_obstacle(self):
        for i in range(self.num_obstacles):
            obstacle = obstacles[str(self.chosen_idx[i])]  # extract the obstacle to decompose
            for j in range(len(obstacle['center'])):
                obstacle['center'][j] = rotate_point(obstacle['center'][j], self.sub_center, self.rot_angs[i])
                loc = obstacle['center'][j] + np.array(self.sub_coordinates[i])
                loc = np.append(loc, self.fix_depth)
                obstacle['scale'][j][0] *= self.res
                obstacle['scale'][j][1] *= self.res
                obstacle['scale'][j][2] *= 3  # 3m depth
                self.env.spawn_prop(prop_type="box", scale=obstacle['scale'][j], location=loc.tolist(),
                                    rotation=[np.tan(np.radians(self.rot_angs[i])), 1, 0],  # it's annoy to be pitch?
                                    material='gold')
