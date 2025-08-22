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

import fcl
FCL_AVAILABLE = True

from auv_env.envs.obstacle import obstacles, rotate_point, polygon_2_points

class Obstacle3D:
    
    def __init__(self, env, fix_depth, config):
        """
        初始化3D障碍物
        
        Parameters:
        -----------
        env : holoocean环境
        depths : list of float, shape (2,)
            两个深度层，默认为[-3, -7]
        config : dict
            配置参数
        """
        self.env = env
        self.fix_depths = fix_depth if fix_depth is not None else [-3, -7]
        self.config = config
        self.num_obstacles = 4
        self.res = 0.2  # m
        self.sub_center = [25 * self.res, 25 * self.res]  # m，子障碍物旋转中心
        self.sub_coordinates = [np.array([20, 25]) * self.res, np.array([-70, 25]) * self.res,
                                np.array([-70, -65]) * self.res, np.array([20, -65]) * self.res]  # m
        np.random.seed()
        self.chosen_idx_layer1 = np.random.choice(len(obstacles), self.num_obstacles, replace=True)
        self.chosen_idx_layer2 = np.random.choice(len(obstacles), self.num_obstacles, replace=True)
        self.rot_angs_layer1 = [np.random.choice(np.arange(-10, 10, 1) / 10. * 180) for _ in range(self.num_obstacles)]
        self.rot_angs_layer2 = [np.random.choice(np.arange(-10, 10, 1) / 10. * 180) for _ in range(self.num_obstacles)]
        
        self.polygons_layer1 = []  # 第一层障碍物多边形
        self.polygons_layer2 = []  # 第二层障碍物多边形
        self.obstacle_boxes_3d = []  # 3D边界框，用于3D碰撞检测
        
        if FCL_AVAILABLE:
            self.obstacle_objects_fcl = []  # 存储FCL几何对象
            self.collision_manager_fcl = fcl.DynamicAABBTreeCollisionManager()

    def reset(self):
        """重置障碍物配置"""
        np.random.seed()
        if not self.config['eval_fixed']:
            self.chosen_idx_layer1 = np.random.choice(len(obstacles), self.num_obstacles, replace=False)
            self.chosen_idx_layer2 = np.random.choice(len(obstacles), self.num_obstacles, replace=False)
            self.rot_angs_layer1 = [np.random.choice(np.arange(-10, 10, 1) / 10. * 180) for _ in range(self.num_obstacles)]
            self.rot_angs_layer2 = [np.random.choice(np.arange(-10, 10, 1) / 10. * 180) for _ in range(self.num_obstacles)]
        else:
            self.chosen_idx_layer1 = np.array([4, 5, 0, 7])
            self.chosen_idx_layer2 = np.array([1, 2, 3, 6])
            self.rot_angs_layer1 = np.array([36.0, -72.0, -144.0, 0.0])
            self.rot_angs_layer2 = np.array([45.0, -90.0, 0.0, 90.0])
        
        self.polygons_layer1 = []
        self.polygons_layer2 = []
        self.obstacle_boxes_3d = []
        if FCL_AVAILABLE:
            self.obstacle_objects_fcl = []
            self.collision_manager_fcl = fcl.DynamicAABBTreeCollisionManager()

    def draw_obstacle(self):
        self._draw_layer(0, self.chosen_idx_layer1, self.rot_angs_layer1, 
                        self.polygons_layer1, self.fix_depths[0])
        self._draw_layer(1, self.chosen_idx_layer2, self.rot_angs_layer2, 
                        self.polygons_layer2, self.fix_depths[1])

    def _draw_layer(self, layer_idx, chosen_idx, rot_angs, polygons_list, depth):
        for i in range(self.num_obstacles):
            obstacle = obstacles[str(chosen_idx[i])]
            for j in range(len(obstacle['center'])):
                points = polygon_2_points(obstacle['center'][j], obstacle['scale'][j], self.res, 
                                        self.sub_center, rot_angs[i], self.sub_coordinates[i])
                if self.config['debug']:
                    self.env.draw_line([points[0][0], points[0][1], depth],
                                       [points[1][0], points[1][1], depth],
                                       thickness=5.0, lifetime=0.0)
                    self.env.draw_line([points[1][0], points[1][1], depth],
                                       [points[2][0], points[2][1], depth],
                                       thickness=5.0, lifetime=0.0)
                    self.env.draw_line([points[2][0], points[2][1], depth],
                                       [points[3][0], points[3][1], depth],
                                       thickness=5.0, lifetime=0.0)
                    self.env.draw_line([points[3][0], points[3][1], depth],
                                       [points[0][0], points[0][1], depth],
                                       thickness=5.0, lifetime=0.0)
                
                polygons_list.append(Polygon(points))
                
                # 计算3D障碍物位置
                loc_center = rotate_point(obstacle['center'][j], self.sub_center, rot_angs[i])
                loc = loc_center + np.array(self.sub_coordinates[i])
                loc = np.append(loc, depth)
                
                # 计算3D障碍物尺寸
                _scale = [obstacle['scale'][j][0] * self.res, 
                         obstacle['scale'][j][1] * self.res,
                         obstacle['scale'][j][2] * 2]
                
                material = 'gold' if layer_idx == 0 else 'wood'
                self.env.spawn_prop(prop_type="box", 
                                    scale=_scale, 
                                    location=loc.tolist(),
                                    rotation=[np.tan(np.radians(rot_angs[i])), 1, 0],
                                    material=material)
                
                # 保存3D边界框用于3D碰撞检测
                bbox_3d = {
                    'center': loc,
                    'size': np.array(_scale),
                    'rotation': rot_angs[i],
                    'depth': depth,
                    'layer': layer_idx,
                    'points_2d': points  # 保存2D投影点
                }
                self.obstacle_boxes_3d.append(bbox_3d)
                
                # 创建3D几何体用于碰撞检测
                if FCL_AVAILABLE:
                    # 使用FCL创建盒子几何体
                    box_geom = fcl.Box(_scale[0], _scale[1], _scale[2])
                    
                    # 绕Z轴旋转的四元数，rot_angs[i]是度数
                    angle_rad = np.radians(rot_angs[i])
                    quat = np.array([np.cos(angle_rad/2), 0, 0, np.sin(angle_rad/2)])
                    translation = np.array(loc)
                    
                    # 创建变换矩阵（直接在构造函数中传入四元数和平移）
                    transform = fcl.Transform(quat, translation)
                    
                    # 创建碰撞对象
                    collision_object = fcl.CollisionObject(box_geom, transform)
                    self.obstacle_objects_fcl.append(collision_object)
                    
                    # 添加到碰撞管理器
                    self.collision_manager_fcl.registerObject(collision_object)

    def check_obstacle_collision(self, point, margin):
        point = np.array(point)
        
        if FCL_AVAILABLE:
            return self._check_collision_with_fcl(point, margin)
    
    def _check_collision_with_fcl(self, point, margin):
        """使用FCL进行高性能3D碰撞检测"""
        # 创建以point为中心、边长为2*margin的立方体
        safety_box = fcl.Box(margin, margin, margin)
        
        # 设置立方体的位置（无旋转）
        translation = np.array(point)
        transform = fcl.Transform(translation)
        
        # 创建查询对象
        query_object = fcl.CollisionObject(safety_box, transform)
        
        # 创建碰撞请求
        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        
        # 检查与所有障碍物的碰撞
        # for obstacle_obj in self.obstacle_objects_fcl:
        #     ret = fcl.collide(query_object, obstacle_obj, request, result)
        #     if ret:
        #         return False
        # or
        # 使用碰撞管理器进行高效检测
        req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
        rdata = fcl.CollisionData(request=req)
        self.collision_manager_fcl.collide(query_object, rdata, fcl.defaultCollisionCallback)
        if rdata.result.is_collision is True:
            return False
        return True

    def check_obstacle_block(self, point1, point2, margin=1):
        """
        检查3D路径是否被障碍物阻挡
        
        Parameters:
        -----------
        point1 : array_like, shape (3,)
            起始点 [x, y, z]
        point2 : array_like, shape (3,)
            结束点 [x, y, z]
        margin : float
            安全边距
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        
        # 沿路径采样检查碰撞
        num_samples = 10
        for i in range(num_samples + 1):
            t = i / num_samples
            sample_point = point1 + t * (point2 - point1)
            
            # 添加一些随机扰动来检查边距
            for _ in range(5):  # 减少随机检查次数以提高性能
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                
                # 3D球面随机偏移
                offset = margin * np.array([
                    np.sin(phi) * np.cos(theta),
                    np.sin(phi) * np.sin(theta),
                    np.cos(phi)
                ])
                
                test_point = sample_point + offset
                
                if not self.check_obstacle_collision(test_point, 0):
                    return False
        
        return True

if __name__ == "__main__":
    import holoocean
    import numpy as np
    import time

    print("=== 3D障碍物测试 ===")
    
    # 测试3D障碍物
    with holoocean.make("SimpleUnderwater-Bluerov2") as env:
        # 创建3D障碍物，在-3m和-7m深度生成两层
        obstacle_3d = Obstacle3D(env, fix_depth=[-3, -7], 
                               config={'eval_fixed': False, 'render': True, 'debug': True})
        obstacle_3d.reset()
        obstacle_3d.draw_obstacle()
        
        # 获取障碍物信息
        
        # 测试碰撞检测
        print("\n=== 碰撞检测测试 ===")
        test_points = [
            [0, 0, -3],    # 可能与第一层碰撞
            [0, 0, -7],    # 可能与第二层碰撞
            [0, 0, -5],    # 两层之间
            [100, 100, -5] # 远离障碍物
        ]
        
        for point in test_points:
            safe = obstacle_3d.check_obstacle_collision(point, 1.0)
            print(f"点 {point}: {'安全' if safe else '碰撞'}")
        
        # 测试路径阻挡检测
        print("\n=== 路径阻挡测试 ===")
        path_tests = [
            ([0, 0, -3], [15, 15, -3]),    # 第一层内路径
            ([0, 0, -7], [15, 15, -7]),    # 第二层内路径
            ([0, 0, -3], [0, 0, -7]),      # 垂直路径
            ([100, 100, -3], [100, 100, -7])  # 远离障碍物的路径
        ]
        env.draw_line([0, 0, -3], [15, 15, -3], thickness=5.0, lifetime=0.0)
        env.draw_line([0, 0, -7], [15, 15, -7], thickness=5.0, lifetime=0.0)
        env.draw_line([0, 0, -3], [0, 0, -7], thickness=5.0, lifetime=0.0)
        for start, end in path_tests:
            clear = obstacle_3d.check_obstacle_block(start, end, 1.0)
            print(f"路径 {start} -> {end}: {'畅通' if clear else '被阻挡'}")
        
        print("\n持续运行环境，按Ctrl+C退出...")
        try:
            while True:
                env.tick()
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("测试结束")
