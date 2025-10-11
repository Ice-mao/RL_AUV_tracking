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
        Initialize 3D obstacles
        
        Parameters:
        -----------
        env : holoocean environment
        depths : list of float, shape (2,)
            Two depth layers, default [-3, -7]
        config : dict
            Configuration parameters
        """
        self.env = env
        self.fix_depths = fix_depth if fix_depth is not None else [-3, -7]
        self.config = config
        self.num_obstacles = 4
        self.res = 0.2  # m
        self.sub_center = [25 * self.res, 25 * self.res]  # m, sub-obstacle rotation center
        self.sub_coordinates = [np.array([20, 25]) * self.res, np.array([-70, 25]) * self.res,
                                np.array([-70, -65]) * self.res, np.array([20, -65]) * self.res]  # m
        np.random.seed()
        self.chosen_idx_layer1 = np.random.choice(len(obstacles), self.num_obstacles, replace=True)
        self.chosen_idx_layer2 = np.random.choice(len(obstacles), self.num_obstacles, replace=True)
        self.rot_angs_layer1 = [np.random.choice(np.arange(-10, 10, 1) / 10. * 180) for _ in range(self.num_obstacles)]
        self.rot_angs_layer2 = [np.random.choice(np.arange(-10, 10, 1) / 10. * 180) for _ in range(self.num_obstacles)]
        
        self.polygons_layer1 = []  # First layer obstacle polygons
        self.polygons_layer2 = []  # Second layer obstacle polygons
        self.obstacle_boxes_3d = []  # 3D bounding boxes for 3D collision detection
        
        if FCL_AVAILABLE:
            self.obstacle_objects_fcl = []  # Store FCL geometry objects
            self.collision_manager_fcl = fcl.DynamicAABBTreeCollisionManager()

    def reset(self):
        """Reset obstacle configuration"""
        np.random.seed()
        if not self.config['eval_fixed']:
            self.chosen_idx_layer1 = np.random.choice(len(obstacles), self.num_obstacles, replace=False)
            self.chosen_idx_layer2 = np.random.choice(len(obstacles), self.num_obstacles, replace=False)
            self.rot_angs_layer1 = [np.random.choice(np.arange(-10, 10, 1) / 10. * 180) for _ in range(self.num_obstacles)]
            self.rot_angs_layer2 = [np.random.choice(np.arange(-10, 10, 1) / 10. * 180) for _ in range(self.num_obstacles)]
        else:
            self.chosen_idx_layer1 = np.array([7, 0, 4, 5])
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
                
                # Calculate 3D obstacle position
                loc_center = rotate_point(obstacle['center'][j], self.sub_center, rot_angs[i])
                loc = loc_center + np.array(self.sub_coordinates[i])
                loc = np.append(loc, depth)
                
                # Calculate 3D obstacle size
                _scale = [obstacle['scale'][j][0] * self.res, 
                         obstacle['scale'][j][1] * self.res,
                         obstacle['scale'][j][2] * 2]
                
                material = 'gold' if layer_idx == 0 else 'wood'
                self.env.spawn_prop(prop_type="box", 
                                    scale=_scale, 
                                    location=loc.tolist(),
                                    rotation=[np.tan(np.radians(rot_angs[i])), 1, 0],
                                    material=material)
                
                # Save 3D bounding box for 3D collision detection
                bbox_3d = {
                    'center': loc,
                    'size': np.array(_scale),
                    'rotation': rot_angs[i],
                    'depth': depth,
                    'layer': layer_idx,
                    'points_2d': points  # Save 2D projection points
                }
                self.obstacle_boxes_3d.append(bbox_3d)
                
                # Create 3D geometry for collision detection
                if FCL_AVAILABLE:
                    # Create box geometry using FCL
                    box_geom = fcl.Box(_scale[0], _scale[1], _scale[2])
                    
                    # Quaternion for rotation around Z-axis, rot_angs[i] is in degrees
                    angle_rad = np.radians(rot_angs[i])
                    quat = np.array([np.cos(angle_rad/2), 0, 0, np.sin(angle_rad/2)])
                    translation = np.array(loc)
                    
                    # Create transformation matrix (pass quaternion and translation directly in constructor)
                    transform = fcl.Transform(quat, translation)
                    
                    # Create collision object
                    collision_object = fcl.CollisionObject(box_geom, transform)
                    self.obstacle_objects_fcl.append(collision_object)
                    
                    # Add to collision manager
                    self.collision_manager_fcl.registerObject(collision_object)

    def check_obstacle_collision(self, point, margin):
        point = np.array(point)
        
        if FCL_AVAILABLE:
            return self._check_collision_with_fcl(point, margin)
    
    def _check_collision_with_fcl(self, point, margin):
        """Use FCL for high-performance 3D collision detection"""
        # Create a cube centered at point with side length 2*margin
        safety_box = fcl.Box(margin, margin, margin)
        
        # Set cube position (no rotation)
        translation = np.array(point)
        transform = fcl.Transform(translation)
        
        # Create query object
        query_object = fcl.CollisionObject(safety_box, transform)
        
        # Create collision request
        request = fcl.CollisionRequest()
        result = fcl.CollisionResult()
        
        # Check collision with all obstacles
        # for obstacle_obj in self.obstacle_objects_fcl:
        #     ret = fcl.collide(query_object, obstacle_obj, request, result)
        #     if ret:
        #         return False
        # or
        # Use collision manager for efficient detection
        req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
        rdata = fcl.CollisionData(request=req)
        self.collision_manager_fcl.collide(query_object, rdata, fcl.defaultCollisionCallback)
        if rdata.result.is_collision is True:
            return False
        return True

    def check_obstacle_block(self, point1, point2, margin=1):
        """
        Check if 3D path is blocked by obstacles
        
        Parameters:
        -----------
        point1 : array_like, shape (3,)
            Start point [x, y, z]
        point2 : array_like, shape (3,)
            End point [x, y, z]
        margin : float
            Safety margin
        """
        point1 = np.array(point1)
        point2 = np.array(point2)
        
        # Sample along path to check for collisions
        num_samples = 10
        for i in range(num_samples + 1):
            t = i / num_samples
            sample_point = point1 + t * (point2 - point1)
            
            # Add some random perturbations to check margins
            for _ in range(5):  # Reduce random checks to improve performance
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                
                # 3D spherical random offset
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

    print("=== 3D Obstacle Testing ===")
    
    # Test 3D obstacles
    with holoocean.make("SimpleUnderwater-Bluerov2") as env:
        # Create 3D obstacles, generate two layers at -3m and -7m depths
        obstacle_3d = Obstacle3D(env, fix_depth=[-3, -7], 
                               config={'eval_fixed': False, 'render': True, 'debug': True})
        obstacle_3d.reset()
        obstacle_3d.draw_obstacle()
        
        # Get obstacle information
        
        # Test collision detection
        print("\n=== Collision Detection Test ===")
        test_points = [
            [0, 0, -3],    # May collide with first layer
            [0, 0, -7],    # May collide with second layer
            [0, 0, -5],    # Between two layers
            [100, 100, -5] # Far from obstacles
        ]
        
        for point in test_points:
            safe = obstacle_3d.check_obstacle_collision(point, 1.0)
            print(f"Point {point}: {'Safe' if safe else 'Collision'}")
        
        # Test path blocking detection
        print("\n=== Path Blocking Test ===")
        path_tests = [
            ([0, 0, -3], [15, 15, -3]),    # Path within first layer
            ([0, 0, -7], [15, 15, -7]),    # Path within second layer
            ([0, 0, -3], [0, 0, -7]),      # Vertical path
            ([100, 100, -3], [100, 100, -7])  # Path far from obstacles
        ]
        env.draw_line([0, 0, -3], [15, 15, -3], thickness=5.0, lifetime=0.0)
        env.draw_line([0, 0, -7], [15, 15, -7], thickness=5.0, lifetime=0.0)
        env.draw_line([0, 0, -3], [0, 0, -7], thickness=5.0, lifetime=0.0)
        for start, end in path_tests:
            clear = obstacle_3d.check_obstacle_block(start, end, 1.0)
            print(f"Path {start} -> {end}: {'Clear' if clear else 'Blocked'}")
        
        print("\nRunning environment continuously, press Ctrl+C to exit...")
        try:
            while True:
                env.tick()
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Test ended")
