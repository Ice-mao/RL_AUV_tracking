import numpy as np
from auv_control import State
from .base import BasePlanner
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import bezier

class RRT_3d(BasePlanner):
    def __init__(self, num_seconds=10, speed=None, obstacles=None, margin=None,
                 bottom_corner=None, size=None,
                 render=True, draw_flag=True):
        """
        3D RRT路径规划器
        
        Parameters:
        -----------
        num_seconds : float
            路径执行时间
        start : array_like, shape (3,)
            起始点 [x, y, z]
        end : array_like, shape (3,)
            目标点 [x, y, z]
        speed : float
            移动速度
        obstacles : object
            障碍物检测对象，需要有check_obstacle_block和check_obstacle_collision方法
        margin : float
            安全边距
        bottom_corner : array_like, shape (3,)
            搜索空间的底部角点
        size : array_like, shape (3,)
            搜索空间的尺寸
        """
        self.num_seconds = num_seconds
        self.speed = speed
        self.obstacles = obstacles
        self.margin = margin + 1.0 if margin is not None else 2.0  # 足够的安全空间
        self.bottom_corner = bottom_corner if bottom_corner is not None else np.array([-30, -30, -10])
        self.size = size if size is not None else np.array([60, 60, 8])
        self.render = render
        self.draw_flag = draw_flag

        # 设置RRT参数
        self.count = 0
        self.desire_path_num = 0
        self.step_size = 2.0  # 增大步长以提高搜索效率
        self.max_iterations = 5000  # 减少最大迭代次数，更合理
        
        # 初始化RRT树
        self.dist = [0]
        self.parent = [0]
        self.finish = [False]
        self.finish_flag = 0
        self.path = None

        self.figID = 3
        if self.draw_flag:
            plt.ion()
            self.fig = plt.figure(self.figID, figsize=(10, 8))

    def reset(self, time, start, end):
        self.start_time = time
        self.start = np.array(start)
        self.end = np.array(end)
            
        self.tree = self.start.reshape(1, -1)
        self.desire_path_num = 0
        self.dist = [0]
        self.parent = [0]
        self.finish = [False]
        self.finish_flag = 0
        self.path = None
        
        return self._run_rrt()

    def _run_rrt(self):
        """运行RRT算法"""       
        count = 0
        while np.sum(self.finish) < 3:
            self._add_node()
            count += 1
            
            if count > self.max_iterations:
                print(f'3D RRT 规划失败: 超过最大迭代次数 {self.max_iterations}')
                print(f'当前树节点数: {len(self.tree)}')
                print(f'当前到达终点的路径数: {np.sum(self.finish)}')
                return False
            
            # 每1000次迭代输出进度
            if count % 1000 == 0:
                print(f'RRT进度: {count}/{self.max_iterations} 迭代, 树节点数: {len(self.tree)}, 到达终点路径数: {np.sum(self.finish)}')

        # 找到连接到终点的节点
        connecting_nodes = np.argwhere(np.array(self.finish) != 0).astype('int')

        # 找到最小代价的路径
        if len(connecting_nodes) == 1:
            idx = connecting_nodes.item(0)
        else:
            idx = connecting_nodes[np.argmin(np.array(self.dist)[connecting_nodes])].item(0)

        # 构建路径
        path_indices = [idx]
        parent = np.inf
        while parent != 0:
            parent = int(self.parent[path_indices[-1]])
            path_indices.append(parent)

        # 获取实际路径位置
        self.path = self.tree[path_indices[::-1]]
        self.path = np.vstack((self.path, self.end))
        
        # print(f"RRT规划成功! 路径长度: {len(self.path)} 点")
        
        # 使用贝塞尔曲线平滑路径
        self._smooth_path_with_bezier()
        return True

    def _smooth_path_with_bezier(self):
        """使用贝塞尔曲线平滑3D路径"""
        if len(self.path) < 3:
            return
        
        num_point = len(self.path)
        path = self.path.T
        
        count = 0
        self.curves = []
        
        if num_point < 6:
            # 如果点数少于6，创建单一曲线
            self.curves.append(bezier.Curve(path, degree=num_point - 1))
        else:
            # 分段创建贝塞尔曲线
            # 第一段曲线
            count = 6
            nodes = path[:, :count]
            self.curves.append(bezier.Curve(nodes, degree=5))
            
            while count < num_point:
                # 选择上一条曲线的最后一个节点
                q0 = path[:, count - 1]
                # 创建新的辅助节点以保证连续性
                q1 = q0 + (q0 - path[:, count - 2]) / 2
                tmp = np.stack((q0, q1), axis=-1)
                
                if num_point - count >= 4:
                    count += 4
                    # 选择新节点
                    nodes = path[:, count - 4:count]
                    nodes = np.hstack((tmp, nodes))
                    self.curves.append(bezier.Curve(nodes, degree=5))
                else:
                    tmp_num = num_point - count
                    nodes = path[:, count:num_point]
                    nodes = np.hstack((tmp, nodes))
                    count = num_point
                    self.curves.append(bezier.Curve(nodes, degree=tmp_num + 1))

        # 在曲线上采样点
        num_sample_points = 2 * num_point
        num_curve = len(self.curves)
        pick_point = int(num_sample_points / num_curve) + 1
        point_on_curves = None
        
        for curve in self.curves:
            s_vals = np.linspace(0, 1, pick_point)
            point_on_curve = curve.evaluate_multi(s_vals[:-1])
            if point_on_curves is None:
                point_on_curves = point_on_curve
            else:
                point_on_curves = np.hstack((point_on_curves, point_on_curve))
        
        # 保存控制点和更新路径
        self.control_points = path
        self.path = np.hstack((point_on_curves, path[:, -1].reshape(-1, 1)))
        
        # print(f"贝塞尔曲线平滑完成! 平滑后路径点数: {len(self.path[1])}")

    def _add_node(self):
        # 在空间中随机采样
        min_xyz = self.bottom_corner
        max_xyz = self.bottom_corner + self.size
        
        # 30%的概率直接朝向目标点采样（引导搜索）
        if np.random.random() < 0.3:
            pose = self.end + np.random.normal(0, 1.0, 3)  # 在目标点附近添加噪声
            pose = np.clip(pose, min_xyz, max_xyz)  # 确保在搜索空间内
        else:
            pose = np.random.uniform(min_xyz, max_xyz)

        # 找到最近的树节点
        dist = np.linalg.norm(self.tree - pose, axis=1)
        close_idx = np.argmin(dist)
        close_pose = self.tree[close_idx]
        close_dist = self.dist[close_idx]

        # 将采样点移动到步长范围内
        direction = (pose - close_pose) / np.linalg.norm(pose - close_pose)
        pose = close_pose + direction * self.step_size

        # 检查碰撞
        if self._check_collision_3d(close_pose, pose):
            return  # 有碰撞，跳过

        # 添加到树中
        self.tree = np.vstack((self.tree, pose))
        self.dist.append(close_dist + np.linalg.norm(pose - close_pose))
        self.parent.append(close_idx)
        finish = np.linalg.norm(pose - self.end) < self.step_size
        self.finish.append(finish)

    def _check_collision_3d(self, close_pose, pose):
        if self.obstacles is None:
            return False
            
        path_clear = self.obstacles.check_obstacle_block(close_pose, pose)
        start_clear = self.obstacles.check_obstacle_collision(pose, self.margin)
        return not (path_clear and start_clear)

    def tick(self, true_state):
        if self.path is None or len(self.path) == 0:
            return self.end
        dis = np.linalg.norm(self.path[:, self.desire_path_num] - true_state[:3])
        
        # 如果到达当前目标点
        if dis < 0.5:  # 3D中可以稍微宽松一些
            self.count = 0
            self.desire_path_num += 1
            if self.render:
                print(self.desire_path_num)

        # 检查是否完成路径
        if self.desire_path_num == len(self.path.T):
            self.finish_flag = 1
            self.desire_path_num -= 1
            if self.render:
                print('finish')
            return self.path[:, self.desire_path_num]
        return self.path[:, self.desire_path_num]

    @property
    def center(self):
        """获取搜索空间中心"""
        return self.bottom_corner + self.size / 2

    @property
    def top_corner(self):
        """获取搜索空间顶部角点"""
        return self.bottom_corner + self.size

    def draw_traj(self, env, t):
        """绘制3D轨迹"""
        if self.path is None:
            return

        for i, p in enumerate(self.path.T):
            color = [255, 0, 0] if i < self.desire_path_num else [0, 255, 0]
            env.draw_point(p.tolist(), color=color, thickness=10, lifetime=50)
        self.fig.clf()
        ax = self.fig.add_subplot(111, projection='3d')
        
        # 绘制路径
        if len(self.path.T) > 0:
            ax.plot(self.path[0, :], self.path[1, :], self.path[2, :], 
                    'b-', linewidth=2, label='3D Path')
            ax.scatter(self.path[0, :], self.path[1, :], self.path[2, :], 
                        c='red', s=50, label='Path Points')
        
        # 绘制起点和终点
        ax.scatter(*self.start, c='green', s=100, marker='o', label='Start')
        ax.scatter(*self.end, c='red', s=100, marker='s', label='Goal')
        
        # 设置坐标轴
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D RRT Path Planning')
        ax.legend()
        
        # 设置坐标轴范围
        ax.set_xlim(self.bottom_corner[0], self.top_corner[0])
        ax.set_ylim(self.bottom_corner[1], self.top_corner[1])
        ax.set_zlim(self.bottom_corner[2], self.top_corner[2])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


