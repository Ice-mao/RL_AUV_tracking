import numpy as np
from auv_control import State
from .base import BasePlanner
import matplotlib.pyplot as plt
import bezier
from metadata import METADATA

class RRT_2d(BasePlanner):
    def __init__(self, num_seconds=10, start=None, end=None, speed=None, obstacles=None, margin=None,
                 fixed_depth=None, bottom_corner=None, size=None, start_time=None):
        # setup goal
        self.start = np.array([0, 0, -5]) if start is None else start
        self.end = np.array([20, 20, -5]) if end is None else end
        self.start = self.start[:2]
        self.end = self.end[:2]
        self.num_seconds = num_seconds
        self.speed = speed
        self.obstacles = obstacles
        self.margin = margin + 1.0  # get enough space to move
        self.fixed_depth = fixed_depth
        self.bottom_corner = bottom_corner
        self.size = size

        # setup RRT
        self.start_time = start_time
        self.count = 0
        self.desire_path_num = 0
        self.step_size = 2
        self.tree = self.start[0:2].reshape(1, -1)
        self.dist = [0]
        self.parent = [0]
        self.finish = [False]
        self.finish_flag = 0  # use for path finish
        # self.Visualization()

        # setup plotter.
        self.figID = 2  # for rrt render
        if METADATA['render']:
            plt.ion()
            self.fig = plt.figure(self.figID)
        # from reset to build path
        # self._run_rrt()


    def reset(self, time, start, end):
        # setup RRT
        self.start = start
        self.end = end
        self.start = self.start[:2]
        self.end = self.end[:2]
        self.start_time = time
        self.tree = self.start[0:2].reshape(1, -1)
        self.desire_path_num = 0
        self.dist = [0]
        self.parent = [0]
        self.finish = [False]
        self.finish_flag = 0  # use for path finish
        self.path = [0]
        return self._run_rrt()

    def _run_rrt(self):
        # Make tree till we have a connecting path
        count = 0
        while np.sum(self.finish) < 10:
            self._add_node()
            count += 1
            if count > 2000:
                print('RRT failed')
                return False


        # find nodes that connect to end_node
        connecting_nodes = np.argwhere(np.array(self.finish) != 0).astype('int')

        # find minimum cost last node
        if len(connecting_nodes) == 1:
            idx = connecting_nodes.item(0)
        else:
            idx = connecting_nodes[np.argmin(np.array(self.dist)[connecting_nodes])].item(0)

        # construct the lowest cost path order
        # from the last node to get the start node
        path = [idx]
        parent = np.inf
        while parent != 0:
            parent = int(self.parent[path[-1]])
            path.append(parent)

        # Get actual path locations
        self.path = self.tree[path[::-1]]
        self.path = np.vstack((self.path, self.end))

        # smooth:
        num_point = len(self.path)
        path = self.path.T
        count = 0
        self.curves = []
        if num_point < 6:
            self.curves.append(bezier.Curve(path, degree=num_point - 1))
        else:
            # get the first curve
            count = 6
            nodes = path[:, :count]
            self.curves.append(bezier.Curve(nodes, degree=5))
            while count < num_point:
                # pick the last node of the last curve
                q0 = path[:, count - 1]
                # create the new helper node
                q1 = q0 + (q0 - path[:, count - 2]) / 2
                tmp = np.stack((q0, q1), axis=-1)
                if num_point - count >= 4:
                    count += 4
                    # pick the new node
                    nodes = path[:, count - 4:count]
                    nodes = np.hstack((tmp, nodes))
                    self.curves.append(bezier.Curve(nodes, degree=5))
                else:
                    tmp_num = num_point - count
                    nodes = path[:, count:num_point]
                    nodes = np.hstack((tmp, nodes))
                    count = num_point
                    self.curves.append(bezier.Curve(nodes, degree=tmp_num + 1))

        # 定义控制点

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
        self.control_points = path
        self.path = np.hstack((point_on_curves, path[:, -1].reshape(-1, 1)))

        # make rot and pos functions
        distance = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        if self.speed is None:
            # self.speed = np.sum(distance) / (self.num_seconds - 3)
            self.speed = np.sum(distance) / (self.num_seconds)
        times = np.cumsum(distance / self.speed)
        times = times + self.start_time
        def rot(t):
            step = np.searchsorted(times, t)

            if step + 1 >= len(self.path):
                return np.zeros(3)
            else:
                t = t - times[step - 1] if step > 0 else t

                p_prev = self.path[step]
                p_next = self.path[step + 1]

                m = self.speed * (p_next - p_prev) / np.linalg.norm(p_next - p_prev)

                yaw = np.arctan2(m[1], m[0])
                # pitch = -np.arctan2(m[2], np.sqrt(m[0] ** 2 + m[1] ** 2)) * 180 / np.pi
                # pitch = np.clip(pitch, -15, 15)

                pitch = 0

                return np.array([0, pitch, yaw])

        self.rot_func = np.vectorize(rot, signature='()->(n)')

        def pos(t):
            step = np.searchsorted(times, t)

            if step + 1 >= len(self.path):
                return self.end
            else:
                t = t - times[step - 1] if step > 0 else t

                p_prev = self.path[step]
                p_next = self.path[step + 1]

                m = self.speed * (p_next - p_prev) / np.linalg.norm(p_next - p_prev)
                return m * t + p_prev

        self.pos_func = np.vectorize(pos, signature='()->(n)')
        return True

    def tick(self, true_state):
        """get the path point replace the time"""
        # if not isinstance(t, float):
        #     raise ValueError("Can't tick with an array")
        # 
        # if t < self.start_time:
        #     return np.array([self.start[:2], 0])
        # pos = self.pos_func(t)
        # yaw_rad = self.rot_func(t)[2]
        # return np.array([pos[0], pos[1], yaw_rad])
        # if arrived, return the next path point
        # if self.finish_flag:
        #     self.reset(true_state)
        dis = np.linalg.norm(self.path[:, self.desire_path_num][: 2] - true_state[:2])
        if dis < 0.3:
            # 到达
            self.count = 0
            self.desire_path_num += 1
            if METADATA['render']:
                print(self.desire_path_num)
            # tmp = self.path[self.desire_path_num] - self.path[self.desire_path_num-1]
            # theta = np.arctan2(tmp[1], tmp[0])
        # elif dis > 2.0:
        #     self.count += 1
        #     if self.count > 50:
        #         # over
        #         self.finish_flag = 1

        if self.desire_path_num == len(self.path.T):
            self.finish_flag = 1
            self.desire_path_num -= 1
            if METADATA['render']:
                print('finish')
            return self.path[:, self.desire_path_num]
        return self.path[:, self.desire_path_num]

    def _add_node(self):
        # make random pose :(get the pose of xy)
        min_xy = np.array([self.bottom_corner[0], self.bottom_corner[1]])
        max_xy = np.array([self.top_corner[0], self.top_corner[1]])
        pose = np.random.uniform(min_xy, max_xy)
        # pose = np.concatenate((pose_xy, [self.fix_depth]), axis=-1)
        # pose = np.random.uniform(self.bottom_corner, self.top_corner)

        # find closest tree element
        dist = np.linalg.norm(self.tree - pose, axis=1)
        close_idx = np.argmin(dist)
        close_pose = self.tree[close_idx]
        close_dist = self.dist[close_idx]

        # Move our sampled pose closer
        direction = (pose - close_pose) / np.linalg.norm(pose - close_pose)
        pose = close_pose + direction * self.step_size

        # Save everything if we don't collide
        if (self.obstacles.check_obstacle_block(close_pose, pose)
                and self.obstacles.check_obstacle_collision(pose, self.margin)):
            self.tree = np.vstack((self.tree, pose))
            self.dist.append(close_dist + self.step_size)
            self.parent.append(close_idx)

            finish = np.linalg.norm(pose - self.end) < self.step_size
            self.finish.append(finish)

    def _collision(self, start, end):
        vals = np.linspace(start, end, 50)
        for v in vals:
            # Check if point is inside of any circle
            dist = np.linalg.norm(self.obstacle_loc - v, axis=1)
            if np.any(dist < self.obstacle_size + 1):
                return True
        return False

    @property
    def center(self):
        return self.bottom_corner + self.size / 2

    @property
    def top_corner(self):
        return self.bottom_corner + self.size

    def draw_traj(self, env, t):
        """Override super class to also make environment appear"""
        for p in self.path.T:
            p = np.append(p, self.fixed_depth)
            env.draw_point(p.tolist(), color=[255, 0, 0], thickness=15, lifetime=50)
        self.fig.clf()
        self.ax = self.fig.subplots()
        self.ax.clear()
        self.ax.plot(self.path[0], self.path[1], label='Bezier Curve')
        self.ax.scatter(self.control_points[0], self.control_points[1], label='ControlPoints')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Bezier Curve')
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_xlim(self.bottom_corner[0], self.top_corner[0])
        self.ax.set_ylim(self.bottom_corner[1], self.top_corner[1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # # Get all positions
        # t = np.arange(0, t, 0.5)
        # des_state = self._traj(t)
        # des_pos = des_state[:, 0:3]
        #
        # # Draw line between each
        # for i in range(len(des_pos) - 1):
        #     env.draw_line(des_pos[i].tolist(), des_pos[i + 1].tolist(), thickness=5.0, lifetime=0.0)

    def Visualization(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.interpolate import splev, splprep

        # 假设已有的路径点
        # self.path.save
        # 进行样条插值拟合
        x = self.path[:, 0]
        y = self.path[:, 1]
        tck, u = splprep([x, y], s=0)
        x1, y1 = splev(u, tck)
        plt.plot(x1, y1, 'r-', x, y, 'bo')
        # # 从拟合的曲线上采样得到路径点
        # num_samples = 100
        # sample_points = np.linspace(0, self.path.shape[0] - 1, num_samples)
        # sampled_path_points = np.column_stack((cs(sample_points), cs(sample_points, 1)))
        #
        # # 绘制贝塞尔曲线和采样得到的路径点
        # plt.figure()
        # plt.plot(x, y, 'o', label='Original Path Points')
        # plt.plot(cs(sample_points), cs(sample_points, 1), '-', label='Cubic Spline')
        # plt.plot(sampled_path_points[:, 0], sampled_path_points[:, 1], 'x', label='Sampled Path Points')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('Bezier Curve Fitting and Sampling')
        # plt.legend()
        # plt.grid(True)
        # plt.show()

# if __name__ == '__main__':
#     return 1