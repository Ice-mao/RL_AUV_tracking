import numpy as np
from auv_control import State
from .base import BasePlanner


class RRT(BasePlanner):
    def __init__(self, num_seconds, start=None, end=None, speed=None):
        # setup goal
        self.start = np.array([0, 0, -5]) if start is None else start
        self.end = np.array([40, 40, -5]) if end is None else end
        self.num_seconds = num_seconds
        self.speed = speed

        # setup environment
        self.size = np.array([50, 50, 25])
        self.bottom_corner = np.array([-5, -5, -25])

        # setup RRT
        self.step_size = 5
        self.tree = self.start.reshape(1, -1)
        self.dist = [0]
        self.parent = [0]
        self.finish = [False]
        self._run_rrt()

    def reset(self):
        self._run_rrt()

    def _run_rrt(self):
        # Make tree till we have a connecting path
        while np.sum(self.finish) < 20:
            self._add_node()

        # find nodes that connect to end_node
        connecting_nodes = np.argwhere(np.array(self.finish) != 0).astype('int')

        # find minimum cost last node
        if len(connecting_nodes) == 1:
            idx = connecting_nodes.item(0)
        else:
            idx = connecting_nodes[np.argmin(np.array(self.dist)[connecting_nodes])].item(0)

        # construct lowest cost path order
        path = [idx]
        parent = np.inf
        while parent != 0:
            parent = int(self.parent[path[-1]])
            path.append(parent)

        # Get actual path locations
        self.path = self.tree[path[::-1]]
        self.path = np.vstack((self.path, self.end))

        # smooth the waypoint path
        smooth = [0]

        # Go til the last or second to last waypoint is in smooth
        while smooth[-1] < len(self.path) - 1:
            # iterate through all waypoints that are left
            for i in range(smooth[-1] + 1, len(self.path)):
                # Check if there's a collision, if there is add in the previous waypoint
                if self._collision(self.path[smooth[-1]:smooth[-1] + 1], self.path[i:i + 1]):
                    smooth.append(i - 1)
                    break
                # If we're at the last element of our path, add it in so we can be done
                if i == len(self.path) - 1:
                    smooth.append(len(self.path) - 1)
                    break

        # Remove unneeded nodes from our path
        self.path = self.path[smooth]

        # make rot and pos functions
        distance = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        if self.speed is None:
            self.speed = np.sum(distance) / (self.num_seconds - 3)
        times = np.cumsum(distance / self.speed)

        def rot(t):
            step = np.searchsorted(times, t)

            if step + 1 >= len(self.path):
                return np.zeros(3)
            else:
                t = t - times[step - 1] if step > 0 else t

                p_prev = self.path[step]
                p_next = self.path[step + 1]

                m = self.speed * (p_next - p_prev) / np.linalg.norm(p_next - p_prev)

                yaw = np.arctan2(m[1], m[0]) * 180 / np.pi
                pitch = -np.arctan2(m[2], np.sqrt(m[0] ** 2 + m[1] ** 2)) * 180 / np.pi
                pitch = np.clip(pitch, -15, 15)

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

    def _add_node(self):
        # make random pose
        pose = np.random.uniform(self.bottom_corner, self.top_corner)

        # find closest tree element
        dist = np.linalg.norm(self.tree - pose, axis=1)
        close_idx = np.argmin(dist)
        close_pose = self.tree[close_idx]
        close_dist = self.dist[close_idx]

        # Move our sampled pose closer
        direction = (pose - close_pose) / np.linalg.norm(pose - close_pose)
        pose = close_pose + direction * self.step_size

        # Save everything if we don't collide
        if not self._collision(close_pose, pose):
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
        # Setup environment
        env.draw_box(self.center.tolist(), (self.size / 2).tolist(), color=[0, 0, 255], thickness=30, lifetime=0)
        for i in range(self.num_obstacles):
            loc = self.obstacle_loc[i].tolist()
            loc[1] *= -1
            env.spawn_prop('sphere', loc, [0, 0, 0], self.obstacle_size[i], False, "white")

        for p in self.path:
            env.draw_point(p.tolist(), color=[255, 0, 0], thickness=20, lifetime=0)

        super().draw_traj(env, t)
