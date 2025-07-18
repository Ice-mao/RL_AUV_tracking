from gymnasium import Wrapper
import numpy as np
from numpy import linalg as LA
import os

import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import animation
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d import Axes3D


class Display2D(Wrapper):
    def __init__(self, env, figID=0, skip=1, confidence=0.95):
        super(Display2D, self).__init__(env)
        self.figID = figID  # figID = 0 : train, figID = 1 : test
        self.env_core = env.env.world
        self.config = self.env_core.config
        self.mapmin = self.env_core.bottom_corner[:2]
        self.mapmax = self.env_core.top_corner[:2]
        self.size = self.env_core.size
        self.fig = plt.figure(self.figID)
        self.n_frames = 0
        self.skip = skip
        self.c_cf = np.sqrt(-2 * np.log(1 - confidence))
        self.traj_num = 0

    def close(self):
        plt.close(self.fig)

    def step(self, action):
        # get the position of agent and targets
        if hasattr(self.env_core, 'targets') and isinstance(self.env_core.targets, list):
            target_true_pos = [self.env_core.targets[i].state.vec[:2] for i in range(len(self.env_core.targets))]
        else:
            # raise ValueError('targets not a list')
            target_true_pos = []


        self.traj[0].append(self.env_core.agent.state.vec[0])
        self.traj[1].append(self.env_core.agent.state.vec[1])
        for i in range(len(self.env_core.targets)):
            self.traj_y[i][0].append(target_true_pos[i][0])
            self.traj_y[i][1].append(target_true_pos[i][1])
        # 如果需要调试，可视化训练进度，则可以进行render
        self.render()
        # self.render_test()
        return self.env.step(action)

    def render(self, record=False, batch_outputs=None):
        state = self.env_core.agent.state.vec
        est_state = self.env_core.agent.est_state.vec
        num_targets = len(self.env_core.targets) if hasattr(self.env_core, 'targets') else 0
        target_true_pos = [self.env_core.targets[i].state.vec[:2] for i in range(num_targets)]
        target_b_state = [self.env_core.belief_targets[i].state for i in range(num_targets)]
        target_cov = [self.env_core.belief_targets[i].cov for i in range(num_targets)]

        if self.n_frames % self.skip == 0:
            self.fig.clf()
            ax = self.fig.subplots()
            im = None

            # show the obstacles and background
            background_rect = patches.Rectangle((self.mapmin[0]-1, self.mapmin[1]-1), self.size[0]+2, self.size[1]+2,
                                                edgecolor='none', facecolor='gray', alpha=0.5)
            ax.add_patch(background_rect)
            if hasattr(self.env_core, 'obstacles'):
                for polygon in self.env_core.obstacles.polygons:
                    x, y = polygon.exterior.xy
                    polygon_patch = Polygon(np.column_stack((x, y)), closed=True, edgecolor='black', facecolor='black')
                    ax.add_patch(polygon_patch)

            # show the target's coordinate
            if hasattr(self.env_core, 'targets'):
                for i in range(num_targets):
                    ax.plot(self.traj_y[i][0], self.traj_y[i][1], 'r.', markersize=2)
                    # Belief on target - Assuming that the first and the second dimension
                    # of the target state vector correspond to xy-coordinate.
                    eig_val, eig_vec = LA.eig(target_cov[i][:2, :2])
                    belief_target = patches.Ellipse(
                        (target_b_state[i][0], target_b_state[i][1]),
                        2 * np.sqrt(eig_val[0]) * self.c_cf,
                        2 * np.sqrt(eig_val[1]) * self.c_cf,
                        angle=180 / np.pi * np.arctan2(np.real(eig_vec[0][1]),
                                                       np.real(eig_vec[0][0])), fill=True, zorder=2,
                        facecolor='g', alpha=0.5)
                    ax.add_patch(belief_target)

                    # if target_cov[i].shape[0] == 4:  # For Velocity
                    #     eig_val, eig_vec = LA.eig(target_cov[i][2:, 2:])
                    #     belief_target_vel = patches.Ellipse(
                    #         (target_b_state[i][0], target_b_state[i][1]),
                    #         2 * np.sqrt(eig_val[0]) * self.c_cf,
                    #         2 * np.sqrt(eig_val[1]) * self.c_cf,
                    #         angle=180 / np.pi * np.arctan2(np.real(eig_vec[0][1]),
                    #                                        np.real(eig_vec[0][0])), fill=True, zorder=2,
                    #         facecolor='m', alpha=0.5)
                    #     ax.add_patch(belief_target_vel)

                    ax.plot(target_b_state[i][0], target_b_state[i][1], marker='o',
                            markersize=10, linewidth=5, markerfacecolor='none',
                            markeredgecolor='g')

                    # The real targets
                    ax.plot(target_true_pos[i][0], target_true_pos[i][1], marker='o',
                            markersize=5, linestyle='None', markerfacecolor='r',
                            markeredgecolor='r')

            # show the agent's coordinate
            ax.plot(state[0], state[1], marker=(4, 0, state[8]),
                    markersize=10, linestyle='None', markerfacecolor='b',
                    markeredgecolor='b')
            ax.plot(est_state[0], est_state[1], marker=(4, 0, state[8]),
                    markersize=10, linestyle='None', markerfacecolor='b',
                    markeredgecolor='b', alpha=0.2)
            # ax.plot(self.traj[0], self.traj[1], 'b.', markersize=2)

            # show the agent's orientation
            if 'agent' in self.config and 'sensor_r' in self.config['agent']:
                sensor_arc = patches.Arc((state[0], state[1]), self.config['agent']['sensor_r'] * 2, self.config['agent']['sensor_r'] * 2,
                                         angle=state[8], theta1=-self.config['agent']['fov'] / 2,
                                         theta2=self.config['agent']['fov'] / 2, edgecolor='black', facecolor='green'
                                         , alpha=0.7)
                ax.add_patch(sensor_arc)
                ax.plot(
                    [state[0], state[0] + self.config['agent']['sensor_r'] * np.cos(np.radians(state[8] + 0.5 * self.config['agent']['fov']))],
                    [state[1], state[1] + self.config['agent']['sensor_r'] * np.sin(np.radians(state[8] + 0.5 * self.config['agent']['fov']))],
                    'k', linewidth=0.5)
                ax.plot(
                    [state[0], state[0] + self.config['agent']['sensor_r'] * np.cos(np.radians(state[8] - 0.5 * self.config['agent']['fov']))],
                    [state[1], state[1] + self.config['agent']['sensor_r'] * np.sin(np.radians(state[8] - 0.5 * self.config['agent']['fov']))],
                    'k', linewidth=0.5)

            # ax.text(self.mapmax[0]+1., self.mapmax[1]-5., 'v_target:%.2f'%np.sqrt(np.sum(self.env_core.targets[0].state[2:]**2)))
            # ax.text(self.mapmax[0]+1., self.mapmax[1]-10., 'v_agent:%.2f'%self.env_core.agent.vw[0])
            ax.set_xlim((self.mapmin[0] - 0.5, self.mapmax[0] + 0.5))
            ax.set_ylim((self.mapmin[1] - 0.5, self.mapmax[1] + 0.5))
            ax.set_title("Eval for the reset")
            ax.set_aspect('equal', 'box')
            ax.grid()

            if not record:
                plt.draw()
                plt.pause(0.0001)

        self.n_frames += 1

    def render_test(self, record=False, batch_outputs=None):
        state = self.env_core.agent.state.vec
        est_state = self.env_core.agent.est_state.vec
        num_targets = len(self.env_core.targets) if hasattr(self.env_core, 'targets') else 0
        target_true_pos = [self.env_core.targets[i].state.vec[:2] for i in range(num_targets)]
        target_b_state = [self.env_core.belief_targets[i].state for i in range(num_targets)]
        target_cov = [self.env_core.belief_targets[i].cov for i in range(num_targets)]

        if self.n_frames % self.skip == 0:
            self.fig.clf()
            ax = self.fig.subplots()
            im = None

            # show the obstacles and background
            background_rect = patches.Rectangle((self.mapmin[0] - 1, self.mapmin[1] - 1), self.size[0] + 2,
                                                self.size[1] + 2,
                                                edgecolor='none', facecolor='gray', alpha=0.5)
            ax.add_patch(background_rect)
            if hasattr(self.env_core, 'obstacles'):
                for polygon in self.env_core.obstacles.polygons:
                    x, y = polygon.exterior.xy
                    polygon_patch = Polygon(np.column_stack((x, y)), closed=True, edgecolor='black', facecolor='black')
                    ax.add_patch(polygon_patch)

            # show the target's coordinate
            if hasattr(self.env_core, 'targets'):
                for i in range(num_targets):
                    ax.plot(self.traj_y[i][0], self.traj_y[i][1], 'r.', markersize=2)
                    # Belief on target - Assuming that the first and the second dimension
                    # of the target state vector correspond to xy-coordinate.
                    # eig_val, eig_vec = LA.eig(target_cov[i][:2, :2])
                    # belief_target = patches.Ellipse(
                    #     (target_b_state[i][0], target_b_state[i][1]),
                    #     2 * np.sqrt(eig_val[0]) * self.c_cf,
                    #     2 * np.sqrt(eig_val[1]) * self.c_cf,
                    #     angle=180 / np.pi * np.arctan2(np.real(eig_vec[0][1]),
                    #                                    np.real(eig_vec[0][0])), fill=True, zorder=2,
                    #     facecolor='g', alpha=0.5)
                    # ax.add_patch(belief_target)

                    # if target_cov[i].shape[0] == 4:  # For Velocity
                    #     eig_val, eig_vec = LA.eig(target_cov[i][2:, 2:])
                    #     belief_target_vel = patches.Ellipse(
                    #         (target_b_state[i][0], target_b_state[i][1]),
                    #         2 * np.sqrt(eig_val[0]) * self.c_cf,
                    #         2 * np.sqrt(eig_val[1]) * self.c_cf,
                    #         angle=180 / np.pi * np.arctan2(np.real(eig_vec[0][1]),
                    #                                        np.real(eig_vec[0][0])), fill=True, zorder=2,
                    #         facecolor='m', alpha=0.5)
                    #     ax.add_patch(belief_target_vel)

                    # ax.plot(target_b_state[i][0], target_b_state[i][1], marker='o',
                    #         markersize=10, linewidth=5, markerfacecolor='none',
                    #         markeredgecolor='g')

                    # The real targets
                    ax.plot(target_true_pos[i][0], target_true_pos[i][1], marker='o',
                            markersize=5, linestyle='None', markerfacecolor='r',
                            markeredgecolor='r')

            # ax.text(self.mapmax[0]+1., self.mapmax[1]-5., 'v_target:%.2f'%np.sqrt(np.sum(self.env_core.targets[0].state[2:]**2)))
            # ax.text(self.mapmax[0]+1., self.mapmax[1]-10., 'v_agent:%.2f'%self.env_core.agent.vw[0])
            ax.set_xlim((self.mapmin[0] - 0.5, self.mapmax[0] + 0.5))
            ax.set_ylim((self.mapmin[1] - 0.5, self.mapmax[1] + 0.5))
            ax.set_title("Eval for the reset")
            ax.set_aspect('equal', 'box')
            ax.grid()

            if not record:
                plt.draw()
                plt.pause(0.0001)

        self.n_frames += 1

    def reset(self, **kwargs):
        self.traj_num += 1
        self.traj = [[], []]
        if hasattr(self.env_core, 'num_targets'):
            self.traj_y = [[[], []]] * self.env_core.num_targets
        else:
            self.traj_y = []
        return self.env.reset(**kwargs)


class Video2D(Wrapper):
    def __init__(self, env, dirname='', skip=1, dpi=80, local_view=0):
        super(Video2D, self).__init__(env)
        self.local_view = local_view
        self.skip = skip
        self.moviewriter = animation.FFMpegWriter()
        fnum = np.random.randint(0, 1000)
        fname = os.path.join(dirname, 'train_%d.mp4' % fnum)
        self.moviewriter.setup(fig=env.fig, outfile=fname, dpi=dpi)
        if self.local_view:
            self.moviewriter0 = animation.FFMpegWriter()
            self.moviewriter0.setup(fig=env.fig0,
                                    outfile=os.path.join(dirname, 'train_%d_local.mp4' % fnum),
                                    dpi=dpi)
        self.n_frames = 0

    def render(self, *args, **kwargs):
        if self.n_frames % self.skip == 0:
            # if traj_num % self.skip == 0:
            self.env.render(record=True, *args, **kwargs)
        self.moviewriter.grab_frame()
        if self.local_view:
            self.moviewriter0.grab_frame()
        self.n_frames += 1

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def finish(self):
        self.moviewriter.finish()
        if self.local_view:
            self.moviewriter0.finish()


class Display3D(Wrapper):
    """
    3D AUV tracking environment visualization wrapper
    Extends Display2D to support 3D visualization with multiple viewports
    """
    def __init__(self, env, figID=0, skip=1, confidence=0.95):
        super(Display3D, self).__init__(env)
        self.figID = figID  # figID = 0 : train, figID = 1 : test
        self.env_core = env.env.world
        self.config = self.env_core.config
        
        # 3D environment boundaries
        self.mapmin = self.env_core.bottom_corner  # [x, y, z]
        self.mapmax = self.env_core.top_corner    # [x, y, z]
        self.size = self.env_core.size
        
        # Create figure for 3D visualization
        self.fig = plt.figure(self.figID, figsize=(12, 8))
        self.n_frames = 0
        self.skip = skip
        self.c_cf = np.sqrt(-2 * np.log(1 - confidence))
        self.traj_num = 0

    def close(self):
        plt.close(self.fig)

    def step(self, action):
        # get the position of agent and targets (3D)
        target_true_pos = [self.env_core.targets[i].state.vec[:3] for i in range(len(self.env_core.targets))]

        # Record trajectories in 3D
        self.traj[0].append(self.env_core.agent.state.vec[0])  # x
        self.traj[1].append(self.env_core.agent.state.vec[1])  # y
        self.traj[2].append(self.env_core.agent.state.vec[2])  # z
        
        for i in range(len(self.env_core.targets)):
            self.traj_y[i][0].append(target_true_pos[i][0])  # x
            self.traj_y[i][1].append(target_true_pos[i][1])  # y
            self.traj_y[i][2].append(target_true_pos[i][2])  # z
            
        # Render 3D visualization
        self.render()
        return self.env.step(action)

    def render(self, record=False, batch_outputs=None):
        state = self.env_core.agent.state.vec
        est_state = self.env_core.agent.est_state.vec
        num_targets = len(self.env_core.targets) if hasattr(self.env_core, 'targets') else 0
        target_true_pos = [self.env_core.targets[i].state.vec[:3] for i in range(num_targets)]
        target_b_state = [self.env_core.belief_targets[i].state for i in range(num_targets)]
        target_cov = [self.env_core.belief_targets[i].cov for i in range(num_targets)]

        if self.n_frames % self.skip == 0:
            self.fig.clf()
            
            # Create single 3D view
            ax_3d = self.fig.add_subplot(111, projection='3d')

            # ========== 3D View ==========
            self._render_3d_view(ax_3d, state, est_state, target_true_pos, target_b_state, target_cov, num_targets)
            
            if not record:
                plt.draw()
                plt.pause(0.01)

        self.n_frames += 1

    def _render_3d_view(self, ax, state, est_state, target_true_pos, target_b_state, target_cov, num_targets):
        """Render the main 3D perspective view"""
        ax.set_title("3D AUV Tracking Environment")
        
        # Draw obstacles if available
        if hasattr(self.env_core, 'obstacles'):
            self._draw_3d_obstacles(ax)
        
        # Draw targets
        if hasattr(self.env_core, 'targets'):
            for i in range(num_targets):
                # Target trajectory (红色轨迹点)
                if len(self.traj_y) > i and len(self.traj_y[i][0]) > 0:
                    ax.plot(self.traj_y[i][0], self.traj_y[i][1], self.traj_y[i][2], 
                           'r.', markersize=2, alpha=0.6, label='Target Trajectory' if i == 0 else "")
                
                # Target belief (projected as sphere)
                if len(target_cov) > i and target_cov[i].shape[0] >= 3:
                    self._draw_belief_ellipsoid(ax, target_b_state[i][:3], target_cov[i][:3, :3])
                
                # Current target position (红色圆点)
                ax.scatter(target_true_pos[i][0], target_true_pos[i][1], target_true_pos[i][2], 
                          c='red', s=80, marker='o', label=f'Target {i}' if i == 0 else "", edgecolors='darkred')
                
                # Target belief center (绿色方块)
                ax.scatter(target_b_state[i][0], target_b_state[i][1], target_b_state[i][2], 
                          c='green', s=100, marker='s', alpha=0.7, 
                          label='Target Belief' if i == 0 else "", edgecolors='darkgreen')

        # Draw agent
        # Agent trajectory (蓝色轨迹点)
        if len(self.traj[0]) > 1:
            ax.plot(self.traj[0], self.traj[1], self.traj[2], 'b-', linewidth=2, alpha=0.7, label='AUV Path')
        
        # Current agent position (蓝色三角形)
        ax.scatter(state[0], state[1], state[2], c='blue', s=150, marker='^', 
                  label='AUV', edgecolors='darkblue', linewidth=2)
        
        # Agent estimated position (青色三角形)
        ax.scatter(est_state[0], est_state[1], est_state[2], c='cyan', s=120, marker='^', 
                  alpha=0.6, label='AUV Estimate', edgecolors='darkcyan')
        
        # Draw agent orientation and sensor FOV (绿色视场角锥体)
        self._draw_3d_sensor_fov(ax, state)
        
        # Set 3D view limits
        ax.set_xlim(self.mapmin[0] - 1, self.mapmax[0] + 1)
        ax.set_ylim(self.mapmin[1] - 1, self.mapmax[1] + 1)
        ax.set_zlim(self.mapmin[2] - 1, self.mapmax[2] + 1)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z - Depth (meters)')
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        # Enable mouse interaction
        ax.mouse_init()

    def _draw_3d_obstacles(self, ax):
        """Draw 3D obstacles as simple rectangular blocks"""
        for i, layer in enumerate([self.env_core.obstacles.polygons_layer1, self.env_core.obstacles.polygons_layer2]):
            for polygon in layer:
                # Get polygon bounding box (assuming rectangular obstacles)
                x_coords, y_coords = polygon.exterior.xy
                x_coords = list(x_coords[:-1])  # Remove duplicate last point
                y_coords = list(y_coords[:-1])
                               
                # Define z-coordinates for obstacles
                z_min = self.env_core.obstacles.fix_depths[i] - 1
                z_max = self.env_core.obstacles.fix_depths[i] + 1

                # Draw a simple rectangular box using plot3D lines
                # Define the 8 vertices of the box
                vertices = [
                    [x_coords[0], y_coords[0], z_min], [x_coords[1], y_coords[1], z_min],
                    [x_coords[2], y_coords[2], z_min], [x_coords[3], y_coords[3], z_min],
                    [x_coords[0], y_coords[0], z_max], [x_coords[1], y_coords[1], z_max], 
                    [x_coords[2], y_coords[2], z_max], [x_coords[3], y_coords[3], z_max]
                ]
                
                # Draw the 12 edges of the box
                edges = [
                    [0, 1], [1, 2], [2, 3], [3, 0],  # bottom face edges
                    [4, 5], [5, 6], [6, 7], [7, 4],  # top face edges
                    [0, 4], [1, 5], [2, 6], [3, 7]   # vertical edges
                ]
                
                for edge in edges:
                    start, end = vertices[edge[0]], vertices[edge[1]]
                    ax.plot3D([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                             'k-', linewidth=2, alpha=0.8)

    def _draw_belief_ellipsoid(self, ax, center, cov_matrix):
        """Draw 3D belief ellipsoid (simplified as sphere)"""
        # Simplified: draw as sphere with radius based on trace of covariance
        radius = np.sqrt(np.trace(cov_matrix)) * self.c_cf / 3
        
        # Create sphere
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        
        ax.plot_surface(x, y, z, alpha=0.3, color='green')

    def _draw_3d_sensor_fov(self, ax, state):
        """Draw 3D sensor field of view as a simple filled cone with edge lines"""
        if 'agent' in self.config and 'sensor_r' in self.config['agent']:
            sensor_range = self.config['agent']['sensor_r']
            h_fov = self.config['agent']['fov'] * np.pi / 180  # horizontal FOV
            v_fov = self.config['agent'].get('h_fov', 60) * np.pi / 180  # vertical FOV
            
            # Agent position and orientation
            pos = state[:3]
            yaw = np.radians(state[8])
            
            # Create simple cone surface
            n_theta = 16  # angular resolution
            n_r = 5       # radial resolution
            
            # Create cone vertices
            vertices = [pos]  # cone tip at agent position
            
            # Generate cone base points in a circle
            for i in range(n_theta):
                angle = 2 * np.pi * i / n_theta
                
                # Calculate direction for this angle within FOV
                h_angle = h_fov * 0.7 * np.cos(angle)  # scale to fit within FOV
                v_angle = v_fov * 0.7 * np.sin(angle)  # scale to fit within FOV
                
                direction = np.array([
                    np.cos(yaw + h_angle) * np.cos(v_angle),
                    np.sin(yaw + h_angle) * np.cos(v_angle),
                    np.sin(v_angle)
                ])
                
                end_point = pos + sensor_range * direction
                vertices.append(end_point)
            
            # Draw cone surface using triangles
            for i in range(n_theta):
                next_i = (i + 1) % n_theta
                
                # Create triangle from tip to two adjacent base points
                triangle_x = [pos[0], vertices[i+1][0], vertices[next_i+1][0]]
                triangle_y = [pos[1], vertices[i+1][1], vertices[next_i+1][1]]
                triangle_z = [pos[2], vertices[i+1][2], vertices[next_i+1][2]]
                
                ax.plot_trisurf(triangle_x, triangle_y, triangle_z, 
                               color='green', alpha=0.2, shade=True)
            
            # Draw four edge lines (generatrices) from tip to base points
            # Choose 4 evenly spaced points on the base circle for edge lines
            edge_indices = [0, n_theta//4, n_theta//2, 3*n_theta//4]
            for idx in edge_indices:
                if idx < len(vertices) - 1:  # Make sure we don't go out of bounds
                    edge_point = vertices[idx + 1]  # +1 because vertices[0] is the tip
                    ax.plot3D([pos[0], edge_point[0]], 
                             [pos[1], edge_point[1]], 
                             [pos[2], edge_point[2]], 
                             'g-', linewidth=1.5, alpha=0.8)
    
    def reset(self, **kwargs):
        self.traj_num += 1
        self.traj = [[], [], []]  # x, y, z trajectories
        if hasattr(self.env_core, 'num_targets'):
            self.traj_y = [[[], [], []]] * self.env_core.num_targets  # x, y, z for each target
        else:
            self.traj_y = []
        return self.env.reset(**kwargs)
