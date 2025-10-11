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
        # Render for debugging and visualizing training progress if needed
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
            ax.plot(self.traj[0], self.traj[1], 'b.', markersize=2)

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
        self.fig = plt.figure(self.figID, figsize=(12, 10))
        self.ax = None  # Save 3D axes object
        self.n_frames = 0
        self.skip = skip
        self.c_cf = np.sqrt(-2 * np.log(1 - confidence))
        self.traj_num = 0
        
        # Save view information to maintain user's rotation state
        # Adjust default view: higher elevation angle for overhead view, avoiding trajectory occlusion
        self.view_elev = 35  # Default elevation angle (increased from 25 to 35 degrees for better overhead view)
        self.view_azim = -60  # Default azimuth angle (adjusted for better field of view)
        
        # Color scheme for better visualization
        self.colors = {
            'agent': '#1f77b4',         # Blue
            'agent_est': '#17becf',     # Cyan
            'target': '#d62728',        # Red
            'belief': '#2ca02c',        # Green
            'trajectory': '#ff7f0e',    # Orange
            'sensor_fov': '#7bccc4',    # Light green
            'obstacle': "#2F2E2E",      # Dark gray
        }

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
            # If axes already exist, save current view angle
            if self.ax is not None:
                try:
                    self.view_elev = self.ax.elev
                    self.view_azim = self.ax.azim
                except:
                    pass  # If getting view angle fails, use default values
            
            self.fig.clf()
            
            # Create single 3D view with interactive controls
            self.ax = self.fig.add_subplot(111, projection='3d')

            # ========== 3D View ==========
            self._render_3d_view(self.ax, state, est_state, target_true_pos, target_b_state, target_cov, num_targets)
            
            # Restore previous view angle
            self.ax.view_init(elev=self.view_elev, azim=self.view_azim)
            
            if not record:
                # Enable interactive mode for mouse rotation
                plt.ion()
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(0.01)

        self.n_frames += 1

    def _render_3d_view(self, ax, state, est_state, target_true_pos, target_b_state, target_cov, num_targets):
        """Render the main 3D perspective view"""
        ax.set_title("3D Perspective View", fontsize=12, fontweight='bold', pad=10)
        
        # Reduce grid line density for cleaner style
        ax.set_facecolor('white')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Reduce number of ticks to minimize grid lines
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.zaxis.set_major_locator(plt.MaxNLocator(5))
        
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        ax.xaxis.pane.set_alpha(0.1)
        ax.yaxis.pane.set_alpha(0.1)
        ax.zaxis.pane.set_alpha(0.1)
        
        # Draw obstacles if available
        if hasattr(self.env_core, 'obstacles'):
            self._draw_3d_obstacles(ax)
        
        # Draw targets
        if hasattr(self.env_core, 'targets'):
            for i in range(num_targets):
                # Target trajectory - restored to original dark red, increased zorder to ensure above obstacles
                if len(self.traj_y) > i and len(self.traj_y[i][0]) > 0:
                    ax.plot(self.traj_y[i][0], self.traj_y[i][1], self.traj_y[i][2], 
                           'r-', linewidth=2.5, alpha=0.9,  # Increased line width and opacity
                           label='Target Path' if i == 0 else "", zorder=10)  # High zorder to ensure topmost layer
                
                # Target belief (simplified as sphere)
                if len(target_cov) > i and target_cov[i].shape[0] >= 3:
                    self._draw_belief_ellipsoid(ax, target_b_state[i][:3], target_cov[i][:3, :3])
                
                # Current target position
                ax.scatter(target_true_pos[i][0], target_true_pos[i][1], target_true_pos[i][2], 
                          c='red', s=100, marker='o', 
                          label='Target' if i == 0 else "", 
                          edgecolors='darkred', linewidth=2, zorder=5)
                
                # Target belief center
                ax.scatter(target_b_state[i][0], target_b_state[i][1], target_b_state[i][2], 
                          c='green', s=80, marker='s', alpha=0.6, 
                          label='Belief' if i == 0 else "", 
                          edgecolors='darkgreen', linewidth=1.5, zorder=4)

        # Draw agent trajectory - restored to original dark blue, increased zorder to ensure above obstacles
        if len(self.traj[0]) > 1:
            ax.plot(self.traj[0], self.traj[1], self.traj[2], 
                   'b-', linewidth=2.5, alpha=0.9,  # Increased line width and opacity
                   label='AUV Path', zorder=10)  # High zorder to ensure topmost layer
        
        # Current agent position
        ax.scatter(state[0], state[1], state[2], 
                  c='blue', s=150, marker='^', 
                  label='AUV', edgecolors='darkblue', linewidth=2, zorder=6)
        
        # Agent estimated position
        ax.scatter(est_state[0], est_state[1], est_state[2], 
                  c='cyan', s=120, marker='^', 
                  alpha=0.5, label='AUV Est.', 
                  edgecolors='darkcyan', linewidth=1.5, zorder=5)
        
        # Draw agent sensor FOV
        self._draw_3d_sensor_fov(ax, state)
        
        # Set 3D view limits
        ax.set_xlim(self.mapmin[0] - 1, self.mapmax[0] + 1)
        ax.set_ylim(self.mapmin[1] - 1, self.mapmax[1] + 1)
        ax.set_zlim(self.mapmin[2] - 1, self.mapmax[2] + 1)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_zlabel('Z (m)', fontsize=10)
        
        # Add legend with better positioning
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)

    def _draw_3d_obstacles(self, ax):
        """Draw 3D obstacles with professional appearance for paper publication"""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        for i, layer in enumerate([self.env_core.obstacles.polygons_layer1, self.env_core.obstacles.polygons_layer2]):
            for polygon in layer:
                # Get polygon bounding box
                x_coords, y_coords = polygon.exterior.xy
                x_coords = list(x_coords[:-1])  # Remove duplicate last point
                y_coords = list(y_coords[:-1])
                               
                # Define z-coordinates for obstacles
                z_min = self.env_core.obstacles.fix_depths[i] - 1
                z_max = self.env_core.obstacles.fix_depths[i] + 1

                # Define the 8 vertices of the box
                vertices = np.array([
                    [x_coords[0], y_coords[0], z_min], [x_coords[1], y_coords[1], z_min],
                    [x_coords[2], y_coords[2], z_min], [x_coords[3], y_coords[3], z_min],
                    [x_coords[0], y_coords[0], z_max], [x_coords[1], y_coords[1], z_max], 
                    [x_coords[2], y_coords[2], z_max], [x_coords[3], y_coords[3], z_max]
                ])
                
                # Define the 6 faces of the box
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
                    [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
                    [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
                    [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
                    [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
                    [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
                ]
                
                # Create solid obstacle with realistic appearance
                # Use light gray and lower transparency, better for paper presentation without excessive trajectory occlusion
                poly3d = Poly3DCollection(faces, 
                                         facecolors="#B0B0B0",  # Light gray
                                         edgecolors='#707070',  # Medium gray border
                                         linewidths=1.0, 
                                         alpha=0.5,  # Reduced opacity to avoid trajectory occlusion
                                         zorder=0)  # Low zorder to ensure below trajectories
                ax.add_collection3d(poly3d)

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
        """Draw 3D sensor field of view as a cone"""
        if 'agent' in self.config and 'sensor_r' in self.config['agent']:
            sensor_range = self.config['agent']['sensor_r']
            h_fov = self.config['agent']['fov'] * np.pi / 180  # horizontal FOV
            v_fov = self.config['agent'].get('h_fov', 60) * np.pi / 180  # vertical FOV
            
            # Agent position and orientation
            pos = state[:3]
            yaw = np.radians(state[8])
            
            # Draw FOV cone with more points for better visualization
            n_points = 16  # number of points around the cone base
            
            # Generate cone base points
            cone_points = []
            for i in range(n_points):
                angle = 2 * np.pi * i / n_points
                
                # Calculate direction within FOV
                h_angle = (h_fov / 2) * np.cos(angle)
                v_angle = (v_fov / 2) * np.sin(angle)
                
                direction = np.array([
                    np.cos(yaw + h_angle) * np.cos(v_angle),
                    np.sin(yaw + h_angle) * np.cos(v_angle),
                    np.sin(v_angle)
                ])
                
                end_point = pos + sensor_range * direction
                cone_points.append(end_point)
            
            # Draw cone edges from tip to base points
            for i in range(0, n_points, 4):  # Draw every 4th line to reduce clutter
                ax.plot3D([pos[0], cone_points[i][0]], 
                         [pos[1], cone_points[i][1]], 
                         [pos[2], cone_points[i][2]], 
                         color='green', linewidth=1.5, alpha=0.5, zorder=2)
            
            # Draw cone base circle
            for i in range(n_points):
                next_i = (i + 1) % n_points
                ax.plot3D([cone_points[i][0], cone_points[next_i][0]], 
                         [cone_points[i][1], cone_points[next_i][1]], 
                         [cone_points[i][2], cone_points[next_i][2]], 
                         color='green', linewidth=1, alpha=0.4, zorder=2)
    
    def _render_top_view(self, ax, state, est_state, target_true_pos, target_b_state, target_cov, num_targets):
        """Render top view (X-Y plane, looking down on Z axis)"""
        ax.set_title("Top View (X-Y Plane)", fontsize=12, fontweight='bold', pad=10)
        ax.set_facecolor('#f8f8f8')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Draw obstacles (project onto X-Y plane)
        if hasattr(self.env_core, 'obstacles'):
            for layer in [self.env_core.obstacles.polygons_layer1, self.env_core.obstacles.polygons_layer2]:
                for polygon in layer:
                    x_coords, y_coords = polygon.exterior.xy
                    ax.fill(x_coords, y_coords, color=self.colors['obstacle'], alpha=0.5, edgecolor='black', linewidth=1.5)
        
        # Draw targets
        if hasattr(self.env_core, 'targets'):
            for i in range(num_targets):
                # Target trajectory - restored to original dark red
                if len(self.traj_y) > i and len(self.traj_y[i][0]) > 0:
                    ax.plot(self.traj_y[i][0], self.traj_y[i][1], 
                           'r-', linewidth=2, alpha=0.8)
                
                # Target belief ellipse
                if len(target_cov) > i and target_cov[i].shape[0] >= 2:
                    self._draw_belief_ellipse_2d(ax, target_b_state[i][:2], target_cov[i][:2, :2])
                
                # Current target position
                ax.scatter(target_true_pos[i][0], target_true_pos[i][1], 
                          c='red', s=100, marker='o', 
                          edgecolors='darkred', linewidth=2, zorder=5)
                
                # Target belief center
                ax.scatter(target_b_state[i][0], target_b_state[i][1], 
                          c='green', s=80, marker='s', alpha=0.6, 
                          edgecolors='darkgreen', linewidth=1.5, zorder=4)
        
        # Draw agent trajectory - restored to original dark blue
        if len(self.traj[0]) > 1:
            ax.plot(self.traj[0], self.traj[1], 
                   'b-', linewidth=2, alpha=0.8, zorder=3)
        
        # Current agent position with orientation arrow
        ax.scatter(state[0], state[1], 
                  c='blue', s=150, marker='^', 
                  edgecolors='darkblue', linewidth=2, zorder=6)
        
        # Agent estimated position
        ax.scatter(est_state[0], est_state[1], 
                  c='cyan', s=120, marker='^', 
                  alpha=0.5, edgecolors='darkcyan', linewidth=1.5, zorder=5)
        
        # Draw orientation arrow
        arrow_length = 5
        yaw = np.radians(state[8])
        ax.arrow(state[0], state[1], 
                arrow_length * np.cos(yaw), arrow_length * np.sin(yaw),
                head_width=2, head_length=1.5, fc='blue', 
                ec='blue', alpha=0.8, linewidth=2, zorder=6)
        
        # Draw sensor FOV in 2D
        if 'agent' in self.config and 'sensor_r' in self.config['agent']:
            self._draw_2d_sensor_fov(ax, state)
        
        # Set limits and labels
        ax.set_xlim(self.mapmin[0] - 2, self.mapmax[0] + 2)
        ax.set_ylim(self.mapmin[1] - 2, self.mapmax[1] + 2)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Y (m)', fontsize=10)
        ax.set_aspect('equal', 'box')
    
    def _render_side_view(self, ax, state, est_state, target_true_pos, target_b_state, target_cov, num_targets):
        """Render side view (X-Z plane, looking along Y axis)"""
        ax.set_title("Side View (X-Z Plane)", fontsize=12, fontweight='bold', pad=10)
        ax.set_facecolor('#f8f8f8')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Draw environment boundaries
        ax.axhline(y=self.mapmin[2], color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax.axhline(y=self.mapmax[2], color='gray', linestyle='--', alpha=0.3, linewidth=1)
        
        # Draw obstacles (project onto X-Z plane - simplified as rectangles)
        if hasattr(self.env_core, 'obstacles'):
            for i, layer in enumerate([self.env_core.obstacles.polygons_layer1, 
                                      self.env_core.obstacles.polygons_layer2]):
                z_depth = self.env_core.obstacles.fix_depths[i]
                for polygon in layer:
                    x_coords, y_coords = polygon.exterior.xy
                    x_min, x_max = min(x_coords), max(x_coords)
                    # Draw as rectangle at fixed depth
                    ax.add_patch(patches.Rectangle((x_min, z_depth - 1), x_max - x_min, 2,
                                                   facecolor=self.colors['obstacle'], 
                                                   edgecolor='black', alpha=0.5, linewidth=1.5))
        
        # Draw targets
        if hasattr(self.env_core, 'targets'):
            for i in range(num_targets):
                # Target trajectory - restored to original dark red
                if len(self.traj_y) > i and len(self.traj_y[i][0]) > 0:
                    ax.plot(self.traj_y[i][0], self.traj_y[i][2], 
                           'r-', linewidth=2, alpha=0.8)
                
                # Current target position
                ax.scatter(target_true_pos[i][0], target_true_pos[i][2], 
                          c='red', s=100, marker='o', 
                          edgecolors='darkred', linewidth=2, zorder=5)
                
                # Target belief center
                ax.scatter(target_b_state[i][0], target_b_state[i][2], 
                          c='green', s=80, marker='s', alpha=0.6, 
                          edgecolors='darkgreen', linewidth=1.5, zorder=4)
        
        # Draw agent trajectory - restored to original dark blue
        if len(self.traj[0]) > 1:
            ax.plot(self.traj[0], self.traj[2], 
                   'b-', linewidth=2, alpha=0.8, zorder=3)
        
        # Current agent position
        ax.scatter(state[0], state[2], 
                  c='blue', s=150, marker='^', 
                  edgecolors='darkblue', linewidth=2, zorder=6)
        
        # Agent estimated position
        ax.scatter(est_state[0], est_state[2], 
                  c='cyan', s=120, marker='^', 
                  alpha=0.5, edgecolors='darkcyan', linewidth=1.5, zorder=5)
        
        # Set limits and labels
        ax.set_xlim(self.mapmin[0] - 2, self.mapmax[0] + 2)
        ax.set_ylim(self.mapmin[2] - 2, self.mapmax[2] + 2)
        ax.set_xlabel('X (m)', fontsize=10)
        ax.set_ylabel('Z (m)', fontsize=10)
        ax.set_aspect('equal', 'box')
    
    def _draw_belief_ellipse_2d(self, ax, center, cov_matrix):
        """Draw 2D belief ellipse for top view"""
        try:
            eig_val, eig_vec = LA.eig(cov_matrix)
            angle = 180 / np.pi * np.arctan2(np.real(eig_vec[0][1]), np.real(eig_vec[0][0]))
            ellipse = patches.Ellipse(
                (center[0], center[1]),
                2 * np.sqrt(np.real(eig_val[0])) * self.c_cf,
                2 * np.sqrt(np.real(eig_val[1])) * self.c_cf,
                angle=angle, fill=True, zorder=2,
                facecolor=self.colors['belief'], alpha=0.3,
                edgecolor=self.colors['belief'], linewidth=1.5)
            ax.add_patch(ellipse)
        except:
            # If eigenvalue decomposition fails, draw a circle
            radius = np.sqrt(np.trace(cov_matrix)) * self.c_cf / 2
            circle = patches.Circle((center[0], center[1]), radius, 
                                   facecolor=self.colors['belief'], alpha=0.3,
                                   edgecolor=self.colors['belief'], linewidth=1.5)
            ax.add_patch(circle)
    
    def _draw_2d_sensor_fov(self, ax, state):
        """Draw 2D sensor field of view for top view"""
        sensor_range = self.config['agent']['sensor_r']
        fov = self.config['agent']['fov']
        yaw = state[8]
        
        # Draw sensor arc
        sensor_arc = patches.Wedge((state[0], state[1]), sensor_range,
                                  theta1=yaw - fov/2, theta2=yaw + fov/2,
                                  facecolor=self.colors['sensor_fov'], 
                                  edgecolor=self.colors['sensor_fov'],
                                  alpha=0.3, linewidth=1.5, zorder=2)
        ax.add_patch(sensor_arc)
        
        # Draw FOV boundary lines
        for angle_offset in [-fov/2, fov/2]:
            angle_rad = np.radians(yaw + angle_offset)
            end_x = state[0] + sensor_range * np.cos(angle_rad)
            end_y = state[1] + sensor_range * np.sin(angle_rad)
            ax.plot([state[0], end_x], [state[1], end_y], 
                   color=self.colors['sensor_fov'], linewidth=1.5, alpha=0.6, zorder=2)
    
    def reset(self, **kwargs):
        self.traj_num += 1
        self.traj = [[], [], []]  # x, y, z trajectories
        if hasattr(self.env_core, 'num_targets'):
            self.traj_y = [[[], [], []]] * self.env_core.num_targets  # x, y, z for each target
        else:
            self.traj_y = []
        return self.env.reset(**kwargs)
