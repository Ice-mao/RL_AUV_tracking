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

from metadata import METADATA


class Display2D(Wrapper):
    def __init__(self, env, figID=0, skip=1, confidence=0.95):
        super(Display2D, self).__init__(env)
        self.figID = figID  # figID = 0 : train, figID = 1 : test
        self.env_core = env.env.world
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
        if type(self.env_core.targets) == list:
            target_true_pos = [self.env_core.targets[i].state.vec[:2] for i in range(len(self.env_core.targets))]
        else:
            raise ValueError('targets not a list')

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
        num_targets = len(self.env_core.targets)
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
            sensor_arc = patches.Arc((state[0], state[1]), METADATA['agent']['sensor_r'] * 2, METADATA['agent']['sensor_r'] * 2,
                                     angle=state[8], theta1=-METADATA['agent']['fov'] / 2,
                                     theta2=METADATA['agent']['fov'] / 2, edgecolor='black', facecolor='green'
                                     , alpha=0.7)
            ax.add_patch(sensor_arc)
            ax.plot(
                [state[0], state[0] + METADATA['agent']['sensor_r'] * np.cos(np.radians(state[8] + 0.5 * METADATA['agent']['fov']))],
                [state[1], state[1] + METADATA['agent']['sensor_r'] * np.sin(np.radians(state[8] + 0.5 * METADATA['agent']['fov']))],
                'k', linewidth=0.5)
            ax.plot(
                [state[0], state[0] + METADATA['agent']['sensor_r'] * np.cos(np.radians(state[8] - 0.5 * METADATA['agent']['fov']))],
                [state[1], state[1] + METADATA['agent']['sensor_r'] * np.sin(np.radians(state[8] - 0.5 * METADATA['agent']['fov']))],
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
        num_targets = len(self.env_core.targets)
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
        self.traj_y = [[[], []]] * self.env_core.num_targets
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


if __name__ == '__main__':
    # from auv_env import TargetTrackingBase
    # from gymnasium import wrappers

    # env0 = TargetTrackingBase(num_targets=1)
    # env = wrappers.TimeLimit(env0, max_episode_steps=200)

    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Polygon as Draw_Polygon

    # fig, ax = plt.subplots()
    # if hasattr(env.env.world, 'obstacles'):
    #     for polygon in env.env.world.obstacles.polygons:
    #         x, y = polygon.exterior.xy
    #         polygon_patch = Draw_Polygon(np.column_stack((x, y)), closed=True, edgecolor='black', facecolor='black')
    #         ax.add_patch(polygon_patch)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Plot of Shapely Rectangle')
    # plt.grid(True)
    # plt.axis('equal')
    # plt.show()

    # env = Display2D(env, figID=0, local_view=0)
    print("hello world")