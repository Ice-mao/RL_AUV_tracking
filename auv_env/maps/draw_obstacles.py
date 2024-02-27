import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
import matplotlib.patches as patches

import numpy as np

import os
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log_dir', type=str, default='.')
parser.add_argument('--start_index', type=int, default=0)

args = parser.parse_args()

class DrawObstacles(object):
    def __init__(self):
        self.fig, self.ax = plt.subplots()

        self.index = args.start_index - 1
        self.dim = (50,50)
        self.update_image()

        def visualize(i):
            pass

        self.fig.canvas.mpl_connect('motion_notify_event', self.onMouseMove)
        self.fig.canvas.callbacks.connect('button_press_event', self.clickCallback)
        self.fig.canvas.callbacks.connect('key_press_event', self.key_callback)
        ani = animation.FuncAnimation(self.fig, visualize, interval=50)

        plt.show()
        plt.close()

    def onMouseMove(self, event):
        if event.xdata is None or event.ydata is None:
            return
        self.ax.lines = []
        self.ax.axvline(x=event.xdata, color="r")
        self.ax.axhline(y=event.ydata, color="r")

    def clickCallback(self, event):
        print("Clicked: ({}, {})".format(event.xdata, event.ydata))
        if len(self.points) == 0 or len(self.points[-1]) == 2:
            self.points.append([])

        x = max(0, min(event.xdata, self.dim[1]))
        y = max(0, min(event.ydata, self.dim[0]))
        self.points[-1].append([x, y])

        if len(self.points[-1]) == 2:
            self.ax.add_patch(patches.Rectangle((self.points[-1][0][0], self.points[-1][0][1]),
                                                self.points[-1][1][0] - self.points[-1][0][0],
                                                self.points[-1][1][1] - self.points[-1][0][1],
                                                fill=False, edgecolor='r'))

    def fill_box(self, box_points):
        points_idx = [[int(np.floor(box_points[0][0] + 0.5)),
                        int(np.floor(box_points[0][1] + 0.5))],
                      [int(np.floor(box_points[1][0] + 0.5)),
                        int(np.floor(box_points[1][1] + 0.5))]]
        if points_idx[0][0] < points_idx[1][0]:
            c0 = points_idx[0][0]
            c1 = points_idx[1][0]
        else:
            c0 = points_idx[1][0]
            c1 = points_idx[0][0]

        if points_idx[0][1] < points_idx[1][1]:
            r0 = points_idx[0][1]
            r1 = points_idx[1][1]
        else:
            r0 = points_idx[1][1]
            r1 = points_idx[0][1]

        self.img[r0:r1, c0:c1] = 1.0
        self.ax.imshow(self.img, cmap='gray_r')

    def key_callback(self, event):
        if event.key == 'n':
            if len(self.points) == 0 or len(self.points[-1]) == 2:
                self.save_objects()
                self.update_image()
            else:
                print("An image must have an even number of bounding box corners")
                print(self.points)
        if event.key == 'c':
            self.points = []
            [p.remove() for p in reversed(self.ax.patches)]
            self.img = np.zeros((self.dim))
            self.ax.imshow(self.img, cmap='gray_r')
            print("Cleared points")

        if event.key == 'f':
            for i in range(len(self.points)):
                self.fill_box(self.points[i])

    def save_objects(self):
        print("Saving the current object")
        np.save(os.path.join(args.log_dir, 'obstacle_%d'%self.index), self.img)

    def update_image(self):
        self.index += 1
        self.points = []
        self.img = np.zeros((self.dim))
        self.ax.clear()

        imgplot = self.ax.imshow(self.img, cmap='gray_r')

if __name__ == '__main__':
    l = DrawObstacles()
