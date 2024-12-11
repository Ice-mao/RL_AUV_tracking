import holoocean
import numpy as np
from pynput import keyboard
import matplotlib.pyplot as plt
import cv2
from metadata import METADATA


class KeyBoardCmd:
    """
        # 实现键盘控制的类
        for example:
        kb_cmd = KeyBoardCmd(force=10)
        if 'q' in kb_cmd.pressed_keys:
            break
        command = kb_cmd.parse_keys()

        ## send to holoocean
        env.act("auv0", command)
        state = env.tick()
    """

    def __init__(self, force=25):
        self.force = force
        self.pressed_keys = list()

        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.start()

    def on_press(self, key):
        if hasattr(key, 'char'):
            self.pressed_keys.append(key.char)
            self.pressed_keys = list(set(self.pressed_keys))

    def on_release(self, key):
        if hasattr(key, 'char'):
            self.pressed_keys.remove(key.char)

    def parse_keys(self, force=None):
        command = np.zeros(8)
        if force != None:
            self.force = force

        if 'i' in self.pressed_keys:
            command[0:4] += self.force
        if 'k' in self.pressed_keys:
            command[0:4] -= self.force
        if 'j' in self.pressed_keys:
            command[[4, 7]] += self.force / 2
            command[[5, 6]] -= self.force / 2
        if 'l' in self.pressed_keys:
            command[[4, 7]] -= self.force / 2
            command[[5, 6]] += self.force / 2

        if 'w' in self.pressed_keys:
            command[4:8] += self.force
        if 's' in self.pressed_keys:
            command[4:8] -= self.force
        if 'a' in self.pressed_keys:
            command[[4, 6]] += self.force
            command[[5, 7]] -= self.force
        if 'd' in self.pressed_keys:
            command[[4, 6]] -= self.force
            command[[5, 7]] += self.force
        return command


class ImagingSonar:
    """
        成像声呐进行进一步算法处理的类
        note:返回的图像，第二个维度代表一个角度上的所有图像返回值
         ———————>y
        |
        |
        |
        |x
    """

    def __init__(self, scenario="Dam-HoveringCamera"):
        config = holoocean.packagemanager.get_scenario(scenario)
        config = config['agents'][0]['sensors'][-1]["configuration"]
        self.azi = config['Azimuth']
        self.minR = config['RangeMin']
        self.maxR = config['RangeMax']
        self.binsR = config['RangeBins']  # 256
        self.binsA = config['AzimuthBins']  # 256
        self.zmax = (self.binsR - 10) / self.binsR * (self.maxR - self.minR) + self.minR  # 选取一个合适的更新范围
        self.getimage = False  # world初始化成功，开始收到声呐数据之后为True
        self.__get_plot_ready()

    def __get_plot_ready(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 5))
        self.ax.set_theta_zero_location("N")
        self.ax.set_thetamin(-self.azi / 2)
        self.ax.set_thetamax(self.azi / 2)
        self.theta = np.linspace(-self.azi / 2, self.azi / 2, self.binsA) * np.pi / 180
        self.r = np.linspace(self.minR, self.maxR, self.binsR)
        self.T, self.R = np.meshgrid(self.theta, self.r)
        self.z = np.zeros_like(self.T)
        plt.grid(False)
        self.plot = self.ax.pcolormesh(self.T, self.R, self.z, cmap='gray', shading='auto', vmin=0, vmax=1)
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def sonar_filter(self):
        # TODO:follow the paper
        # 加入了噪声
        # matrix = (self.s * 255).astype(np.uint8)
        # matrix = cv2.medianBlur(matrix, 2)  # 中值滤波
        # # otsu_threshold, _ = cv2.threshold(matrix, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 动态阈值
        # # _, self.s = cv2.threshold(matrix, 1 * otsu_threshold, 255, cv2.THRESH_BINARY)
        # otsu_threshold, self.s = cv2.threshold(self.s, 0, 255, cv2.THRESH_OTSU)
        # self.s = self.s / 255
        # self.s[128:255, :] = 0
        # 不加入噪声
        self.s = (self.s * 255).astype(np.uint8)
        # otsu_threshold, self.s = cv2.threshold(self.s, 1, 255, cv2.THRESH_BINARY)
        otsu_threshold, self.s = cv2.threshold(self.s, 0, 255, cv2.THRESH_OTSU)
        self.s = self.s / 255

    def update(self, state):
        "每个tick更新声呐数据"
        if "ImagingSonar" in state:
            self.s = state['ImagingSonar']
            self.sonar_filter()
            self.getimage = True
        else:
            self.getimage = False

    def draw_pic(self, state):
        "绘制声呐图像"
        self.update(state)
        if self.getimage:
            self.plot.set_array(self.s.ravel())
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    # def determine_borderline(self, state, pose, angle):
    #     # 建立边界地图，但不是栅格地图
    #     if 'ImagingSonar' in state:
    #         # 加入了噪声
    #         # self.s[128:255,:] = 0
    #         self.s = state['ImagingSonar']
    #         x_coordinates, y_coordinates = np.where(self.s)
    #         self.distance = x_coordinates / self.binsR * (self.maxR - self.minR) + self.minR
    #         self.angle = -self.azi / 2 + y_coordinates / self.binsA * (self.azi)
    #         self.x_world = pose[0] + self.distance * np.cos((angle + self.angle) * np.pi / 180)
    #         self.y_world = pose[1] + self.distance * np.sin((angle + self.angle) * np.pi / 180)
    #         # self.borderline_world = list(zip(x_coordinates, y_coordinates))
    #
    #         ax1.scatter(self.x_world, self.y_world)
    #         ax1.scatter(pose[0], pose[1], color='red')
    #         plt.show()

    def scan(self, state, pose, angle):
        distances_x = []
        distances_y = []
        self.distance = []
        nearest_distances_x = []
        nearest_distances_y = []
        nearest_distances = []

        # 生成所有坐标信息
        self.s2map = np.empty(self.s.shape, dtype=object)
        for i in range(self.s.shape[0]):
            for j in range(self.s.shape[1]):
                self.s2map[i, j] = ((i / self.binsR * (self.maxR - self.minR) + self.minR,
                                     -self.azi / 2 + j / self.binsA * (self.azi)))

        x_coordinates, y_coordinates = np.where(self.s)  # x是距离变化，y是角度变换
        # 映射图像中每一个障碍物的坐标到机器人物理坐标
        self.distance = x_coordinates / self.binsR * (self.maxR - self.minR) + self.minR
        self.angle = -self.azi / 2 + y_coordinates / self.binsA * (self.azi)
        distances_x = pose[0] + self.distance * np.cos((angle + self.angle) * np.pi / 180)
        distances_y = pose[1] + self.distance * np.sin((angle + self.angle) * np.pi / 180)

        # 生成每个角度下最近的障碍物坐标,如果没有识别出障碍物，则假设最远位置有障碍物
        for i in range(self.s.shape[1]):
            dis_index = np.argmax(self.s[:, i] > 0.1)
            if dis_index == 0:
                dis_index = self.binsR
            nearest_distance = dis_index / self.binsR * (self.maxR - self.minR) + self.minR
            nearest_angle = -self.azi / 2 + i / self.binsA * (self.azi)
            nearest_distance_x = pose[0] + nearest_distance * np.cos((angle + nearest_angle) * np.pi / 180)
            nearest_distance_y = pose[1] + nearest_distance * np.sin((angle + nearest_angle) * np.pi / 180)
            nearest_distances_x.append(nearest_distance_x)
            nearest_distances_y.append(nearest_distance_y)
            nearest_distances.append(nearest_distance)

        return (distances_x, distances_y, self.distance, nearest_distances_x, nearest_distances_y, nearest_distances)


class PoseLocation:
    def __init__(self):
        self.location = np.zeros(3)
        self.locationxy = np.zeros(2)
        self.direction = 0

    def update(self, state):
        self.s = state['PoseSensor']
        self.location = np.array([self.s[0][3], self.s[1][3], self.s[2][3]])
        self.truelocation = state['LocationSensor']
        self.locationxy = self.location[0:2]
        self.truelocationxy = self.location[0:2]
        self.angle = np.degrees(np.arctan2(self.s[1, 0], self.s[0, 0]))
        if self.angle < 0:
            self.angle = 360 + self.angle
        self.direction = self.angle * np.pi / 180  # 用rad表示的方位角

class RangeFinder:
    """
        Returns distances to nearest collisions in the directions specified by the parameters.
    """

    def __init__(self, scenario):
        # init config
        config = holoocean.packagemanager.get_scenario(scenario)
        self.LaserMaxDistance = 1
        self.LaserCount = 1
        self.LaserDebug = 1
        for sensor in config['agents'][0]['sensors']:
            if 'sensor_type' in sensor:
                if sensor['sensor_type'] == 'RangeFinderSensor':
                    config = sensor["configuration"]
                    self.LaserMaxDistance = config['LaserMaxDistance']
                    self.LaserCount = config['LaserCount']
                    self.LaserDebug = config['LaserDebug']
        self.min_distance = self.LaserMaxDistance
        self.min_angle = 0
        self.azi = int(self.LaserCount/3)
        self.azi_half = int(self.LaserCount / 6)
        self.angle = np.linspace(0, 360, self.LaserCount, endpoint=False)  # angle list
        self.get_scan = False

    def update(self, state):
        if not METADATA['agent']['use_sonar']:
            self.get_scan = False
            if 'RangeFinderSensor' in state:
                self.get_scan = True
                range_data = state['RangeFinderSensor']
                # update the minimum distance and its angle to nearest obstacle
                # if np.min(range_data) < self.LaserMaxDistance:
                self.min_distance = np.min(range_data)
                self.min_angle = (360 / self.LaserCount) * np.argmin(range_data)
                # else:
                #     self.min_distance = None
                #     self.angle = None
        else:
            # use just to simulate sonar update process
            self.update_like_sonar(state)

    def update_like_sonar(self, state):
        self.get_scan = False
        if 'RangeFinderSensor' in state:
            self.get_scan = True
            range_data = state['RangeFinderSensor']
            # update the minimum distance and its angle to nearest obstacle
            # if np.min(range_data) < self.LaserMaxDistance:

            # The first definition of sonar-like
            # left_part = range_data[:self.azi_half][::-1]
            # right_part = range_data[-self.azi_half:][::-1]
            # self.data = np.concatenate((left_part, right_part))
            # self.min_distance = np.min(self.data)
            # self.min_angle = (120 / self.azi) * np.argmin(self.data) - 60

            # The second definition keep form of raw data
            part_1 = range_data[:self.azi_half]
            part_2 = range_data[-self.azi_half:]
            angel_1 = self.angle[:self.azi_half]
            angel_2 = self.angle[-self.azi_half:]
            self.data = np.concatenate((part_1, part_2))
            self.angle = np.concatenate((angel_1, angel_2))

            self.min_distance = np.min(range_data)
            self.min_angle = (360 / self.LaserCount) * np.argmin(range_data)


    def scan(self, pose, angle):
        distances_x = []
        distances_y = []
        nearest_distances_x = []
        nearest_distances_y = []
        nearest_distances = []

        # 投影到全局坐标系下
        distances_x = pose[0] + self.data * np.cos((angle + self.angle) * np.pi / 180)
        distances_y = pose[1] + self.data * np.sin((angle + self.angle) * np.pi / 180)

        # # 生成每个角度下最近的障碍物坐标,如果没有识别出障碍物，则假设最远位置有障碍物
        # for i in range(self.s.shape[1]):
        #     dis_index = np.argmax(self.s[:, i] > 0.1)
        #     if dis_index == 0:
        #         dis_index = self.binsR
        #     nearest_distance = dis_index / self.binsR * (self.maxR - self.minR) + self.minR
        #     nearest_angle = -self.azi / 2 + i / self.binsA * (self.azi)
        #     nearest_distance_x = pose[0] + nearest_distance * np.cos((angle + nearest_angle) * np.pi / 180)
        #     nearest_distance_y = pose[1] + nearest_distance * np.sin((angle + nearest_angle) * np.pi / 180)
        #     nearest_distances_x.append(nearest_distance_x)
        #     nearest_distances_y.append(nearest_distance_y)
        #     nearest_distances.append(nearest_distance)

        return (distances_x, distances_y, self.data)


import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
import seaborn as sns

sns.set(context="paper", style="whitegrid", font_scale=0.8)
"""
    绘图类的使用
"""
class Plotter:
    def __init__(self, names):
        # Where all the data is stored
        self.t = []
        self.data = None

        self.num_row = 5
        self.num_col = 3
        self.num_items = len(names)

        # Setup figure
        self.fig, self.ax = plt.subplots(
            self.num_row, self.num_col, figsize=(6, 8), sharex=True
        )

        # Setup all lines
        self.lines = [[[] for _ in range(self.num_row)] for _ in range(len(names))]
        for i in range(self.num_row):
            for j in range(self.num_col):
                for k, n in enumerate(names):
                    (p,) = self.ax[i, j].plot([], [], label=n)

                    self.lines[k][i].append(p)

        self.ax[-1, 2].legend()

        # Add axes labels
        titles = ["Position", "Velocity", "RPY", "Bias - Omega", "Bias - Acceleration"]
        for i in range(self.num_row):
            self.ax[i, 1].set_title(titles[i])
        self.fig.tight_layout()
        plt.show(block=False)

    def add_timestep(self, t, states):
        # Keep the time
        self.t.append(t)

        # Plop our data at the end of the other data
        new_state = np.stack([s.data_plot for s in states])
        if self.data is None:
            self.data = new_state
        else:
            self.data = np.dstack((self.data, new_state))

    def _rot_to_rpy(self, mat):
        return Rotation.from_matrix(mat).as_euler("xyz")

    def update_plots(self):
        # Update all lines
        for i in range(self.num_row):
            for j in range(self.num_col):
                for k in range(self.num_items):
                    self.lines[k][i][j].set_data(self.t, self.data[k,self.num_col*i+j])

                self.ax[i, j].relim()
                self.ax[i, j].autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
