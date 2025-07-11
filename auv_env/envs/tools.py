import holoocean
import numpy as np
from pynput import keyboard
# import keyboard
import matplotlib.pyplot as plt
import cv2
from auv_env import util


class KeyBoardCmd:
    """
        # 实现键盘控制的类
        for example:
        from auv_env.envs.tools import KeyBoardCmd
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
            command[[4, 7]] += self.force / 4
            command[[5, 6]] -= self.force / 4
        if 'l' in self.pressed_keys:
            command[[4, 7]] -= self.force / 4
            command[[5, 6]] += self.force / 4

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

    def __init__(self, scenario, config):
        # init config
        self.config = config
        config_ho = holoocean.packagemanager.get_scenario(scenario)
        self.LaserMaxDistance = 1
        self.LaserCount = 1
        self.LaserDebug = 1
        for sensor in config_ho['agents'][0]['sensors']:
            if 'sensor_type' in sensor:
                if sensor['sensor_type'] == 'RangeFinderSensor':
                    config_ho = sensor["configuration"]
                    self.LaserMaxDistance = config_ho['LaserMaxDistance']
                    self.LaserCount = config_ho['LaserCount']
                    self.LaserDebug = config_ho['LaserDebug']
        self.min_distance = self.LaserMaxDistance
        self.min_angle = 0
        self.azi = int(self.LaserCount / 3)
        self.azi_half = int(self.LaserCount / 6)
        self.angle = np.linspace(0, 360, self.LaserCount, endpoint=False)  # angle list
        self.get_scan = False

    def update(self, state):
        if not self.config['agent']['use_sonar']:
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


from collections import deque
from PIL import Image
from torchvision import transforms

class BaseImageBuffer:
    """
    Base class for an image buffer that handles common buffer logic.
    """
    def __init__(self, buffer_size, image_shape, time_gap):
        """
        Initializes the image buffer.
        :param buffer_size: The maximum length of the buffer.
        :param image_shape: The shape of the images, e.g., (3, 224, 224).
        :param time_gap: The minimum time interval to accept a new image.
        """
        if not isinstance(image_shape, tuple) or len(image_shape) != 3:
            raise ValueError("image_shape must be a tuple containing (channels, height, width)")
            
        self.buffer_size = buffer_size
        self.image_shape = image_shape
        self.time_gap = time_gap
        self.buffer = deque([self._create_empty_image()] * self.buffer_size, maxlen=self.buffer_size)
        self.t = 0

    def _create_empty_image(self):
        return np.zeros(self.image_shape, dtype=np.uint8)

    def reset(self):
        self.buffer = deque([self._create_empty_image()] * self.buffer_size, maxlen=self.buffer_size)
        self.t = 0

    def _preprocess(self, image):
        """
        Abstract method for preprocessing an image. Subclasses must implement this.
        Should return a numpy array of shape self.image_shape and dtype uint8.
        """
        raise NotImplementedError("Subclasses must implement the _preprocess method")

    def add_image(self, image, t):
        """
        Preprocesses and adds an image to the buffer if the time gap condition is met.
        :param image: The raw image data.
        :param t: The timestamp of the image.
        """
        if abs(t - self.t) >= self.time_gap:
            processed_image = self._preprocess(image)
            self.t = t
            self.buffer.append(processed_image)

    def get_buffer(self):
        return list(self.buffer)

class CameraBuffer(BaseImageBuffer):
    """A buffer for storing and processing camera images."""
    def __init__(self, buffer_size=5, image_shape=(3, 224, 224), time_gap=0.1):
        super().__init__(buffer_size, image_shape, time_gap)

    def _preprocess(self, image):
        """
        Preprocesses a camera image:
        1. Extracts BGR channels.
        2. Converts BGR to RGB.
        3. Resizes the image.
        4. Converts to uint8.
        5. Transposes dimensions to (C, H, W).
        """
        bgr_image = image[:, :, :3]
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(rgb_image, (self.image_shape[2], self.image_shape[1]), interpolation=cv2.INTER_AREA)
        # Ensure the type is uint8
        uint8_image = resized_image.astype(np.uint8)
        # Transpose from HWC to CHW
        return uint8_image.transpose(2, 0, 1)

class SonarBuffer(BaseImageBuffer):
    """A buffer for storing and processing sonar images."""
    def __init__(self, buffer_size=5, image_shape=(1, 128, 128), time_gap=0.1):
        super().__init__(buffer_size, image_shape, time_gap)

    def _preprocess(self, image):
        """
        Preprocesses a sonar image:
        1. Converts float data to the 0-255 range.
        2. Resizes the image.
        3. Ensures it is a single-channel grayscale image.
        4. Converts to uint8.
        5. Ensures the shape is (1, H, W).
        """
        # Assuming the input is a float from 0-1
        sonar_data = (image * 255).astype(np.uint8)
        resized_image = cv2.resize(sonar_data, (self.image_shape[2], self.image_shape[1]), interpolation=cv2.INTER_AREA)
        
        if len(resized_image.shape) == 2:
            # Add the channel dimension
            return np.expand_dims(resized_image, axis=0).astype(np.uint8)
        elif len(resized_image.shape) == 3 and resized_image.shape[2] == 1:
             # Transpose from H, W, C -> C, H, W
            return resized_image.transpose(2, 0, 1).astype(np.uint8)
        else:
            raise ValueError(f"Sonar image should be single-channel, but received shape {resized_image.shape}")


import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
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
                    self.lines[k][i][j].set_data(self.t, self.data[k, self.num_col * i + j])

                self.ax[i, j].relim()
                self.ax[i, j].autoscale_view()

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

if __name__=='__main__':
    # --- CameraBuffer 示例 ---
    print("--- 测试 CameraBuffer ---")
    cam_buffer_size = 5
    cam_image_shape = (3, 224, 224)
    cam_buffer = CameraBuffer(cam_buffer_size, cam_image_shape, time_gap=0.1)

    # 查看初始状态
    print(f"初始状态: {cam_buffer}")
    initial_cam_images = cam_buffer.get_buffer()
    print(f"初始缓冲区中有 {len(initial_cam_images)} 张图像")
    print(f"第一张空图像的形状: {initial_cam_images[0].shape}, 类型: {initial_cam_images[0].dtype}")

    # 添加一些随机相机图像 (模拟 HoloOcean RGBA 输出)
    for i in range(7):
        # 模拟一个 480x640 的 4 通道图像
        new_cam_image = np.random.randint(0, 256, (480, 640, 4), dtype=np.uint8)
        cam_buffer.add_image(new_cam_image, t=i * 0.1)
        print(f"添加第 {i + 1} 张相机图像后: {cam_buffer}")

    final_cam_images = cam_buffer.get_buffer()
    print(f"\n最终相机缓冲区中有 {len(final_cam_images)} 张图像")
    print(f"最后一幅图像的形状: {final_cam_images[-1].shape}, 类型: {final_cam_images[-1].dtype}")
    assert final_cam_images[-1].shape == cam_image_shape
    assert final_cam_images[-1].dtype == np.uint8
    print("-" * 25)

    # --- SonarBuffer 示例 ---
    print("\n--- 测试 SonarBuffer ---")
    sonar_buffer_size = 4
    sonar_image_shape = (1, 128, 128)
    sonar_buffer = SonarBuffer(sonar_buffer_size, sonar_image_shape, time_gap=0.2)

    # 查看初始状态
    print(f"初始状态: {sonar_buffer}")
    initial_sonar_images = sonar_buffer.get_buffer()
    print(f"初始缓冲区中有 {len(initial_sonar_images)} 张图像")
    print(f"第一张空图像的形状: {initial_sonar_images[0].shape}, 类型: {initial_sonar_images[0].dtype}")

    # 添加一些随机声呐图像 (模拟 0-1 浮点数输出)
    for i in range(6):
        new_sonar_image = np.random.rand(256, 256) # 模拟原始声呐数据
        sonar_buffer.add_image(new_sonar_image, t=i * 0.2)
        print(f"添加第 {i + 1} 张声呐图像后: {sonar_buffer}")

    final_sonar_images = sonar_buffer.get_buffer()
    print(f"\n最终声呐缓冲区中有 {len(final_sonar_images)} 张图像")
    print(f"最后一幅图像的形状: {final_sonar_images[-1].shape}, 类型: {final_sonar_images[-1].dtype}")
    assert final_sonar_images[-1].shape == sonar_image_shape
    assert final_sonar_images[-1].dtype == np.uint8
    print("-" * 25)
