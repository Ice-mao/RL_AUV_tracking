import holoocean
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pynput import keyboard

pressed_keys = list()
force = 25

def on_press(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.append(key.char)
        pressed_keys = list(set(pressed_keys))

def on_release(key):
    global pressed_keys
    if hasattr(key, 'char'):
        pressed_keys.remove(key.char)

listener = keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener.start()

def feature_extractor(state,showpicture):
    if "LeftCamera" in state and showpicture:
        cv2.namedWindow("Camera Output")
        pixels = state["LeftCamera"]
        leftCameraRGB = pixels[:, :, 0:3]
        leftCameragray = cv2.cvtColor(leftCameraRGB, cv2.COLOR_BGR2GRAY)
        keyframe = Keyframe(leftCameragray)
        keyframe.computeBRIEFPoint()
        print("Number of Keypoints:", len(keyframe.keypoints))
        # print("Descriptor Shape:", keyframe.brief_descriptors.shape)
        leftCameraBRISK = cv2.drawKeypoints(leftCameragray, keyframe.keypoints, None)
        cv2.imshow("Camera Output", leftCameraBRISK)
        cv2.waitKey(1)
    elif not showpicture:
        cv2.destroyAllWindows()
def parse_keys(keys, val):
    command = np.zeros(8)
    if 'i' in keys:
        command[0:4] += val
    if 'k' in keys:
        command[0:4] -= val
    if 'j' in keys:
        command[[4,7]] += val
        command[[5,6]] -= val
    if 'l' in keys:
        command[[4,7]] -= val
        command[[5,6]] += val

    if 'w' in keys:
        command[4:8] += val*10
    if 's' in keys:
        command[4:8] -= val
    if 'a' in keys:
        command[[4,6]] += val
        command[[5,7]] -= val
    if 'd' in keys:
        command[[4,6]] -= val
        command[[5,7]] += val

    return command

class Keyframe:
    def __init__(self, image):
        self.image = image
        self.keypoints = None
        self.brief_descriptors = None
        self.brisk_param = {'thres': 20, 'octaves': 0}

    def computeBRIEFPoint(self):
        brisk_detector = cv2.BRISK_create(thresh=self.brisk_param['thres'], octaves=self.brisk_param['octaves'])
        # brisk_detector = cv2.BRISK_create()
        self.keypoints, self.brief_descriptors = brisk_detector.detectAndCompute(self.image, None)



class ImagingSonar:
    def __init__(self, scenario="Dam-HoveringCamera"):
        config = holoocean.packagemanager.get_scenario(scenario)
        config = config['agents'][0]['sensors'][-1]["configuration"]
        self.azi = config['Azimuth']
        self.minR = config['RangeMin']
        self.maxR = config['RangeMax']
        self.binsR = config['RangeBins']
        self.binsA = config['AzimuthBins']

    def get_plot_ready(self):
        plt.ion()
        self.fig, self.ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8, 5))
        self.ax.set_theta_zero_location("N")
        self.ax.set_thetamin(-self.azi / 2)
        self.ax.set_thetamax(self.azi / 2)
        self.theta = np.linspace(-self.azi/2,self.azi/2,self.binsA)*np.pi/180
        self.r = np.linspace(self.minR, self.maxR, self.binsR)
        self.T, self.R = np.meshgrid(self.theta, self.r)
        self.z = np.zeros_like(self.T)
        plt.grid(False)
        self.plot = self.ax.pcolormesh(self.T, self.R, self.z, cmap='gray', shading='auto', vmin=0, vmax=1)
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    def sonar_filter(self):
        # 加入了噪声
        # matrix = (self.s * 255).astype(np.uint8)
        # matrix = cv2.medianBlur(matrix, 3)  # 中值滤波
        # otsu_threshold, _ = cv2.threshold(matrix, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # 动态阈值
        # _, self.s = cv2.threshold(matrix, 1 * otsu_threshold, 255, cv2.THRESH_BINARY)
        # self.s = self.s / 255
        # self.s[128:255, :] = 0
        # 不加入噪声
        self.s = (self.s * 255).astype(np.uint8)
        otsu_threshold, self.s = cv2.threshold(self.s, 1, 255, cv2.THRESH_BINARY)
        self.s = self.s / 255
    def  determine_borderline(self, state, pose, angle):
        if 'ImagingSonar' in state:
            # 加入了噪声
            # self.s[128:255,:] = 0
            x_coordinates, y_coordinates = np.where(self.s)
            self.distance = x_coordinates/self.binsR*(self.maxR-self.minR)+self.minR
            self.angle = -self.azi / 2 +y_coordinates/self.binsA*(self.azi)
            self.x_world = pose[0] + self.distance*np.cos((angle+self.angle)*np.pi/180)
            self.y_world = pose[1] + self.distance*np.sin((angle+self.angle)*np.pi/180)
            # self.borderline_world = list(zip(x_coordinates, y_coordinates))

            ax1.scatter(self.x_world, self.y_world)
            ax1.scatter(pose[0],pose[1],color='red')
            plt.show()

    def draw_pic(self, state):
        if 'ImagingSonar' in state:
            self.s = state['ImagingSonar']
            self.sonar_filter()
            self.plot.set_array(self.s.ravel())
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

class PoseCalculate:
    def __init__(self):
        self.location = np.zeros(3)
        self.locationxy = np.zeros(2)
        self.direction = 0
    def pose_calculate(self,state):
        self.s = state['PoseSensor']
        self.location = np.array([self.s[0][3], self.s[1][3], self.s[2][3]])
        self.truelocation = state['LocationSensor']
        self.locationxy = self.location[0:2]
        self.truelocationxy = self.location[0:2]
        self.angle = np.degrees(np.arctan2(self.s[1, 0], self.s[0, 0]))
        if self.angle < 0:
            self.angle = 360 + self.angle
        self.direction = self.angle*np.pi/180


with holoocean.make("Dam-HoveringCamera") as env:
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    # env.spawn_prop(prop_type='cylinder', location=[31.5, 95, -37], scale=[2,2,2], sim_physics=0, material='steel')
    # env.draw_box(center=[31.5, 95, -37],extent=[1,1,1],lifetime=0)
    # env.weather.stop_day_cycle()
    imagingsonar = ImagingSonar()
    posecalculate = PoseCalculate()
    imagingsonar.get_plot_ready()
    showpicture = False
    while True:
        if 'q' in pressed_keys:
            break
        if 'p' in pressed_keys:
           if showpicture==True: showpicture=False
        if 'o' in pressed_keys:
           if showpicture==False: showpicture=True

        command = parse_keys(pressed_keys, force)
        #send to holoocean
        env.act("auv0", command)
        state = env.tick()
        feature_extractor(state, showpicture)
        imagingsonar.draw_pic(state)
        posecalculate.pose_calculate(state)
        imagingsonar.determine_borderline(state, posecalculate.location, posecalculate.angle)