import holoocean, cv2
import numpy as np
from auv_control import scenario
from auv_env.tools import KeyBoardCmd
from pynput import keyboard
from auv_env.obstacle import Obstacle
from pynput import keyboard

env = holoocean.make(scenario_cfg=scenario)
# env = holoocean.make('PierHarbor-Hovering')
state = env.tick()
current_time = state['t']
last_control_time = current_time
target_action = np.random.randint(0, 3)

kb_cmd = KeyBoardCmd(force=15)
obstacles = Obstacle(env, -12)
obstacles.draw_obstacle()
size = np.array([45, 45, 25])
bottom_corner = np.array([-22.5, -22.5, -25])
fix_depth = -5
center = bottom_corner + size / 2
env.draw_box(center.tolist(), (size / 2).tolist(), color=[0, 0, 255], thickness=30, lifetime=0) # draw the area
# np.tan(np.radians(-120))
# env.spawn_prop(prop_type="box", scale=[10, 2, 1], location=[10, -10, -12], rotation=[0, 0, 0.0], material='gold')
# env.spawn_prop(prop_type="box", location=[10.5, 0, -12], material='gold')

for _ in range(20000000):
    if 'q' in kb_cmd.pressed_keys:
        break
    command = kb_cmd.parse_keys()
    env.act("auv0", command)

    target_action = (0.01, 0.1)
    # env.act("target", target_action)
    state = env.tick()

    current_time = state['t']
    # if current_time - last_control_time >= 1.0:
    #     last_control_time = current_time
    #     target_action = np.random.randint(0, 3)

