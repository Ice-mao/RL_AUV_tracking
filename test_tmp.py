import holoocean, cv2
import numpy as np
from auv_control import scenario
from auv_env.tools import KeyBoardCmd
from pynput import keyboard
from auv_env.obstacle import Obstacle
from pynput import keyboard

# env = holoocean.make(scenario_cfg=scenario)
env = holoocean.make('PierHarbor-Hovering')
state = env.tick()
current_time = state['t']
last_control_time = current_time
target_action = np.random.randint(0, 3)

kb_cmd = KeyBoardCmd(force=15)
obstacles = Obstacle(env, -12)
obstacles.draw_obstacle()
# env.spawn_prop(prop_type="box", scale=[3, 2, 1], location=[10, 0, -12], rotation=[np.tan(np.radians(-120)), 1, 0.0], material='gold')
# env.spawn_prop(prop_type="box", location=[10.5, 0, -12], material='gold')

for _ in range(20000):
    if 'q' in kb_cmd.pressed_keys:
        break
    command = kb_cmd.parse_keys()
    env.act("auv0", command)

    target_action = np.random.randint(0, 3)
    # env.act("target", target_action)
    state = env.tick()

    current_time = state['t']
    # if current_time - last_control_time >= 1.0:
    #     last_control_time = current_time
    #     target_action = np.random.randint(0, 3)

