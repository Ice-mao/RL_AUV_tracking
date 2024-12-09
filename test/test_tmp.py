import holoocean
import numpy as np
from auv_control import scenario
from auv_env.envs.tools import KeyBoardCmd
from auv_env.envs.obstacle import Obstacle

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
env.draw_line([0, 0,-5], [10, 10, -5], thickness=5.0, lifetime=0.0)
# np.tan(np.radians(-120))
# env.spawn_prop(prop_type="box", scale=[10, 2, 1], location=[10, -10, -12], rotation=[0, 0, 0.0], material='gold')
# env.spawn_prop(prop_type="box", location=[10.5, 0, -12], material='gold')
env.agents['auv0'].set_physics_state(location=[-20.41332375,  -0.3027992  , -5.        ],
                                    rotation=[0.0, 0.0, np.rad2deg(1.4958537957633062)],
                                    velocity=[0.0, 0.0, 0.0],
                                    angular_velocity=[0.0, 0.0, 0.0])
env.agents['target'].set_physics_state(location=[-19.65225727 ,  7.03218074 , -5.        ],
                                              rotation=[0.0, 0.0, np.rad2deg(1.467407793510139)],
                                              velocity=[0.0, 0.0, 0.0],
                                              angular_velocity=[0.0, 0.0, 0.0])
from auv_control.state import rot_to_rpy
for _ in range(20000000):
    if 'q' in kb_cmd.pressed_keys:
        break
    command = kb_cmd.parse_keys()
    env.act("auv0", command)

    target_action = (0, 0)
    env.act("target", target_action)
    state = env.tick()

    current_time = state['t']
    pose = rot_to_rpy(state['target']['PoseSensor'][:3, :3])
    yaw = np.radians(pose[2])
    print(yaw)

    env.agents['auv0'].teleport(location=[state['t'], 0, -5],
                                rotation=[0.0, 0.0, 0.0])
    # if current_time - last_control_time >= 1.0:
    #     last_control_time = current_time
    #     target_action = np.random.randint(0, 3)

