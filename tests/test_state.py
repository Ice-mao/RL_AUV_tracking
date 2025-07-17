import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import holoocean

from auv_env.envs.tools import KeyBoardCmd
from auv_control import State
if __name__ == '__main__':

    ##################### Init section #####################

    # Init env
    scenario = "SimpleUnderwater-Bluerov2"
    env = holoocean.make(scenario)
    # Init keyboard control
    kb_cmd = KeyBoardCmd(force=5)
    
    env.spawn_prop(prop_type='box', location=[10, 10, -5], scale=[5,5,5],
                    material='steel', sim_physics=True)
    i = 0
    while True:
        i += 1
        if 'q' in kb_cmd.pressed_keys:
            break
        command = kb_cmd.parse_keys()

        env.act("auv0", command)
        state = env.tick()
        agent_state = State(state['auv0'])
        if i % 100 == 0:
            print(f"=== 步骤 {i} ===")
            print(f"位置 (HoloOcean): {agent_state.vec[:3]}")
            print(f"速度 (HoloOcean): {agent_state.vec[3:6]}")
            print(f"RPY角度: {agent_state.vec[6:9]}")
            
            # 分析旋转矩阵的含义
            R = agent_state.mat[:3,:3]
            print(f"旋转矩阵 (NED相对于HoloOcean):")
            print(f"  NED_X(前): {R[:, 0]} -> HoloOcean坐标系中的方向")
            print(f"  NED_Y(右): {R[:, 1]} -> HoloOcean坐标系中的方向") 
            print(f"  NED_Z(下): {R[:, 2]} -> HoloOcean坐标系中的方向")
            
            # 使用新的属性方法
            body_vel = agent_state.body_velocity
            body_ang_vel = agent_state.body_angular_velocity
            
            print(f"机体速度 (NED): {body_vel}")
            print(f"机体角速度 (NED): {body_ang_vel}")
            print("-" * 50)
