import numpy as np
import matplotlib.pyplot as plt

# ...（上述代码段保持不变）

def save_to_file(commands, path_filename="path_data.csv"):
    # 将路径点和agent移动指令保存到CSV文件中
    with open(path_filename, 'w') as f:
        f.write("time,position_x,position_y,angle\n")
        time = 0.0
        for command in commands:
            linear_velocity = command["linear_velocity"]
            angular_velocity = command["angular_velocity"]

            # 简化处理：假设每个命令执行的时间间隔相同（实际应用中应根据控制系统的频率来计算）
            dt = 0.1  # 假设命令执行时间间隔为0.1秒
            position_delta = linear_velocity * dt * np.array([np.cos(current_state["angle"]), np.sin(current_state["angle"])])
            angle_delta = angular_velocity * dt

            current_state["position"] += position_delta
            current_state["angle"] = (current_state["angle"] + angle_delta) % (2 * np.pi)

            f.write(f"{time},{current_state['position'][0]},{current_state['position'][1]},{current_state['angle']}\n")
            time += dt

save_to_file(commands)

# 可视化部分
def visualize_path(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    time = data[:, 0]
    positions = data[:, 1:]
    
    fig, ax = plt.subplots()
    ax.plot(positions[:, 0], positions[:, 1], marker='o')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Agent Trajectory')

    # 绘制初始位置
    ax.plot(position_init[0], position_init[1], 'ro', label='Initial Position')
    
    # 如果你想显示每个路径点，可以添加如下代码：
    for point in path_points:
        ax.plot(point["position"][0], point["position"][1], 'go', markersize=5, label='Path Point')

    ax.legend()
    plt.show()

visualize_path("path_data.csv")
