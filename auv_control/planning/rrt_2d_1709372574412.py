import numpy as np

# 初始化agent的初始状态（位置和角度）
position_init = np.array([0, 0])  # 初始位置
angle_init = 0.0  # 初始角度

# 定义路径点列表 (每个点包含位置坐标(x, y) 和期望到达的角度 theta)
path_points = [
    {"position": np.array([10, 0]), "angle": np.pi / 2},
    {"position": np.array([10, 10]), "angle": 0},
    # 更多路径点...
]

# 设定速度和角速度最大值
v_max = 5.0  # 最大线速度
w_max = 1.0  # 最大角速度

def generate_commands(path_points, current_state):
    commands = []
    for point in path_points:
        # 根据当前位置和目标位置计算速度和角速度指令
        position_target = point["position"]
        angle_target = point["angle"]

        # 这里仅作演示，实际中应根据动力学模型和控制算法（如PID、轨迹规划等）来计算速度指令
        distance = np.linalg.norm(position_target - current_state["position"])
        time_to_reach = distance / v_max  # 简化处理，假设匀速直线运动到目标点
        linear_velocity = min(distance / time_to_reach, v_max)
        angular_velocity = (angle_target - current_state["angle"]) / time_to_reach

        # 对角速度进行限制
        angular_velocity = np.clip(angular_velocity, -w_max, w_max)

        commands.append({"linear_velocity": linear_velocity, "angular_velocity": angular_velocity})

        # 更新当前状态（这里简化处理，实际情况需要考虑时间步进和动力学模型）
        current_state["position"] = position_target
        current_state["angle"] = angle_target

    return commands

# 初始化agent的当前状态
current_state = {"position": position_init, "angle": angle_init}

# 生成速度和角速度指令
commands = generate_commands(path_points, current_state)

# 输出或应用这些指令以控制agent的运动
for command in commands:
    print(f"Linear Velocity: {command['linear_velocity']}, Angular Velocity: {command['angular_velocity']}")
