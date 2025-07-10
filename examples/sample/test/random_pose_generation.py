import numpy as np

def generate_random_poses(agent_pos = None, map_range=(-500, 500), fix_depth=-45.0, sensor_distance=5.0, 
                          sensor_fov=120.0, safety_margin=10.0):
    """
    生成随机的agent和target初始位姿，确保target在agent的FOV内
    
    参数:
        map_range: 地图x,y范围的元组 (min, max)
        fix_depth: 固定的z坐标深度
        sensor_distance: agent的传感器最大距离
        sensor_fov: agent的视场角度（度）
        safety_margin: 与地图边界的安全距离
        
    返回:
        agent_pos: agent的初始位置 [x, y, z]
        agent_yaw: agent的初始朝向（弧度）
        target_pos: target的初始位置 [x, y, z]
        target_yaw: target的初始朝向（弧度）
    """
    map_min, map_max = map_range
    effective_range = (map_min + safety_margin, map_max - safety_margin)
    
    # 1. 随机生成agent位置，保持安全距离
    if agent_pos is None:
        agent_x = np.random.uniform(effective_range[0], effective_range[1])
        agent_y = np.random.uniform(effective_range[0], effective_range[1])
        agent_pos = np.array([agent_x, agent_y, fix_depth])
    else:
        agent_x = agent_pos[0]
        agent_y = agent_pos[1]
        agent_pos = np.array(agent_pos)
        
    # 2. 随机生成agent朝向
    agent_yaw = np.random.uniform(-np.pi, np.pi)  # 弧度，范围[-π, π]
    
    # 3. 在agent的FOV内随机生成target
    half_fov = np.radians(sensor_fov/2)  # 半视场角（弧度）
    target_angle = np.random.uniform(-half_fov, half_fov)  # 相对于agent朝向的角度
    target_distance = np.random.uniform(1.0, sensor_distance)  # 避免距离太近，最小距离设为1.0
    
    # 计算target的全局角度
    target_global_angle = agent_yaw + target_angle
    
    # 计算target位置（极坐标转笛卡尔坐标）
    target_x = agent_x + target_distance * np.cos(target_global_angle)
    target_y = agent_y + target_distance * np.sin(target_global_angle)
    
    # 确保target在地图范围内
    target_x = np.clip(target_x, effective_range[0], effective_range[1])
    target_y = np.clip(target_y, effective_range[0], effective_range[1])
    target_pos = np.array([target_x, target_y, fix_depth])
    
    # 4. 随机生成target朝向
    target_yaw = np.random.uniform(-np.pi, np.pi)  # 弧度，范围[-π, π]
    
    # 检验target是否确实在agent的FOV内
    dx = target_x - agent_x
    dy = target_y - agent_y
    actual_distance = np.sqrt(dx**2 + dy**2)
    actual_angle = np.arctan2(dy, dx) - agent_yaw
    # 标准化角度到[-π, π]
    actual_angle = np.arctan2(np.sin(actual_angle), np.cos(actual_angle))
    
    # 如果条件不满足，递归重试
    if actual_distance > sensor_distance or abs(actual_angle) > half_fov:
        return generate_random_poses(map_range, fix_depth, sensor_distance, sensor_fov, safety_margin)
    
    # 打印调试信息
    print(f"生成的位姿满足条件:")
    print(f"Agent位置: [{agent_x:.8f}, {agent_y:.8f}, {fix_depth:.1f}], 朝向: {agent_yaw:.16f}")
    print(f"Target位置: [{target_x:.8f}, {target_y:.8f}, {fix_depth:.1f}], 朝向: {target_yaw:.16f}")
    print(f"距离: {actual_distance:.2f}, 角度: {np.degrees(actual_angle):.2f}°")
    
    return agent_pos, agent_yaw, target_pos, target_yaw

# 使用示例 - 生成10组随机位姿
for i in range(10):
    print(f"\n===== 随机位姿 #{i+1} =====")
    agent_pos = np.array([50, 0, -45.0])  # 固定agent位置
    agent_pos, agent_yaw, target_pos, target_yaw = generate_random_poses(
        agent_pos=agent_pos,
        map_range=(-500, 500),  # 可以根据需要调整地图范围
        fix_depth=-45.0,        # 固定深度
        sensor_distance=8.0,    # 传感器最大距离
        sensor_fov=120.0        # 视场角度
    )
    
    # 生成可直接复制到代码中的格式
    print("\n代码格式:")
    print(f"[{agent_pos[0]:.8f}, {agent_pos[1]:.8f}, {agent_pos[2]:.1f}]")
    print(f"{agent_yaw}")
    print(f"[{target_pos[0]:.8f}, {target_pos[1]:.8f}, {target_pos[2]:.1f}]")
    print(f"{target_yaw}")