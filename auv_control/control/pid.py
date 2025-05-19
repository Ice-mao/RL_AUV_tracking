import numpy as np
import time

class CmdVel:
    def __init__(self):
        self.linear = type('', (), {'x': 0.0, 'y': 0.0, 'z': 0.0})()
        self.angular = type('', (), {'x': 0.0, 'y': 0.0, 'z': 0.0})()

class PID:
    def __init__(self):
        # ----------- 水下机器人参数 -----------#
        self.gravity = 9.81
        self.cob = np.array([0, 0, 5.0]) / 100
        self.m = 31.02
        self.rho = 997
        self.V = self.m / self.rho
        self.J = np.eye(3) * 2

        # 推进器位置配置
        self.thruster_p = np.array([[18.18, -22.14, -4],
                                   [18.18, 22.14, -4],
                                   [-31.43, 22.14, -4],
                                   [-31.43, -22.14, -4],
                                   [7.39, -18.23, -0.21],
                                   [7.39, 18.23, -0.21],
                                   [-20.64, 18.23, -0.21],
                                   [-20.64, -18.23, -0.21]]) / 100

        # 调整推进器位置（相对于质心）
        self.com = (self.thruster_p[0] + self.thruster_p[2]) / 2
        self.com[2] = self.thruster_p[-1][2]
        self.thruster_p -= self.com

        # 推进器方向
        self.thruster_d = np.array([[0, 0, 1],
                                  [0, 0, 1],
                                  [0, 0, 1],
                                  [0, 0, 1],
                                  [np.sqrt(2), np.sqrt(2), 0],
                                  [np.sqrt(2), -np.sqrt(2), 0],
                                  [np.sqrt(2), np.sqrt(2), 0],
                                  [np.sqrt(2), -np.sqrt(2), 0]])

        # 推力分配矩阵
        self.M = np.zeros((6, 8))
        for i in range(8):
            self.M[:3, i] = self.thruster_d[i]
            self.M[3:, i] = -np.cross(self.thruster_d[i], self.thruster_p[i])

        self.Minv = self.M.T @ np.linalg.inv(self.M @ self.M.T)

        # ----------- PID控制参数 -----------#
        # 前向速度(linear.x)的PID参数
        self.Kp_lin_x = 500  # 比例增益
        self.Ki_lin_x = 1.0  # 积分增益
        self.Kd_lin_x = 0.5  # 微分增益
        
        # 角速度(angular.z)的PID参数
        self.Kp_ang_z = 100  # 比例增益
        self.Ki_ang_z = 1.0  # 积分增益
        self.Kd_ang_z = 0.5  # 微分增益
        
        # 积分项上限，防止积分饱和
        self.lin_x_int_limit = 2.0
        self.ang_z_int_limit = 5.0
        
        # 误差累积和上一次误差
        self.lin_x_error_sum = 0.0
        self.ang_z_error_sum = 0.0
        self.lin_x_last_error = 0.0
        self.ang_z_last_error = 0.0
        
        # 深度控制PID参数（可选，用于保持恒定深度）
        self.Kp_depth = 10.0
        self.depth_target = None  # 初始深度目标为None
        
    def set_depth_target(self, depth):
        """设置深度保持目标"""
        self.depth_target = depth
        
    def reset(self):
        """重置PID控制器状态"""
        self.lin_x_error_sum = 0.0
        self.ang_z_error_sum = 0.0
        self.lin_x_last_error = 0.0
        self.ang_z_last_error = 0.0
        
    def compute_control(self, current_state, cmd_vel):
        """
        根据cmd_vel计算PID控制输出
        
        参数:
        - current_state: 当前状态 (包含位置、速度、姿态等)
        - cmd_vel: 包含linear.x和angular.z的目标速度
        
        返回:
        - 推进器控制输出 (8个推进器的力)
        """
        # 提取当前速度状态
        rotation_matrix = current_state.mat[:3, :3]
        world_vel = current_state.vec[3:6]  # 全局坐标系速度[vx, vy, vz]
        body_vel = rotation_matrix.T @ world_vel
        current_lin_x = body_vel[0]  # 前向速度 vx
        current_ang_z = current_state.vec[11]  # 绕Z轴角速度
        
        # 提取目标速度
        target_lin_x = cmd_vel.linear.x
        target_ang_z = cmd_vel.angular.z
        
        # 计算时间间隔
        dt = 0.01
            
        # 计算线速度误差
        lin_x_error = target_lin_x - current_lin_x
        # if abs(lin_x_error) < 0.01:
        #     lin_x_error = 0
            # self.lin_x_error_sum = 0
        # 计算积分项
        self.lin_x_error_sum += lin_x_error * dt
        # 限制积分项，防止积分饱和
        self.lin_x_error_sum = np.clip(self.lin_x_error_sum, -self.lin_x_int_limit, self.lin_x_int_limit)
        # 计算微分项
        lin_x_error_diff = (lin_x_error - self.lin_x_last_error) / dt
        self.lin_x_last_error = lin_x_error
        # 计算前向PID输出
        lin_x_pid_output = (self.Kp_lin_x * lin_x_error + 
                          self.Ki_lin_x * self.lin_x_error_sum + 
                          self.Kd_lin_x * lin_x_error_diff)
        
        # 计算角速度误差
        ang_z_error = target_ang_z - current_ang_z
        if abs(ang_z_error) < 0.005:
            ang_z_error = 0
            self.ang_z_error_sum = 0
        # 计算积分项
        self.ang_z_error_sum += ang_z_error * dt
        # 限制积分项，防止积分饱和
        self.ang_z_error_sum = np.clip(self.ang_z_error_sum, -self.ang_z_int_limit, self.ang_z_int_limit)
        # 计算微分项
        ang_z_error_diff = (ang_z_error - self.ang_z_last_error) / dt
        self.ang_z_last_error = ang_z_error
        # 计算角速度PID输出
        ang_z_pid_output = (self.Kp_ang_z * ang_z_error + 
                          self.Ki_ang_z * self.ang_z_error_sum + 
                          self.Kd_ang_z * ang_z_error_diff)
        
        # 创建六维控制向量 [Fx, Fy, Fz, Tx, Ty, Tz]
        u_til = np.zeros(6)
        u_til[0] = lin_x_pid_output  # X方向力 (前进/后退)
        u_til[5] = ang_z_pid_output  # Z方向力矩 (转向)
        
        # 如果需要保持深度
        if self.depth_target is not None:
            current_depth = current_state.vec[2]
            depth_error = self.depth_target - current_depth
            u_til[2] = self.Kp_depth * depth_error  # 简单P控制器用于深度
        
        # 补偿浮力力矩（如果需要）
        # 从状态中获取旋转矩阵
        rotation_matrix = current_state.mat[:3, :3]
        u_til[3:] += np.cross(rotation_matrix.T @ np.array([0, 0, 1]),
                            self.cob) * self.V * self.rho * self.gravity
        
        # 将力转换到机体坐标系
        # u_til[:3] = rotation_matrix.T @ u_til[:3]
        
        # 将力矩转换为推进器输出
        thruster_forces = self.Minv @ u_til
        
        return thruster_forces
        
    def u(self, x, cmd_vel):
        """
        与原LQR接口兼容的方法
        
        参数:
        - x: 当前状态
        - cmd_vel: 包含linear.x和angular.z的目标速度对象
        
        返回:
        - 推进器控制输出
        """
        return self.compute_control(x, cmd_vel)