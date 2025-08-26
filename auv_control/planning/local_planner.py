"""
Trajectory Planner for AUV Control
Generates smooth trajectories from single waypoints/knots for LQR control
"""

import numpy as np
from typing import List, Tuple


class TrajectoryPlanner:  
    def __init__(self, control_dt: float = 0.02, planning_duration: float = 0.25):
        """
        Args:
            control_dt: 控制频率对应的时间步长 (50Hz = 0.02s)
            planning_duration: 每个RL step的执行时间 (0.25s)
        """
        self.control_dt = control_dt
        self.planning_duration = planning_duration
        self.num_steps = int(self.planning_duration / self.control_dt)
        
    def generate_trajectory_to_knot(self, current_state: List[float], target_knot: List[float]) -> List[List[float]]:
        """
        从当前状态生成到目标knot的平滑轨迹
        
        Args:
            current_state: [x, y, yaw] 当前状态 (yaw in radians)
            target_knot: [x, y, yaw] 目标状态 (yaw in radians)
            
        Returns:
            List of waypoints: [[x, y, yaw], ...] for the planning duration
        """
        trajectory = []
        
        for i in range(self.num_steps):
            t = i / max(1, self.num_steps - 1)  # 从0到1的插值参数
            
            # 位置线性插值
            x = current_state[0] + t * (target_knot[0] - current_state[0])
            y = current_state[1] + t * (target_knot[1] - current_state[1])
            
            # 角度插值（处理角度连续性）
            yaw_diff = self._wrap_angle(target_knot[2] - current_state[2])
            yaw = current_state[2] + t * yaw_diff
            yaw = self._wrap_angle(yaw)
            
            trajectory.append([x, y, yaw])
            
        return trajectory
    
    def _wrap_angle(self, angle: float) -> float:
        """将角度包装到[-π, π]范围内"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class TrajectoryBuffer:   
    def __init__(self):
        self.trajectory = []
        self.current_index = 0
        
    def update_trajectory(self, new_trajectory: List[List[float]]):
        self.trajectory = new_trajectory
        self.current_index = 0
        
    def get_current_waypoint(self) -> List[float]:
        if self.current_index < len(self.trajectory):
            waypoint = self.trajectory[self.current_index]
            self.current_index += 1
            return waypoint
        elif len(self.trajectory) > 0:
            # 轨迹结束，返回最后一个点
            return self.trajectory[-1]
        else:
            return None
            
    def is_empty(self) -> bool:
        return len(self.trajectory) == 0 or self.current_index >= len(self.trajectory)

