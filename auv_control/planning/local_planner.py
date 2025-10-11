"""
Trajectory Planner for AUV Control
Generates smooth trajectories from single waypoints/knots for LQR control
"""

import numpy as np
from typing import List, Tuple, Union


class TrajectoryPlanner:  
    def __init__(self, control_dt: float = 0.02, planning_duration: float = 0.25):
        """
        Args:
            control_dt: Time step corresponding to control frequency (50Hz = 0.02s)
            planning_duration: Execution time for each RL step (0.25s)
        """
        self.control_dt = control_dt
        self.planning_duration = planning_duration
        self.num_steps = int(self.planning_duration / self.control_dt)
        
    def generate_trajectory_to_knot(self, current_state: np.ndarray, target_knot: np.ndarray) -> np.ndarray:
        """
        Generate smooth trajectory from current state to target knot (supports 2D and 3D)
        
        Args:
            current_state: np.array([x, y, yaw]) for 2D or np.array([x, y, z, yaw]) for 3D
            target_knot: np.array([x, y, yaw]) for 2D or np.array([x, y, z, yaw]) for 3D
            
        Returns:
            np.ndarray: shape (num_steps, 3) for 2D or (num_steps, 4) for 3D
                       [[x, y, yaw], ...] or [[x, y, z, yaw], ...]
        """
        # Automatically detect 2D or 3D
        is_3d = len(current_state) == 4 and len(target_knot) == 4
        state_dim = 4 if is_3d else 3
        
        trajectory = np.zeros((self.num_steps, state_dim))
        
        for i in range(self.num_steps):
            t = i / max(1, self.num_steps - 1)  # Interpolation parameter from 0 to 1
            
            if is_3d:
                # 3D version: Linear interpolation for position (x, y, z)
                trajectory[i, :3] = current_state[:3] + t * (target_knot[:3] - current_state[:3])
                # Angle interpolation (handle angle continuity)
                yaw_diff = self._wrap_angle(target_knot[3] - current_state[3])
                trajectory[i, 3] = self._wrap_angle(current_state[3] + t * yaw_diff)
            else:
                # 2D version: Linear interpolation for position (x, y)
                trajectory[i, :2] = current_state[:2] + t * (target_knot[:2] - current_state[:2])
                # Angle interpolation (handle angle continuity)
                yaw_diff = self._wrap_angle(target_knot[2] - current_state[2])
                trajectory[i, 2] = self._wrap_angle(current_state[2] + t * yaw_diff)
            
        return trajectory
    
    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-π, π] range"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class TrajectoryBuffer:   
    def __init__(self):
        self.trajectory: np.ndarray = np.array([])
        self.current_index: int = 0
        self.is_3d: bool = False
        
    def update_trajectory(self, new_trajectory: np.ndarray):
        """
        Args:
            new_trajectory: np.ndarray shape (num_steps, 3) for 2D or (num_steps, 4) for 3D
        """
        self.trajectory = new_trajectory
        self.current_index = 0
        # Automatically detect trajectory dimensions
        self.is_3d = new_trajectory.shape[1] == 4 if len(new_trajectory) > 0 else False
        
    def get_current_waypoint(self) -> Union[np.ndarray, None]:
        """
        Returns:
            np.ndarray: [x, y, yaw] for 2D or [x, y, z, yaw] for 3D, or None if trajectory is empty
        """
        if self.current_index < len(self.trajectory):
            waypoint = self.trajectory[self.current_index]
            self.current_index += 1
            return waypoint
        elif len(self.trajectory) > 0:
            # Trajectory finished, return the last point
            return self.trajectory[-1]
        else:
            return None
            
    def is_empty(self) -> bool:
        return len(self.trajectory) == 0 or self.current_index >= len(self.trajectory)
        
    def get_trajectory_info(self) -> dict:
        """
        Get trajectory information for debugging
        """
        return {
            'is_3d': self.is_3d,
            'total_points': len(self.trajectory) if len(self.trajectory) > 0 else 0,
            'current_index': self.current_index,
            'remaining_points': max(0, len(self.trajectory) - self.current_index) if len(self.trajectory) > 0 else 0
        }

