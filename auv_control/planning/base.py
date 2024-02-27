import numpy as np
from auv_control import State

class BasePlanner:
    """Base class to handle visualization of trajectories.
    
    Variables to set when inheriting from this class:
        - pos_func
        - rot_func
    """
    def _traj(self, t):
        """Get desired trajectory at time t"""
        eps = 1e-5

        pos = self.pos_func(t)
        rot = self.rot_func(t)

        lin_vel = (self.pos_func(t+eps) - pos) / eps
        ang_vel = (self.rot_func(t+eps) - rot) / eps

        return np.hstack((pos, lin_vel, rot, ang_vel))

    def tick(self, t):
        """Gets desired trajectory at time t, only as a state"""
        if not isinstance(t, float):
            raise ValueError("Can't tick with an array")

        return State(self._traj(t))

    def draw_step(self, env, t, ts):
        """Draw points on the next steps"""
        des = self._traj(t)
        env.draw_point(des[:3].tolist(), color=[0,255,0], thickness=20, lifetime=ts)

    def draw_traj(self, env, t):
        """Makes trajectory line show up"""
        # Get all positions
        t = np.arange(0, t, 0.5)
        des_state = self._traj(t)
        des_pos = des_state[:,0:3]

        # Draw line between each
        for i in range(len(des_pos)-1):
            env.draw_line(des_pos[i].tolist(), des_pos[i+1].tolist(), thickness=5.0, lifetime=0.0)