import numpy as np
import time

class CmdVel:
    def __init__(self):
        self.linear = type('', (), {'x': 0.0, 'y': 0.0, 'z': 0.0})()
        self.angular = type('', (), {'x': 0.0, 'y': 0.0, 'z': 0.0})()

# base NED coordinates, use negative for util
class PID:
    # basically for HoveringROV
    def __init__(self, robo_type="HoveringAUV"):
        # ----------- Underwater Robot Parameters -----------#
        self.gravity = 9.81
        self.cob = np.array([0, 0, 5.0]) / 100
        self.rho = 997

        # Different robot parameters
        if robo_type == "HoveringAUV":
            self.m = 31.02
            self.thruster_p = np.array([[18.18, -22.14, -4],
                                    [18.18, 22.14, -4],
                                    [-31.43, 22.14, -4],
                                    [-31.43, -22.14, -4],
                                    [7.39, -18.23, -0.21],
                                    [7.39, 18.23, -0.21],
                                    [-20.64, 18.23, -0.21],
                                    [-20.64, -18.23, -0.21]]) / 100
            self.J = np.eye(3) * 2
            # PID gains for HoveringAUV
            self.Kp_lin_x = 500
            self.Ki_lin_x = 1.0
            self.Kd_lin_x = 0.5
            self.Kp_ang_z = 30
            self.Ki_ang_z = 0.2
            self.Kd_ang_z = 0.8
            self.Kp_depth = 10.0
            # Z direction (vertical) PID parameters
            self.Kp_lin_z = 10
            self.Ki_lin_z = 0.5
            self.Kd_lin_z = 0.3

        elif robo_type == "BlueROV2":
            self.m = 11.5
            self.thruster_p = np.array([
                [18.18, -22.14, -4],
                [18.18, 22.14, -4], 
                [-31.43, 22.14, -4],
                [-31.43, -22.14, -4],
                [7.39, -18.23, -0.21],
                [7.39, 18.23, -0.21],
                [-20.64, 18.23, -0.21],
                [-20.64, -18.23, -0.21]
            ]) / 100
            self.J = np.eye(3) * 1.0
            # PID gains for BlueROV2
            self.Kp_lin_x = 150
            self.Ki_lin_x = 0.3
            self.Kd_lin_x = 0.1
            self.Kp_ang_z = 10
            self.Ki_ang_z = 0.05
            self.Kd_ang_z = 0.2
            # Z direction (vertical) PID parameters
            self.Kp_depth = 15.0
            self.Kp_lin_z = 100
            self.Ki_lin_z = 0.3
            self.Kd_lin_z = 0.1
        else:
            raise ValueError(f"Unknown robo_type: {robo_type}")
        
        self.V = self.m / self.rho # volume
        # Adjust thruster positions (relative to center of mass)
        self.com = (self.thruster_p[0] + self.thruster_p[2]) / 2
        self.com[2] = self.thruster_p[-1][2]
        self.thruster_p -= self.com

        # Thruster directions
        self.thruster_d = np.array([[0, 0, 1],
                                  [0, 0, 1],
                                  [0, 0, 1],
                                  [0, 0, 1],
                                  [np.sqrt(2), np.sqrt(2), 0],
                                  [np.sqrt(2), -np.sqrt(2), 0],
                                  [np.sqrt(2), np.sqrt(2), 0],
                                  [np.sqrt(2), -np.sqrt(2), 0]])

        # Thrust allocation matrix
        self.M = np.zeros((6, 8))
        for i in range(8):
            self.M[:3, i] = self.thruster_d[i]
            self.M[3:, i] = -np.cross(self.thruster_d[i], self.thruster_p[i])

        self.Minv = self.M.T @ np.linalg.inv(self.M @ self.M.T)

        # ----------- PID Control Parameters -----------#
        # Integral term limits to prevent integral windup
        self.lin_x_int_limit = 2.0
        self.lin_z_int_limit = 2.0
        self.ang_z_int_limit = 1.0
        
        # Error accumulation and previous error
        self.lin_x_error_sum = 0.0
        self.lin_z_error_sum = 0.0
        self.ang_z_error_sum = 0.0
        self.lin_x_last_error = 0.0
        self.lin_z_last_error = 0.0
        self.ang_z_last_error = 0.0
        
        # Depth control PID parameters (optional, for maintaining constant depth)
        self.depth_target = None  # Initial depth target is None
        
    def set_depth_target(self, depth):
        """Set depth holding target"""
        self.depth_target = depth
        
    def reset(self):
        """Reset PID controller state"""
        self.lin_x_error_sum = 0.0
        self.lin_z_error_sum = 0.0
        self.ang_z_error_sum = 0.0
        self.lin_x_last_error = 0.0
        self.lin_z_last_error = 0.0
        self.ang_z_last_error = 0.0
        self.depth_target = None
        
    def compute_control(self, current_state, cmd_vel):
        """
        Calculate PID control output based on cmd_vel
        
        Parameters:
        - current_state: Current state (including position, velocity, attitude, etc.)
        - cmd_vel: Target velocity containing linear.x and angular.z
        
        Returns:
        - Thruster control output (forces for 8 thrusters)
        """
        # Extract current velocity state
        body_vel = current_state.body_velocity
        body_ang_vel = current_state.body_angular_velocity
        current_lin_x = body_vel[0]
        current_lin_z = body_vel[2]
        current_ang_z = body_ang_vel[2]

        # Extract target velocity
        target_lin_x = cmd_vel.linear.x
        target_lin_z = cmd_vel.linear.z
        target_ang_z = cmd_vel.angular.z
        
        # Calculate time interval
        dt = 0.01
            
        # Calculate linear velocity error
        lin_x_error = target_lin_x - current_lin_x
        lin_z_error = target_lin_z - current_lin_z
        
        # X direction PID control
        # if abs(lin_x_error) < 0.01:
        #     lin_x_error = 0
            # self.lin_x_error_sum = 0
        # Calculate integral term
        self.lin_x_error_sum += lin_x_error * dt
        # Limit integral term to prevent integral windup
        self.lin_x_error_sum = np.clip(self.lin_x_error_sum, -self.lin_x_int_limit, self.lin_x_int_limit)
        # Calculate derivative term
        lin_x_error_diff = (lin_x_error - self.lin_x_last_error) / dt
        self.lin_x_last_error = lin_x_error
        # Calculate forward PID output
        lin_x_pid_output = (self.Kp_lin_x * lin_x_error + 
                          self.Ki_lin_x * self.lin_x_error_sum + 
                          self.Kd_lin_x * lin_x_error_diff)
        
        # Z direction PID control
        if abs(lin_z_error) < 0.005:  # Reduce dead zone
            lin_z_error = 0
            self.lin_z_error_sum = 0
        # Calculate integral term
        self.lin_z_error_sum += lin_z_error * dt
        # Limit integral term to prevent integral windup
        self.lin_z_error_sum = np.clip(self.lin_z_error_sum, -self.lin_z_int_limit, self.lin_z_int_limit)
        # Calculate derivative term
        lin_z_error_diff = (lin_z_error - self.lin_z_last_error) / dt
        self.lin_z_last_error = lin_z_error
        # Calculate vertical PID output
        lin_z_pid_output = (self.Kp_lin_z * lin_z_error + 
                          self.Ki_lin_z * self.lin_z_error_sum + 
                          self.Kd_lin_z * lin_z_error_diff)
        
        # Calculate angular velocity error
        ang_z_error = target_ang_z - current_ang_z
        if abs(ang_z_error) < 0.005:
            ang_z_error = 0
            self.ang_z_error_sum = 0
        # Calculate integral term
        self.ang_z_error_sum += ang_z_error * dt
        # Limit integral term to prevent integral windup
        self.ang_z_error_sum = np.clip(self.ang_z_error_sum, -self.ang_z_int_limit, self.ang_z_int_limit)
        # Calculate derivative term
        ang_z_error_diff = (ang_z_error - self.ang_z_last_error) / dt
        self.ang_z_last_error = ang_z_error
        # Calculate angular velocity PID output
        ang_z_pid_output = (self.Kp_ang_z * ang_z_error + 
                          self.Ki_ang_z * self.ang_z_error_sum + 
                          self.Kd_ang_z * ang_z_error_diff)
        
        # Create six-dimensional control vector [Fx, Fy, Fz, Tx, Ty, Tz] (NED coordinate system)
        u_til = np.zeros(6)
        u_til[0] = lin_x_pid_output  # X direction force
        u_til[5] = -ang_z_pid_output  # Z direction torque
        
        if self.depth_target is not None:
            current_depth = current_state.vec[2]
            depth_error = self.depth_target - current_depth
            u_til[2] = self.Kp_depth * depth_error
        else:
            u_til[2] = -lin_z_pid_output
        
        # Compensate buoyancy torque (if needed)
        # Get rotation matrix from state
        rotation_matrix = current_state.mat[:3, :3]
        u_til[3:] += np.cross(rotation_matrix.T @ np.array([0, 0, 1]),
                            self.cob) * self.V * self.rho * self.gravity
        
        # Convert forces to body coordinate system
        # u_til[:3] = rotation_matrix.T @ u_til[:3]
        
        # Convert torques to thruster output
        thruster_forces = self.Minv @ u_til
        
        return thruster_forces
        
    def u(self, x, cmd_vel):
        """
        Method compatible with the original LQR interface
        
        Parameters:
        - x: Current state
        - cmd_vel: Target velocity object containing linear.x and angular.z
        
        Returns:
        - Thruster control output
        """
        return self.compute_control(x, cmd_vel)