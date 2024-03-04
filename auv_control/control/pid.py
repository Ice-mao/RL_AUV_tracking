import numpy as np
from auv_env.util import wrap_around
class SE2PIDController:
    def __init__(self):
        self.kp_linear = 0.008
        self.ki_linear = 0.0
        self.kd_linear = 0.0001

        self.kp_angular = 1.0
        self.ki_angular = 0.0
        self.kd_angular = 0.0

        self.prev_linear_error = [0, 0]  # Previous linear error (x, y)
        self.prev_angular_error = 0        # Previous angular error (theta)
        self.integral_linear_error = [0, 0]  # Integral of linear error (x, y)
        self.integral_angular_error = 0       # Integral of angular error (theta)
    def clear(self):
        self.prev_linear_error = [0, 0]
        self.prev_angular_error = 0
        self.integral_linear_error = [0, 0]
        self.integral_angular_error = 0

    # def u(self, current_pose, target_pose, dt):
    #     # Compute linear error
    #     linear_error = [target_pose[0] - current_pose[0],  # Error in x
    #                     target_pose[1] - current_pose[1]]  # Error in y
    #
    #     # Compute angular error (orientation error)
    #     angular_error = target_pose[2] - current_pose[2]
    #
    #     # Update integral error
    #     self.integral_linear_error[0] += linear_error[0] * dt
    #     self.integral_linear_error[1] += linear_error[1] * dt
    #     self.integral_angular_error += angular_error * dt
    #
    #     # Compute derivative of linear error
    #     linear_derivative = [(linear_error[0] - self.prev_linear_error[0]) / dt,
    #                          (linear_error[1] - self.prev_linear_error[1]) / dt]
    #
    #     # Compute derivative of angular error
    #     angular_derivative = (angular_error - self.prev_angular_error) / dt
    #
    #     # Update previous errors
    #     self.prev_linear_error = linear_error
    #     self.prev_angular_error = angular_error
    #
    #     # Compute control commands
    #     linear_control = [self.kp_linear * linear_error[0] + self.ki_linear * self.integral_linear_error[0] + self.kd_linear * linear_derivative[0],
    #                       self.kp_linear * linear_error[1] + self.ki_linear * self.integral_linear_error[1] + self.kd_linear * linear_derivative[1]]
    #     theta = np.arctan(linear_control[1] / linear_control[0])
    #     v = (linear_control[1] + linear_control[0]) / (np.sin(theta) + np.cos(theta))
    #     forward_speed = np.sqrt(linear_control[0] ** 2 + linear_control[1] ** 2)
    #     angular_control = self.kp_angular * angular_error + self.ki_angular * self.integral_angular_error + self.kd_angular * angular_derivative
    #
    #     return forward_speed, angular_control
    def u(self, current_pose, target_pose, dt):
        # Compute linear error

        linear_error = [target_pose[0] - current_pose[0],  # Error in x
                        target_pose[1] - current_pose[1]]  # Error in y

        # Update integral error
        self.integral_linear_error[0] += linear_error[0] * dt
        self.integral_linear_error[1] += linear_error[1] * dt

        # Compute derivative of linear error
        linear_derivative = [(linear_error[0] - self.prev_linear_error[0]) / dt,
                             (linear_error[1] - self.prev_linear_error[1]) / dt]

        # Update previous errors
        self.prev_linear_error = linear_error

        # Compute control commands
        linear_control = [self.kp_linear * linear_error[0] + self.ki_linear * self.integral_linear_error[0] + self.kd_linear * linear_derivative[0],
                          self.kp_linear * linear_error[1] + self.ki_linear * self.integral_linear_error[1] + self.kd_linear * linear_derivative[1]]
        # v = np.sqrt(linear_control[0] ** 2 + linear_control[1] ** 2)
        v = self.kp_linear * np.sqrt(linear_error[0] ** 2 + linear_error[1] ** 2)
        if v >= 0.1:
            v = 0.1

        # Compute angular error (orientation error)
        theta = np.arctan2(linear_error[1] , linear_error[0])
        # if linear_error[0] < 0 and linear_error[1] < 0:
        #     angular_error = np.pi - theta - current_pose[2]
        # elif linear_error[0] < 0 and linear_error[1] > 0:
        #     angular_error = -np.pi - theta - current_pose[2]
        # else:
        angular_error = - theta - current_pose[2]
        # angular_error = wrap_around(angular_error)
        self.integral_angular_error += angular_error * dt

        # Compute derivative of angular error
        angular_derivative = (angular_error - self.prev_angular_error) / dt
        self.prev_angular_error = angular_error

        angular_control = self.kp_angular * angular_error + self.ki_angular * self.integral_angular_error + self.kd_angular * angular_derivative

        return v, angular_control