import numpy as np

class SE2PIDController:
    def __init__(self):
        self.kp_linear = 0.015
        self.ki_linear = 0.0
        self.kd_linear = 0.0

        self.kp_angular = 0.8
        self.ki_angular = 0.0
        self.kd_angular = 0.001

        self.windup_guard = 0.001

        self.prev_linear_error = [0, 0]  # Previous linear error (x, y)
        self.prev_angular_error = 0        # Previous angular error (theta)
        self.integral_linear_error = [0, 0]  # Integral of linear error (x, y)
        self.integral_angular_error = 0       # Integral of angular error (theta)
    def reset(self):
        self.prev_linear_error = [0, 0]
        self.prev_angular_error = 0
        self.integral_linear_error = [0, 0]
        self.integral_angular_error = 0

    def u(self, current_pose, target_pose, dt):
        # Compute linear error

        linear_error = [target_pose[0] - current_pose[0],  # Error in x
                        target_pose[1] - current_pose[1]]  # Error in y

        # Compute angular error (orientation error)
        theta = np.arctan2(linear_error[1], linear_error[0])
        # if linear_error[0] < 0 and linear_error[1] < 0:
        #     angular_error = np.pi - theta - current_pose[2]
        # elif linear_error[0] < 0 and linear_error[1] > 0:
        #     angular_error = -np.pi - theta - current_pose[2]
        # else:
        angular_error = - theta - current_pose[2]
        # an easy but annoying bug, I have fix it.PID  never lies
        if angular_error < -np.pi:
            angular_error = angular_error + 2 * np.pi
        elif angular_error > np.pi:
            angular_error = angular_error - 2 * np.pi

        # angular_error = wrap_around(angular_error)
        self.integral_angular_error += angular_error * dt
        if (self.integral_angular_error < -self.windup_guard):
            self.integral_angular_error = -self.windup_guard
        elif (self.integral_angular_error > self.windup_guard):
            self.integral_angular_error = self.windup_guard

        # Compute derivative of angular error
        angular_derivative = (angular_error - self.prev_angular_error) / dt
        self.prev_angular_error = angular_error

        angular_control = self.kp_angular * angular_error + self.ki_angular * self.integral_angular_error + self.kd_angular * angular_derivative
        angular_control = max(min(angular_control, 0.8), -0.8)

        # Update integral error
        self.integral_linear_error[0] += linear_error[0] * dt
        self.integral_linear_error[1] += linear_error[1] * dt

        # Compute derivative of linear error
        linear_derivative = [(linear_error[0] - self.prev_linear_error[0]) / dt,
                             (linear_error[1] - self.prev_linear_error[1]) / dt]

        # Update previous errors
        self.prev_linear_error = linear_error

        # Compute control commands
        dis = np.sqrt(linear_error[0] ** 2 + linear_error[1] ** 2)
        if np.abs(angular_error) > 1.0:
            v = 0.001
        elif dis >= 3:
            v = 0.001
        else:
            v = self.kp_linear * dis + 0.003 * np.random.normal(1, 0.5)
        return v, angular_control

def wrap_around(x):
    # x \in [-pi,pi)
    if x >= np.pi:
        return x - 2 * np.pi
    elif x < -np.pi:
        return x + 2 * np.pi
    else:
        return x