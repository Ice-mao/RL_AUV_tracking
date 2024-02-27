import numpy as np
from scipy.linalg import solve_continuous_are

class LQR:
    def __init__(self):
        #----------- PARAMETERS OF AUV -----------#
        self.gravity = 9.81
        self.cob = np.array([0, 0, 5.0]) / 100
        self.m = 31.02
        self.rho = 997
        self.V = self.m / self.rho
        self.J = np.eye(3)*2

        # True thruster locations
        self.thruster_p = np.array([[18.18, -22.14, -4],
                                    [18.18, 22.14, -4],
                                    [-31.43, 22.14, -4], # First element should be -31.43
                                    [-31.43, -22.14, -4], # First element should be -31.43
                                    [7.39, -18.23, -0.21], # First element should be 7.39
                                    [7.39, 18.23, -0.21], # First element should be 7.39
                                    [-20.64, 18.23, -0.21],
                                    [-20.64, -18.23, -0.21]])/100

        # We offset them by our tweaked COM
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
        
        self.M = np.zeros((6,8))
        for i in range(8):
            self.M[:3,i] = self.thruster_d[i]
            self.M[3:,i] = -np.cross(self.thruster_d[i], self.thruster_p[i])

        self.Minv = self.M.T@np.linalg.inv(self.M@self.M.T)


        #----------- PARAMETERS FOR LQR -----------#
        self.v_damp = -np.eye(3)*1
        self.w_damp = -np.eye(3)*1

        self.A = np.zeros((12,12))
        self.A[0:3,3:6] = np.eye(3)
        self.A[3:6,3:6] = self.v_damp
        self.A[6:9,9:12] = np.eye(3)
        self.A[9:12,9:12] = self.w_damp
        
        self.B = np.zeros((12,6))
        self.B[3:6,0:3] = np.eye(3) / self.m
        self.B[9:12,3:6] = np.linalg.inv(self.J)

        self.Q = np.zeros(12)
        self.Q[0:3] = 100 # position
        self.Q[3:6] = 0.001 # velocity
        self.Q[6:9] = [.01, .01, .01] #0.01 # rotation
        self.Q[9:12] = 0.01 # angular velocity
        self.Q = np.diag(self.Q)

        self.R = np.zeros(6)
        self.R[0:3] = .01 # force
        self.R[3:6] = 1 # torque
        self.R = np.diag(self.R)

        self.P = solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R)@self.B.T@self.P


    def u(self, x, x_d):
        # wrap angles
        for i in range(6,9):
            x_d.vec[i] = wrap(x_d.vec[i], x.vec[i])

        # Compute LQR terms
        e = x.vec - x_d.vec
        u_til = -self.K@e

        # Feedback linearization
        u_til[:3] = x.mat[:3,:3].T@u_til[:3] # rotate force to body framed
        u_til[3:] += np.cross(x.mat[:3,:3].T@np.array([0,0,1]), self.cob)*self.V*self.rho*self.gravity # subtract off bouyant torque

        # Convert forces/torques to thruster commands
        f = self.Minv@u_til

        return f

def wrap(chi_1, chi_2):
    while chi_1 - chi_2 > 180:
        chi_1 = chi_1 - 2.0 * 180
    while chi_1 - chi_2 < -180:
        chi_1 = chi_1 + 2.0 * 180
    return chi_1