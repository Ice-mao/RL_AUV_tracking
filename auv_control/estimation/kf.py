import numpy as np
from numpy import linalg as LA

from filterpy.kalman import JulierSigmaPoints, UnscentedKalmanFilter, ExtendedKalmanFilter


class KFbelief(object):
    """
    Kalman Filter for the target tracking problem.

    state : target state
    x : agent state
    z : observation (r, alpha)
    """

    def __init__(self, dim, limit, dim_z=2, A=None, W=None,
                 obs_noise_func=None, collision_func=None):
        """
        dim : dimension of state
        limit : An array of two vectors.
                limit[0] = minimum values for the state,
                limit[1] = maximum value for the state
        dim_z : dimension of observation,
        A : state transition matrix
        W : state noise matrix
        obs_noise_func : observation noise matrix function of z
        collision_func : collision checking function
        """
        self.dim = dim
        self.limit = limit
        self.A = np.eye(self.dim) if A is None else A
        self.W = W if W is not None else np.zeros((self.dim, self.dim))
        self.obs_noise_func = obs_noise_func
        self.collision_func = collision_func

    def reset(self, init_state, init_cov):
        self.state = init_state
        self.cov = init_cov * np.eye(self.dim)

    def predict(self):
        # Prediction
        state_new = np.matmul(self.A, self.state)  # state_check
        self.cov = np.matmul(np.matmul(self.A, self.cov), self.A.T) + self.W
        self.state = np.clip(state_new, self.limit[0], self.limit[1])

    def update(self, z_t, x_t):
        """
        Parameters
        --------
        z_t : observation - radial and angular distances from the agent.
        x_t : agent state (x, y, orientation) in the global frame.
        dim==4 加入对速度的估计
        """
        # Kalman Filter Update
        r_pred, alpha_pred = relative_distance_polar(
            self.state[:2], x_t[:2], x_t[2])
        diff_pred = np.array(self.state[:2]) - np.array(x_t[:2])
        if self.dim == 2:
            Hmat = np.array([[diff_pred[0], diff_pred[1]],
                             [-diff_pred[1] / r_pred, diff_pred[0] / r_pred]]) / r_pred
        elif self.dim == 4:
            # 观测测量矩阵
            Hmat = np.array([[diff_pred[0], diff_pred[1], 0.0, 0.0],
                             [-diff_pred[1] / r_pred, diff_pred[0] / r_pred, 0.0, 0.0]]) / r_pred
        else:
            raise ValueError('target dimension for KF must be either 2 or 4')
        innov = z_t - np.array([r_pred, alpha_pred])
        innov[1] = wrap_around(innov[1])

        R = np.matmul(np.matmul(Hmat, self.cov), Hmat.T) \
            + self.obs_noise_func((r_pred, alpha_pred))
        K = np.matmul(np.matmul(self.cov, Hmat.T), LA.inv(R))
        C = np.eye(self.dim) - np.matmul(K, Hmat)  # 简易更新协方差矩阵：P_hat = (I- K_k*H_k)*P_check

        self.cov = np.matmul(C, self.cov)
        self.state = np.clip(self.state + np.matmul(K, innov), self.limit[0], self.limit[1])

    def greedy_update(self, z_t, x_t):
        """
        Just for greedy policy
        return the result,but not update the class param
        """
        # Kalman Filter Update
        r_pred, alpha_pred = relative_distance_polar(
            self.state[:2], x_t[:2], x_t[2])
        diff_pred = np.array(self.state[:2]) - np.array(x_t[:2])
        if self.dim == 2:
            Hmat = np.array([[diff_pred[0], diff_pred[1]],
                             [-diff_pred[1] / r_pred, diff_pred[0] / r_pred]]) / r_pred
        elif self.dim == 4:
            # 观测测量矩阵
            Hmat = np.array([[diff_pred[0], diff_pred[1], 0.0, 0.0],
                             [-diff_pred[1] / r_pred, diff_pred[0] / r_pred, 0.0, 0.0]]) / r_pred
        else:
            raise ValueError('target dimension for KF must be either 2 or 4')
        innov = z_t - np.array([r_pred, alpha_pred])
        innov[1] = wrap_around(innov[1])

        R = np.matmul(np.matmul(Hmat, self.cov), Hmat.T) \
            + self.obs_noise_func((r_pred, alpha_pred))
        K = np.matmul(np.matmul(self.cov, Hmat.T), LA.inv(R))
        C = np.eye(self.dim) - np.matmul(K, Hmat)  # 简易更新协方差矩阵：P_hat = (I- K_k*H_k)*P_check

        return np.matmul(C, self.cov), np.clip(self.state + np.matmul(K, innov), self.limit[0], self.limit[1])

class UKFbelief(object):
    """
    Unscented Kalman Filter from filterpy
    """

    def __init__(self, dim, limit, dim_z=2, fx=None, W=None, obs_noise_func=None,
                 collision_func=None, sampling_period=0.5, kappa=1):
        """
        dim : dimension of state
            ***Assuming dim==3: (x,y,theta), dim==4: (x,y,xdot,ydot), dim==5: (x,y,theta,v,w)
        limit : An array of two vectors. limit[0] = minimum values for the state,
                                            limit[1] = maximum value for the state
        dim_z : dimension of observation,
        fx : x_tp1 = fx(x_t, dt), state dynamic function
        W : state noise matrix
        obs_noise_func : observation noise matrix function of z
        collision_func : collision checking function
        n : the number of sigma points
        """
        self.dim = dim
        self.limit = limit
        self.W = W if W is not None else np.zeros((self.dim, self.dim))
        self.obs_noise_func = obs_noise_func
        self.collision_func = collision_func

        def hx(y, agent_state, measure_func=relative_distance_polar):
            r_pred, alpha_pred = measure_func(y[:2], agent_state[:2],
                                              agent_state[2])
            return np.array([r_pred, alpha_pred])

        def x_mean_fn_(sigmas, Wm):
            """
            follow the filter.doc
            """
            if dim == 3:
                # (x,y,theta)
                x = np.zeros(dim)
                sum_sin, sum_cos = 0., 0.
                for i in range(len(sigmas)):
                    s = sigmas[i]
                    x[0] += s[0] * Wm[i]
                    x[1] += s[1] * Wm[i]
                    sum_sin += np.sin(s[2]) * Wm[i]
                    sum_cos += np.cos(s[2]) * Wm[i]
                x[2] = np.arctan2(sum_sin, sum_cos)
                return x
            elif dim == 5:
                x = np.zeros(dim)
                sum_sin, sum_cos = 0., 0.
                for i in range(len(sigmas)):
                    s = sigmas[i]
                    x[0] += s[0] * Wm[i]
                    x[1] += s[1] * Wm[i]
                    x[3] += s[3] * Wm[i]
                    x[4] += s[4] * Wm[i]
                    sum_sin += np.sin(s[2]) * Wm[i]
                    sum_cos += np.cos(s[2]) * Wm[i]
                x[2] = np.arctan2(sum_sin, sum_cos)
                return x
            else:
                return None

        def z_mean_fn_(sigmas, Wm):
            x = np.zeros(dim_z)
            sum_sin, sum_cos = 0., 0.
            for i in range(len(sigmas)):
                s = sigmas[i]
                x[0] += s[0] * Wm[i]
                sum_sin += np.sin(s[1]) * Wm[i]
                sum_cos += np.cos(s[1]) * Wm[i]
            x[1] = np.arctan2(sum_sin, sum_cos)
            return x

        def residual_x_(x, xp):
            """
            x : state, [x, y, theta]
            xp : predicted state
            """
            if dim == 3 or dim == 5:
                r_x = x - xp
                r_x[2] = wrap_around(r_x[2])
                return r_x
            else:
                return None

        def residual_z_(z, zp):
            """
            z : observation, [r, alpha]
            zp : predicted observation
            """
            r_z = z - zp
            r_z[1] = wrap_around(r_z[1])
            return r_z

        sigmas = JulierSigmaPoints(n=dim, kappa=kappa)
        self.ukf = UnscentedKalmanFilter(dim, dim_z, sampling_period, fx=fx,
                                         hx=hx, points=sigmas, x_mean_fn=x_mean_fn_,
                                         z_mean_fn=z_mean_fn_, residual_x=residual_x_,
                                         residual_z=residual_z_)

    def reset(self, init_state, init_cov):
        self.state = init_state
        self.cov = init_cov * np.eye(self.dim)
        self.ukf.x = self.state
        self.ukf.P = self.cov
        self.ukf.Q = self.W  # process noise matrix

    def predict(self, u_t=None):
        if u_t is None:
            u_t = np.array([0.1 * np.random.random(),
                            np.pi * np.random.random() - 0.5 * np.pi])

        # Kalman Filter Update
        self.ukf.predict(u=u_t)
        self.cov = self.ukf.P
        self.state = np.clip(self.ukf.x, self.limit[0], self.limit[1])

    def update(self, z_t, x_t):
        """
        z_t:(r,theta) //target相对于agent的极坐标
        x_t:agent.state(x,y,theta)
        """
        # Kalman Filter Update
        r_pred, alpha_pred = relative_distance_polar(self.ukf.x[:2], x_t[:2], x_t[2])
        self.ukf.update(z_t, R=self.obs_noise_func((r_pred, alpha_pred)),
                        agent_state=x_t)

        self.cov = self.ukf.P
        self.state = np.clip(self.ukf.x, self.limit[0], self.limit[1])


def relative_distance_polar(xy_target, xy_base, theta_base):
    xy_target_base = transform_2d(xy_target, theta_base, xy_base)
    return cartesian2polar(xy_target_base)

def transform_2d(vec, theta_base, xy_base=[0.0, 0.0]):
    """
    Both vec and frame_xy are in the global coordinate. vec is a vector
    you want to transform with respect to a certain frame which is located at
    frame_xy with ang.
    R^T * (vec - frame_xy).
    R is a rotation matrix of the frame w.r.t the global frame.
    这是一个向量从世界坐标系到agent坐标系的坐标变换函数
    """
    assert (len(vec) == 2)
    return np.matmul([[np.cos(theta_base), np.sin(theta_base)],
                      [-np.sin(theta_base), np.cos(theta_base)]],
                     vec - np.array(xy_base))


def cartesian2polar(xy):
    """
    笛卡尔坐标系坐标转极坐标系坐标
    """
    r = np.sqrt(np.sum(xy ** 2))
    alpha = np.arctan2(xy[1], xy[0])
    return r, alpha


def wrap_around(x):
    # x \in [-pi,pi)
    if x >= np.pi:
        return x - 2 * np.pi
    elif x < -np.pi:
        return x + 2 * np.pi
    else:
        return x