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
        z_t : observation 
            - For 2D/4D: [r, alpha] - radial and angular distances from the agent
            - For 6D: [r, theta, gamma] - spherical coordinates (range, azimuth, elevation)
        x_t : agent state in the global frame
            - For 2D/4D: [x, y, orientation] 
            - For 6D: [x, y, z, orientation]
        """
        # Kalman Filter Update
        if self.dim == 6:
            # 3D spherical coordinate observation
            r_pred, theta_pred, gamma_pred = relative_distance_spherical(
                self.state[:3], x_t[:3], x_t[3])
            diff_pred = np.array(self.state[:3]) - np.array(x_t[:3])
            
            # Build 3D observation matrix H (3x6)
            # For spherical coordinates: z = [r, θ, γ]
            # where r = ||p_t - p_a||, θ = azimuth, γ = elevation
            
            r_xy = np.sqrt(diff_pred[0]**2 + diff_pred[1]**2)  # horizontal distance
            r_total = np.sqrt(np.sum(diff_pred**2))  # total distance
            
            if r_total < 1e-8:  # Avoid singularity
                Hmat = np.zeros((3, 6))
            else:
                # ∂r/∂[x,y,z,vx,vy,vz]
                dr_dx = diff_pred[0] / r_total
                dr_dy = diff_pred[1] / r_total  
                dr_dz = diff_pred[2] / r_total
                # dr_dvx = dr_dvy = dr_dvz = 0
                
                # ∂θ/∂[x,y,z,vx,vy,vz] (azimuth)
                if r_xy < 1e-8:  # Avoid singularity at poles
                    dtheta_dx = 0
                    dtheta_dy = 0
                else:
                    dtheta_dx = -diff_pred[1] / (r_xy**2)
                    dtheta_dy = diff_pred[0] / (r_xy**2)
                dtheta_dz = 0
                # dtheta_dvx = dtheta_dvy = dtheta_dvz = 0
                
                # ∂γ/∂[x,y,z,vx,vy,vz] (elevation)
                if r_total < 1e-8:
                    dgamma_dx = 0
                    dgamma_dy = 0
                    dgamma_dz = 0
                else:
                    dgamma_dx = -diff_pred[0] * diff_pred[2] / (r_total**2 * r_xy + 1e-8)
                    dgamma_dy = -diff_pred[1] * diff_pred[2] / (r_total**2 * r_xy + 1e-8)
                    dgamma_dz = r_xy / (r_total**2)
                
                Hmat = np.array([
                    [dr_dx, dr_dy, dr_dz, 0.0, 0.0, 0.0],           # ∂r/∂state
                    [dtheta_dx, dtheta_dy, dtheta_dz, 0.0, 0.0, 0.0], # ∂θ/∂state  
                    [dgamma_dx, dgamma_dy, dgamma_dz, 0.0, 0.0, 0.0]  # ∂γ/∂state
                ])
            
            # Innovation for 3D
            innov = z_t - np.array([r_pred, theta_pred, gamma_pred])
            innov[1] = wrap_around(innov[1])  # wrap azimuth
            innov[2] = wrap_around(innov[2])  # wrap elevation
            
        else:
            # 2D polar coordinate observation (existing code)
            r_pred, alpha_pred = relative_distance_polar(
                self.state[:2], x_t[:2], x_t[2])
            diff_pred = np.array(self.state[:2]) - np.array(x_t[:2])
            
            if self.dim == 2:
                Hmat = np.array([[diff_pred[0], diff_pred[1]],
                                 [-diff_pred[1] / r_pred, diff_pred[0] / r_pred]]) / r_pred
            elif self.dim == 4:
                # Observation measurement matrix
                Hmat = np.array([[diff_pred[0], diff_pred[1], 0.0, 0.0],
                                 [-diff_pred[1] / r_pred, diff_pred[0] / r_pred, 0.0, 0.0]]) / r_pred
            else:
                raise ValueError('target dimension for KF must be 2, 4, or 6')
                
            # Innovation for 2D
            innov = z_t - np.array([r_pred, alpha_pred])
            innov[1] = wrap_around(innov[1])

        # Kalman filter update step
        if self.dim == 6:
            R = np.matmul(np.matmul(Hmat, self.cov), Hmat.T) \
                + self.obs_noise_func((r_pred, theta_pred, gamma_pred))
        else:
            R = np.matmul(np.matmul(Hmat, self.cov), Hmat.T) \
                + self.obs_noise_func((r_pred, alpha_pred))
        
        K = np.matmul(np.matmul(self.cov, Hmat.T), LA.inv(R))
        C = np.eye(self.dim) - np.matmul(K, Hmat)  # Simple covariance update: P_hat = (I - K_k*H_k)*P_check

        self.cov = np.matmul(C, self.cov)
        self.state = np.clip(self.state + np.matmul(K, innov), self.limit[0], self.limit[1])

    def greedy_update(self, z_t, x_t):
        """
        Just for greedy policy
        return the result,but not update the class param
        
        Parameters
        --------
        z_t : observation 
            - For 2D/4D: [r, alpha] - radial and angular distances from the agent
            - For 6D: [r, theta, gamma] - spherical coordinates (range, azimuth, elevation)
        x_t : agent state in the global frame
            - For 2D/4D: [x, y, orientation] 
            - For 6D: [x, y, z, orientation]
        """
        # Kalman Filter Update
        if self.dim == 6:
            # 3D spherical coordinate observation
            r_pred, theta_pred, gamma_pred = relative_distance_spherical(
                self.state[:3], x_t[:3], x_t[3])
            diff_pred = np.array(self.state[:3]) - np.array(x_t[:3])
            
            # Build 3D observation matrix H (3x6)
            r_xy = np.sqrt(diff_pred[0]**2 + diff_pred[1]**2)  # horizontal distance
            r_total = np.sqrt(np.sum(diff_pred**2))  # total distance
            
            if r_total < 1e-8:  # Avoid singularity
                Hmat = np.zeros((3, 6))
            else:
                # ∂r/∂[x,y,z,vx,vy,vz]
                dr_dx = diff_pred[0] / r_total
                dr_dy = diff_pred[1] / r_total  
                dr_dz = diff_pred[2] / r_total
                
                # ∂θ/∂[x,y,z,vx,vy,vz] (azimuth)
                if r_xy < 1e-8:  # Avoid singularity at poles
                    dtheta_dx = 0
                    dtheta_dy = 0
                else:
                    dtheta_dx = -diff_pred[1] / (r_xy**2)
                    dtheta_dy = diff_pred[0] / (r_xy**2)
                dtheta_dz = 0
                
                # ∂γ/∂[x,y,z,vx,vy,vz] (elevation)
                if r_total < 1e-8:
                    dgamma_dx = 0
                    dgamma_dy = 0
                    dgamma_dz = 0
                else:
                    dgamma_dx = -diff_pred[0] * diff_pred[2] / (r_total**2 * r_xy + 1e-8)
                    dgamma_dy = -diff_pred[1] * diff_pred[2] / (r_total**2 * r_xy + 1e-8)
                    dgamma_dz = r_xy / (r_total**2)
                
                Hmat = np.array([
                    [dr_dx, dr_dy, dr_dz, 0.0, 0.0, 0.0],           # ∂r/∂state
                    [dtheta_dx, dtheta_dy, dtheta_dz, 0.0, 0.0, 0.0], # ∂θ/∂state  
                    [dgamma_dx, dgamma_dy, dgamma_dz, 0.0, 0.0, 0.0]  # ∂γ/∂state
                ])
            
            # Innovation for 3D
            innov = z_t - np.array([r_pred, theta_pred, gamma_pred])
            innov[1] = wrap_around(innov[1])  # wrap azimuth
            innov[2] = wrap_around(innov[2])  # wrap elevation
            
            # 3D noise calculation - Use predicted values instead of observed values, consistent with update method
            R = np.matmul(np.matmul(Hmat, self.cov), Hmat.T) \
                + self.obs_noise_func((r_pred, theta_pred, gamma_pred))
            
        else:
            # 2D polar coordinate observation (existing code)
            r_pred, alpha_pred = relative_distance_polar(
                self.state[:2], x_t[:2], x_t[2])
            diff_pred = np.array(self.state[:2]) - np.array(x_t[:2])
            
            if self.dim == 2:
                Hmat = np.array([[diff_pred[0], diff_pred[1]],
                                 [-diff_pred[1] / r_pred, diff_pred[0] / r_pred]]) / r_pred
            elif self.dim == 4:
                # Observation measurement matrix
                Hmat = np.array([[diff_pred[0], diff_pred[1], 0.0, 0.0],
                                 [-diff_pred[1] / r_pred, diff_pred[0] / r_pred, 0.0, 0.0]]) / r_pred
            else:
                raise ValueError('target dimension for KF must be 2, 4, or 6')
            
            # Innovation for 2D
            innov = z_t - np.array([r_pred, alpha_pred])
            innov[1] = wrap_around(innov[1])
            
            # 2D noise calculation
            R = np.matmul(np.matmul(Hmat, self.cov), Hmat.T) \
                + self.obs_noise_func((r_pred, alpha_pred))

        K = np.matmul(np.matmul(self.cov, Hmat.T), LA.inv(R))
        C = np.eye(self.dim) - np.matmul(K, Hmat)  # Simple covariance update: P_hat = (I - K_k*H_k)*P_check

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
        z_t:(r,theta) // target's polar coordinates relative to agent
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

def relative_distance_spherical(xyz_target, xyz_base, theta_base):
    """
    Calculate spherical coordinates (r, θ, γ) of target relative to agent
    
    Parameters
    ----------
    xyz_target : array_like, shape (3,)
        Target position [x, y, z] in global frame
    xyz_base : array_like, shape (3,)  
        Agent position [x, y, z] in global frame
    theta_base : float
        Agent orientation (yaw) in global frame
        
    Returns
    -------
    r : float
        Distance from agent to target
    theta : float
        Azimuth angle (horizontal angle from agent's forward direction)
    gamma : float
        Elevation angle (vertical angle from horizontal plane)
    """
    # Transform to agent's local frame
    xyz_target_base = transform_3d(xyz_target, theta_base, xyz_base)
    return cartesian2spherical(xyz_target_base)

def transform_3d(vec, theta_base, xyz_base=[0.0, 0.0, 0.0]):
    """
    Transform 3D vector from global frame to agent's local frame
    
    Parameters
    ----------
    vec : array_like, shape (3,)
        3D vector in global coordinate
    theta_base : float
        Agent's yaw angle in global frame
    xyz_base : array_like, shape (3,)
        Agent's position in global frame
        
    Returns
    -------
    vec_local : array_like, shape (3,)
        3D vector in agent's local frame
    """
    assert len(vec) == 3
    # Only rotate around z-axis (yaw rotation)
    # R_z(θ)^T * (vec - xyz_base)
    cos_theta = np.cos(theta_base)
    sin_theta = np.sin(theta_base)
    
    R_T = np.array([
        [cos_theta, sin_theta, 0],
        [-sin_theta, cos_theta, 0], 
        [0, 0, 1]
    ])
    
    return np.matmul(R_T, vec - np.array(xyz_base))

def cartesian2spherical(xyz):
    """
    Convert 3D Cartesian coordinates to spherical coordinates
    
    Parameters
    ----------
    xyz : array_like, shape (3,)
        Cartesian coordinates [x, y, z]
        
    Returns
    -------
    r : float
        Radial distance
    theta : float
        Azimuth angle (in x-y plane from x-axis)
    gamma : float
        Elevation angle (from x-y plane toward z-axis)
    """
    x, y, z = xyz[0], xyz[1], xyz[2]
    
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)  # azimuth
    
    # Elevation: angle from horizontal plane
    r_xy = np.sqrt(x**2 + y**2)
    gamma = np.arctan2(z, r_xy)  # elevation
    
    return r, theta, gamma

def transform_2d(vec, theta_base, xy_base=[0.0, 0.0]):
    """
    Both vec and frame_xy are in the global coordinate. vec is a vector
    you want to transform with respect to a certain frame which is located at
    frame_xy with ang.
    R^T * (vec - frame_xy).
    R is a rotation matrix of the frame w.r.t the global frame.
    This is a coordinate transformation function from world coordinate system to agent coordinate system
    """
    assert (len(vec) == 2)
    return np.matmul([[np.cos(theta_base), np.sin(theta_base)],
                      [-np.sin(theta_base), np.cos(theta_base)]],
                     vec - np.array(xy_base))


def cartesian2polar(xy):
    """
    Convert Cartesian coordinates to polar coordinates
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