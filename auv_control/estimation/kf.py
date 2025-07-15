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
                # 观测测量矩阵
                Hmat = np.array([[diff_pred[0], diff_pred[1], 0.0, 0.0],
                                 [-diff_pred[1] / r_pred, diff_pred[0] / r_pred, 0.0, 0.0]]) / r_pred
            else:
                raise ValueError('target dimension for KF must be 2, 4, or 6')
                
            # Innovation for 2D
            innov = z_t - np.array([r_pred, alpha_pred])
            innov[1] = wrap_around(innov[1])

        # Kalman filter update step
        R = np.matmul(np.matmul(Hmat, self.cov), Hmat.T) \
            + self.obs_noise_func((z_t))
        K = np.matmul(np.matmul(self.cov, Hmat.T), LA.inv(R))
        C = np.eye(self.dim) - np.matmul(K, Hmat)  # 简易更新协方差矩阵：P_hat = (I- K_k*H_k)*P_check

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
            
            # 3D noise calculation
            R = np.matmul(np.matmul(Hmat, self.cov), Hmat.T) \
                + self.obs_noise_func(z_t)
            
        else:
            # 2D polar coordinate observation (existing code)
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
                raise ValueError('target dimension for KF must be 2, 4, or 6')
            
            # Innovation for 2D
            innov = z_t - np.array([r_pred, alpha_pred])
            innov[1] = wrap_around(innov[1])
            
            # 2D noise calculation
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
    
if "__main__" == __name__:
    print("=== 3D卡尔曼滤波器测试 ===")
    
    # 定义3D观测噪声函数
    def obs_noise_3d(z):
        """3D球坐标观测噪声矩阵"""
        # z = [r, theta, gamma]
        return np.diag([0.1**2, 0.05**2, 0.05**2])  # 距离、方位角、俯仰角的噪声方差
    
    def obs_noise_2d(z):
        """2D极坐标观测噪声矩阵"""
        # z = (r, alpha)
        return np.diag([0.1**2, 0.05**2])  # 距离、角度的噪声方差
    
    # 测试场景设置
    print("\n1. 初始化3D KF")
    
    # 状态限制 [x_min, y_min, z_min, vx_min, vy_min, vz_min], [x_max, y_max, z_max, vx_max, vy_max, vz_max]
    limit_3d = [np.array([-100, -100, -50, -5, -5, -5]), 
                np.array([100, 100, 10, 5, 5, 5])]
    
    # 状态转移矩阵A (6x6) - 简单的恒速模型
    dt = 0.1
    A_3d = np.array([
        [1, 0, 0, dt, 0, 0],   # x = x + vx*dt
        [0, 1, 0, 0, dt, 0],   # y = y + vy*dt  
        [0, 0, 1, 0, 0, dt],   # z = z + vz*dt
        [0, 0, 0, 1, 0, 0],    # vx = vx
        [0, 0, 0, 0, 1, 0],    # vy = vy
        [0, 0, 0, 0, 0, 1]     # vz = vz
    ])
    
    # 过程噪声矩阵W (6x6)
    W_3d = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
    
    # 创建3D KF
    kf_3d = KFbelief(dim=6, limit=limit_3d, A=A_3d, W=W_3d, obs_noise_func=obs_noise_3d)
    
    # 初始状态：目标在(10, 5, 3)位置，速度(1, 0.5, 0.1)
    init_state_3d = np.array([10.0, 5.0, 3.0, 1.0, 0.5, 0.1])
    init_cov_3d = 1.0
    kf_3d.reset(init_state_3d, init_cov_3d)
    
    print(f"初始状态: {init_state_3d}")
    print(f"初始协方差矩阵: \n{kf_3d.cov}")
    
    # 智能体状态 [x, y, z, yaw]
    agent_state = np.array([0.0, 0.0, 0.0, 0.0])  # 原点，朝向x轴正方向
    
    print("\n2. 测试3D球坐标转换")
    
    # 计算理论观测值
    r_true, theta_true, gamma_true = relative_distance_spherical(
        init_state_3d[:3], agent_state[:3], agent_state[3])
    print(f"理论观测值: r={r_true:.3f}m, θ={np.rad2deg(theta_true):.1f}°, γ={np.rad2deg(gamma_true):.1f}°")
    
    # 添加噪声的观测
    z_obs = np.array([r_true, theta_true, gamma_true]) + np.array([0.05, 0.02, 0.01])
    print(f"带噪声观测: r={z_obs[0]:.3f}m, θ={np.rad2deg(z_obs[1]):.1f}°, γ={np.rad2deg(z_obs[2]):.1f}°")
    
    print("\n3. 测试KF预测步骤")
    print(f"预测前状态: {kf_3d.state}")
    kf_3d.predict()
    print(f"预测后状态: {kf_3d.state}")
    
    print("\n4. 测试KF更新步骤")
    print(f"更新前状态: {kf_3d.state}")
    print(f"更新前协方差对角线: {np.diag(kf_3d.cov)}")
    
    kf_3d.update(z_obs, agent_state)
    
    print(f"更新后状态: {kf_3d.state}")
    print(f"更新后协方差对角线: {np.diag(kf_3d.cov)}")
    
    print("\n5. 测试贪婪更新")
    
    # 测试greedy_update方法
    new_cov, new_state = kf_3d.greedy_update(z_obs, agent_state)
    print(f"贪婪更新结果状态: {new_state}")
    print(f"贪婪更新结果协方差对角线: {np.diag(new_cov)}")
    
    print("\n6. 多步跟踪仿真")
    
    # 重新初始化
    kf_3d.reset(init_state_3d, init_cov_3d)
    
    # 模拟目标运动和观测
    true_positions = []
    estimated_positions = []
    
    # 真实目标状态
    true_state = init_state_3d.copy()
    
    for step in range(10):
        # 目标真实运动（使用相同的A矩阵）
        true_state = np.matmul(A_3d, true_state) + np.random.multivariate_normal(np.zeros(6), W_3d)
        true_positions.append(true_state[:3].copy())
        
        # KF预测
        kf_3d.predict()
        
        # 生成观测
        r_true, theta_true, gamma_true = relative_distance_spherical(
            true_state[:3], agent_state[:3], agent_state[3])
        
        # 添加观测噪声
        obs_noise = np.random.multivariate_normal(np.zeros(3), obs_noise_3d([r_true, theta_true, gamma_true]))
        z_noisy = np.array([r_true, theta_true, gamma_true]) + obs_noise
        
        # KF更新
        kf_3d.update(z_noisy, agent_state)
        estimated_positions.append(kf_3d.state[:3].copy())
        
        print(f"步骤{step+1}: 真实位置[{true_state[0]:.2f}, {true_state[1]:.2f}, {true_state[2]:.2f}], 估计位置[{kf_3d.state[0]:.2f}, {kf_3d.state[1]:.2f}, {kf_3d.state[2]:.2f}]")
    
    print("\n7. 跟踪误差统计")
    
    true_positions = np.array(true_positions)
    estimated_positions = np.array(estimated_positions)
    
    position_errors = np.linalg.norm(true_positions - estimated_positions, axis=1)
    print(f"平均位置误差: {np.mean(position_errors):.3f}m")
    print(f"最大位置误差: {np.max(position_errors):.3f}m")
    print(f"最终估计状态: {kf_3d.state}")
    print(f"最终协方差行列式: {np.linalg.det(kf_3d.cov):.6f}")
    
    print("\n8. 边界情况测试")
    
    # 测试目标在智能体正上方的情况（奇点）
    print("测试奇点情况：目标在智能体正上方")
    overhead_target = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    kf_3d.reset(overhead_target, 1.0)
    
    r_overhead, theta_overhead, gamma_overhead = relative_distance_spherical(
        overhead_target[:3], agent_state[:3], agent_state[3])
    z_overhead = np.array([r_overhead, theta_overhead, gamma_overhead])
    
    print(f"正上方观测: r={r_overhead:.3f}m, θ={np.rad2deg(theta_overhead):.1f}°, γ={np.rad2deg(gamma_overhead):.1f}°")
    
    try:
        kf_3d.update(z_overhead, agent_state)
        print("奇点情况处理成功")
        print(f"更新后状态: {kf_3d.state}")
    except Exception as e:
        print(f"奇点情况出现错误: {e}")
    
    # 测试目标非常接近智能体的情况
    print("\n测试近距离情况：目标非常接近智能体")
    close_target = np.array([0.001, 0.001, 0.001, 0.0, 0.0, 0.0])
    kf_3d.reset(close_target, 1.0)
    
    r_close, theta_close, gamma_close = relative_distance_spherical(
        close_target[:3], agent_state[:3], agent_state[3])
    z_close = np.array([r_close, theta_close, gamma_close])
    
    print(f"近距离观测: r={r_close:.6f}m, θ={np.rad2deg(theta_close):.1f}°, γ={np.rad2deg(gamma_close):.1f}°")
    
    try:
        kf_3d.update(z_close, agent_state)
        print("近距离情况处理成功")
        print(f"更新后状态: {kf_3d.state}")
    except Exception as e:
        print(f"近距离情况出现错误: {e}")
    
    print("\n=== 与2D KF对比测试 ===")
    
    # 创建2D KF进行对比
    limit_2d = [np.array([-100, -100]), np.array([100, 100])]
    A_2d = np.eye(2)
    W_2d = np.diag([0.01, 0.01])
    
    kf_2d = KFbelief(dim=2, limit=limit_2d, A=A_2d, W=W_2d, obs_noise_func=obs_noise_2d)
    
    # 使用相同的x,y位置进行2D测试
    init_state_2d = np.array([10.0, 5.0])
    kf_2d.reset(init_state_2d, 1.0)
    
    # 2D观测（忽略z坐标）
    agent_state_2d = np.array([0.0, 0.0, 0.0])
    r_2d, alpha_2d = relative_distance_polar(init_state_2d, agent_state_2d[:2], agent_state_2d[2])
    z_2d = np.array([r_2d, alpha_2d])
    
    print(f"2D观测: r={r_2d:.3f}m, α={np.rad2deg(alpha_2d):.1f}°")
    print(f"3D观测投影到2D: r={r_true:.3f}m, θ={np.rad2deg(theta_true):.1f}°")
    print(f"2D与3D水平距离差异: {abs(r_2d - np.sqrt(r_true**2 - 3.0**2)):.6f}m")
    
    kf_2d.update(z_2d, agent_state_2d)
    print(f"2D KF更新后状态: {kf_2d.state}")
    print(f"3D KF水平位置: {kf_3d.state[:2]}")
    
    print("\n=== 测试完成 ===")
