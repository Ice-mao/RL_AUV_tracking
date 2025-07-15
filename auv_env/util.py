import numpy as np
from numpy import linalg as LA

from PIL import Image
from torchvision import transforms


# Convention : VARIABLE_OBJECT_FRAME. If FRAME is omitted, it means it is with
# respect to the global frame.
def image_preprocess(image: np.ndarray) -> np.ndarray:
    """
        In HoloOcean, we get the image in OpenCV format, which is BGR format.
    """
    # for key in images:
    #     rgb_image = images[key][:, :, :3]  # 取前 3 个通道 (H, W, 3)
    #     pil_image = Image.fromarray(rgb_image)
    #     preprocess = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #     tensor_image = preprocess(pil_image)
    #     image = tensor_image.numpy()
    #     images[key] = image
    bgr_image = image[:, :, :3]  # 取前 3 个通道 (H, W, 3)
    rgb_image = bgr_image[:, :, ::-1] 
    pil_image = Image.fromarray(rgb_image)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor_image = preprocess(pil_image)
    image = tensor_image.numpy()
    return image


def wrap_around(x):
    # x \in [-pi,pi)
    if x >= np.pi:
        return x - 2 * np.pi
    elif x < -np.pi:
        return x + 2 * np.pi
    else:
        return x


def cartesian2polar(xy):
    """
    笛卡尔坐标系坐标转极坐标系坐标
    """
    r = np.sqrt(np.sum(xy ** 2))
    alpha = np.arctan2(xy[1], xy[0])
    return r, alpha


def cartesian2polar_dot(x, y, x_dot, y_dot):
    """
    笛卡尔坐标系速度转极坐标系径向速度和角速度
    """
    r2 = x * x + y * y
    if r2 == 0.0:
        return 0.0, 0.0
    r_dot = (x * x_dot + y * y_dot) / np.sqrt(r2)
    alpha_dot = (x * y_dot - x_dot * y) / r2
    return r_dot, alpha_dot


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


def transform_2d_inv(vec, theta_base, xy_base):
    """
    Both vec and frame_xy are in the global coordinate. vec is a vector
    you want to transform with respect to a certain frame which is located at
    frame_xy with ang.
    R^T * (vec - frame_xy).
    R is a rotation matrix of the frame w.r.t the global frame.
    这是一个向量经过旋转角度和平移之后得到新向量的坐标逆变换函数
    """
    assert (len(vec) == 2)
    return np.matmul([[np.cos(theta_base), -np.sin(theta_base)],
                      [np.sin(theta_base), np.cos(theta_base)]],
                     vec) + np.array(xy_base)


def rotation_2d_dot(xy_target, xy_dot_target, theta_base, theta_dot_base):
    """
    Cartesian velocity in a rotating frame.
    函数根据旋转坐标系的角度和角速度，计算了目标点在旋转坐标系中的速度。
    环境中的应用：将目标的全局坐标速度带入到agent的旋转坐标系中计算目标在旋转坐标系中的速度
    """
    s_b = np.sin(theta_base)
    c_b = np.cos(theta_base)
    x_dot_target_bframe = (-s_b * xy_target[0] + c_b * xy_target[1]) * theta_dot_base + \
                          c_b * xy_dot_target[0] + s_b * xy_dot_target[1]
    y_dot_target_bframe = - (c_b * xy_target[0] + s_b * xy_target[1]) * theta_dot_base - \
                          s_b * xy_dot_target[0] + c_b * xy_dot_target[1]
    return x_dot_target_bframe, y_dot_target_bframe


def transform_2d_dot(xy_target, xy_dot_target, theta_base, theta_dot_base, xy_base, xy_dot_base):
    """
    Cartesian velocity in a rotating and translating frame.
    函数返回旋转坐标系中的速度
    """
    rotated_xy_dot_target_bframe = rotation_2d_dot(xy_target, xy_dot_target, theta_base, theta_dot_base)
    rotated_xy_dot_base_bframe = rotation_2d_dot(xy_base, xy_dot_base, theta_base, theta_dot_base)
    return np.array(rotated_xy_dot_target_bframe) - np.array(rotated_xy_dot_base_bframe)


def relative_distance_polar(xy_target, xy_base, theta_base):
    xy_target_base = transform_2d(xy_target, theta_base, xy_base)
    return cartesian2polar(xy_target_base)


def polar_distance_global(polar_vec, xy_base, theta_base):
    # 将极坐标转换为笛卡尔坐标
    xy_target = np.array([polar_vec[0] * np.cos(polar_vec[1]), polar_vec[0] * np.sin(polar_vec[1])])
    xy_target_global = transform_2d_inv(xy_target, theta_base, xy_base)
    return xy_target_global


def relative_velocity_polar(xy_target, xy_dot_target, xy_base, theta_base, v_base, w_base):
    """
    Relative velocity in a given polar coordinate (radial velocity, angular velocity).

    Parameters:
    ---------
    xy_target : xy coordinate of a target in the global frame.
    xy_dot_target : xy velocity of a target in the global frame.
    xy_base : xy coordinate of the origin of a base frame in the global frame.
    theta_base : orientation of a base frame in the global frame.
    v_base : translational velocity of a base frame in the global frame.
    w_base : rotational velocity of a base frame in the global frame.
    """
    xy_dot_base = vw_to_xydot(v_base, w_base, theta_base)
    xy_target_base = transform_2d(xy_target, theta_base, xy_base)
    xy_dot_target_base = transform_2d_dot(xy_target, xy_dot_target, theta_base,
                                          w_base, xy_base, xy_dot_base)
    r_dot_b, alpha_dot_b = cartesian2polar_dot(xy_target_base[0],
                                               xy_target_base[1], xy_dot_target_base[0], xy_dot_target_base[1])
    return r_dot_b, alpha_dot_b


def relative_velocity_polar_se2(xyth_target, vw_target, xyth_base, vw_base):
    """
    Radial and angular velocity of the target with respect to the base frame
    located at xy_b with a rotation of theta_b, moving with a translational
    velocity v and rotational velocity w. This function is designed specifically
    for the SE2 Agent and SE2 Target target case.

    Parameters
    ---------
    xyth_target : (x, y, orientation) of a target in the global frame.
    vw_target : translational and rotational velocity of a target in the global frame.
    xyth_base : (x, y, orientation) of a base frame in the global frame.
    vw_base : translational and rotational velocity of a base frame in the global frame.
    """
    xy_dot_target = vw_to_xydot(vw_target[0], vw_target[1], xyth_base[2])
    return relative_velocity_polar(xyth_target[:2], xy_dot_target, xyth_base[:2],
                                   xyth_base[2], vw_base[0], vw_base[1])


def vw_to_xydot(v, w, theta):
    """
    Conversion from translational and rotational velocity to cartesian velocity
    in the global frame according to differential-drive dynamics.

    Parameters
    ---------
    v : translational velocity.
    w : rotational velocity.
    theta : orientation of a base object in the global frame.
    """
    if w < 0.001:
        x_dot = v * np.cos(theta + w / 2)
        y_dot = v * np.sin(theta + w / 2)
    else:
        x_dot = v / w * (np.sin(theta + w) - np.sin(theta))
        y_dot = v / w * (np.cos(theta) - np.cos(theta + w))
    return x_dot, y_dot


def iterative_mare(X_0, A, W, C, R, l):
    """
    Solving a modified algebraic Riccati equation for the Kalman Filter by
    iteration.

    Parameters
    ---------
    x_t+1 = Ax_t + w_t  where w_t ~ W
    z_t = Cx_t + v_t  where v_t ~ R
    l = Bernoulli process parameter for the arrival of an observation.
    """

    def mare(X):
        K = np.matmul(C, np.matmul(X, C.T)) + R
        B = np.matmul(A, np.matmul(X, C.T))
        G = np.matmul(C, np.matmul(X, A.T))

        return np.matmul(A, np.matmul(X, A.T)) + W \
            - l * np.matmul(B, np.matmul(LA.inv(K), G))

    X = X_0
    error = 1.0
    count = 0
    while (error > 1e-3):
        X_next = mare(X)
        error = np.abs(LA.det(X_next) - LA.det(X))
        X = X_next
        count += 1
        if count > 1000:
            raise ValueError('No convergence.')

    return X


def get_nlogdetcov_bounds(P0, A, W, TH):
    """
    The upper and lower bounds of a sum of negative log determinant of a belief
    covariance.
    The upper bound follows the Theorem 4 in sinopoli et. al. with a probability
    of the arrival of an observation set to 1. The lower bound is the case when
    there is no observation during the episode and only the prediction step of
    the Kalman Filter proceeded.

    Parameters:
    ---------
    P0 : The initial covariance of a belief
    A : Target belief state matrix
    W : Target belief state noise matrix
    T : Time horizon of an episode
    """
    from numpy import linalg as LA
    upper_bound = - TH * np.log(LA.det(W))
    lower_bound = 0
    X = P0
    X = np.matmul(np.matmul(A, X), A.T) + W
    for _ in range(TH):
        X = np.matmul(np.matmul(A, X), A.T) + W
        lower_bound += - np.log(LA.det(X))
    return lower_bound, upper_bound


def get_nlogdetcov_bounds_step(P0, A, W, TH):
    """
    The upper and lower bounds of a sum of negative log determinant of a belief
    covariance.
    The upper bound follows the Theorem 4 in sinopoli et. al. with a probability
    of the arrival of an observation set to 1. The lower bound is the case when
    there is no observation during the episode and only the prediction step of
    the Kalman Filter proceeded.

    Parameters:
    ---------
    P0 : The initial covariance of a belief
    A : Target belief state matrix
    W : Target belief state noise matrix
    T : Time horizon of an episode
    """
    from numpy import linalg as LA
    upper_bound = - np.log(LA.det(W))
    X = P0
    X = np.matmul(np.matmul(A, X), A.T) + W
    for _ in range(TH):
        X = np.matmul(np.matmul(A, X), A.T) + W
    lower_bound = - np.log(LA.det(X))
    return lower_bound, upper_bound


def cartesian2spherical(xyz):
    """
    笛卡尔坐标系坐标转球坐标系坐标
    
    Parameters
    ----------
    xyz : array_like, shape (3,)
        笛卡尔坐标 [x, y, z]
    
    Returns
    -------
    r : float
        径向距离
    theta : float  
        方位角 (azimuth) - 从x轴到xy平面投影的角度 [-π, π]
    gamma : float
        俯仰角 (elevation) - 从xy平面到z轴的角度 [-π/2, π/2]
    """
    x, y, z = xyz[0], xyz[1], xyz[2]
    
    # 径向距离
    r = np.sqrt(x*x + y*y + z*z)
    
    # 方位角 (azimuth) - 水平角度
    theta = np.arctan2(y, x)
    
    # 俯仰角 (elevation) - 垂直角度
    if r > 1e-8:  # 避免除零
        gamma = np.arcsin(z / r)
    else:
        gamma = 0.0
    
    return r, theta, gamma


def transform_3d(vec, theta_base, xyz_base=[0.0, 0.0, 0.0]):
    """
    3D坐标变换：从世界坐标系到智能体坐标系
    只考虑绕z轴的旋转(yaw)，保持x-forward, y-left, z-up的约定
    
    Parameters
    ----------
    vec : array_like, shape (3,)
        世界坐标系下的3D向量
    theta_base : float
        智能体相对于世界坐标系的yaw角 (绕z轴旋转)
    xyz_base : array_like, shape (3,)
        智能体在世界坐标系中的位置
    
    Returns
    -------
    vec_transformed : array_like, shape (3,)
        智能体坐标系下的3D向量
    """
    assert len(vec) == 3
    
    # 平移
    vec_translated = vec - np.array(xyz_base)
    
    # 绕z轴旋转 (yaw rotation)
    cos_theta = np.cos(theta_base)
    sin_theta = np.sin(theta_base)
    
    rotation_matrix = np.array([
        [cos_theta,  sin_theta, 0],
        [-sin_theta, cos_theta, 0],
        [0,          0,         1]
    ])
    
    return np.matmul(rotation_matrix, vec_translated)


def relative_distance_spherical(xyz_target, xyz_base, theta_base):
    """
    计算目标相对于智能体的3D球坐标
    
    Parameters
    ----------
    xyz_target : array_like, shape (3,)
        目标在世界坐标系中的位置 [x, y, z]
    xyz_base : array_like, shape (3,)  
        智能体在世界坐标系中的位置 [x, y, z]
    theta_base : float
        智能体的yaw角 (绕z轴旋转角度)
    
    Returns
    -------
    r : float
        目标到智能体的直线距离
    theta : float
        方位角 - 目标相对于智能体前向方向的水平角度 [-π, π]
        正值表示目标在智能体右侧，负值表示在左侧
    gamma : float  
        俯仰角 - 目标相对于智能体水平面的垂直角度 [-π/2, π/2]
        正值表示目标在智能体上方，负值表示在下方
    """
    # 将目标坐标转换到智能体坐标系
    xyz_target_relative = transform_3d(xyz_target, theta_base, xyz_base)
    
    # 转换为球坐标
    return cartesian2spherical(xyz_target_relative)


if __name__ == "__main__":
    print('=== 3D球坐标转换函数测试 ===')

    # 测试1: 正前方目标
    agent_pos = np.array([0, 0, 0])
    target_pos = np.array([10, 0, 0])  # 正前方10米
    agent_yaw = 0.0

    r, theta, gamma = relative_distance_spherical(target_pos, agent_pos, agent_yaw)
    print(f'测试1 - 正前方目标:')
    print(f'  距离: {r:.2f}m')
    print(f'  方位角: {np.rad2deg(theta):.1f}° (应该约为0°)')
    print(f'  俯仰角: {np.rad2deg(gamma):.1f}° (应该约为0°)')

    # 测试2: 右侧目标
    target_pos = np.array([0, -10, 0])  # 右侧10米
    r, theta, gamma = relative_distance_spherical(target_pos, agent_pos, agent_yaw)
    print(f'\n测试2 - 右侧目标:')
    print(f'  距离: {r:.2f}m')
    print(f'  方位角: {np.rad2deg(theta):.1f}° (应该约为-90°)')
    print(f'  俯仰角: {np.rad2deg(gamma):.1f}° (应该约为0°)')

    # 测试3: 上方目标
    target_pos = np.array([7, 0, 7])  # 前方7米，上方7米
    r, theta, gamma = relative_distance_spherical(target_pos, agent_pos, agent_yaw)
    print(f'\n测试3 - 上方目标:')
    print(f'  距离: {r:.2f}m (应该约为{np.sqrt(7*7+7*7):.2f}m)')
    print(f'  方位角: {np.rad2deg(theta):.1f}° (应该约为0°)')
    print(f'  俯仰角: {np.rad2deg(gamma):.1f}° (应该约为45°)')

    # 测试4: 智能体旋转情况
    agent_yaw = np.pi/2  # 智能体向左转90度
    target_pos = np.array([10, 0, 5])  # 世界坐标系中的位置
    r, theta, gamma = relative_distance_spherical(target_pos, agent_pos, agent_yaw)
    print(f'\n测试4 - 智能体旋转90°:')
    print(f'  距离: {r:.2f}m')
    print(f'  方位角: {np.rad2deg(theta):.1f}°')
    print(f'  俯仰角: {np.rad2deg(gamma):.1f}°')