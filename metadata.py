import numpy as np

"""
gather the param to be used
ready to rewrite
"""

METADATA_v0 = {
    'version': 0,
    'sensor_r': 10.0,
    'fov': 120,
    'sensor_r_sd': 0.2,  # sensor range noise.
    'sensor_b_sd': 0.01,  # sensor bearing noise.
    'target_init_cov': 30.0,  # initial target diagonal Covariance.
    'target_init_vel': [0.0, 0.0],  # target's initial velocity.
    'target_speed_limit': 2.0,  # velocity limit of targets.
    'lin_dist_range_a2b': (0.0, 8.0),
    'ang_dist_range_a2b': (-np.pi, np.pi),
    'lin_dist_range_b2t': (0.0, 5.0),
    'ang_dist_range_b2t': (-np.pi, np.pi),
    'margin': 1.0,  # a marginal distance btw targets and the agent.
    'margin2wall': 0.5,  # a marginal distance from a wall. 最小的安全距离
    'action_v': [3, 2, 1, 0],  # action primitives - linear velocities.
    'action_w': [np.pi / 2, 0, -np.pi / 2],  # action primitives - angular velocities.
    'const_q': 0.01,  # target noise constant in beliefs.
    'const_q_true': 0.01,  # target noise constant of actual targets.
}

METADATA_v1 = {
    'version': 1,
    # init agent's param
    'sensor_r': 10.0,
    'fov': 100,
    'sensor_r_sd': 0.2,  # sensor range noise.
    'sensor_b_sd': 0.01,  # sensor bearing noise.
    # init target's param
    'target_init_cov': 50.0,  # initial target diagonal Covariance.
    'lin_dist_range_a2t': (3.0, 8.0),
    'ang_dist_range_a2t': (-np.pi / 5, np.pi / 5),
    'insight': True,
    'noblock': True,
    # init target's param
    'margin': 1,  # a marginal distance btw targets and the agent.
    'margin2wall': 1.0,  # a marginal distance from a wall.
    'const_q': 3.0,  # target noise constant in beliefs.
    'const_q_true': 0.01,  # target noise constant of actual targets.

    # reinforcement learning setting.
    'action_range_high': [1, 1, 1],
    'action_range_low': [0, 0, 0],
    'action_range_scale': [3, np.pi, np.pi/2],
    'action_dim': 6,

    'target_num': 1,
    'target_dim': 4,  # x, y, xdot, ydot
    # reward setting
    'c_mean': 0.2,
    'c_std': 0.0,
    'c_penalty': 5.0,
    'k_3': 0.0,   # 0.3,
    'k_4': 0.0,   # 0.01,
    'k_5': 0.0,   # 0.0002,
    # render setting
    'render': False,
    # control_period
    'control_period': 0.5
}

METADATA_multi_v1 = {
    'version': 'm1',
    'sensor_r': 10.0,
    'fov': 120,
    'sensor_r_sd': 0.2,  # sensor range noise.
    'sensor_b_sd': 0.01,  # sensor bearing noise.
    'target_init_cov': 30.0,  # initial target diagonal Covariance.
    'target_init_vel': [0.0, 0.0],  # target's initial velocity.
    'target_speed_limit': 1.0,  # velocity limit of targets.
    'lin_dist_range_a2b': (5.0, 10.0),
    'ang_dist_range_a2b': (-np.pi, np.pi),
    'lin_dist_range_b2t': (0.0, 10.0),
    'ang_dist_range_b2t': (-np.pi / 2, np.pi / 2),
    'margin': 1.0,  # a marginal distance btw targets and the agent.
    'margin2wall': 1.0,  # a marginal distance from a wall.
    'action_v': [3, 2, 1, 0],  # action primitives - linear velocities.
    'action_w': [np.pi / 2, 0, -np.pi / 2],  # action primitives - angular velocities.
    'const_q': 0.2,  # target noise constant in beliefs.
    'const_q_true': 0.2,  # target noise constant of actual targets.
}

# Designate a metadata version to be used throughout the target tracking env.
METADATA = METADATA_v1

TTENV_EVAL_SET = [
    {  # Tracking
        'lin_dist_range_a2b': (3.0, 10.0),
        'ang_dist_range_a2b': (-np.pi, np.pi),
        'lin_dist_range_b2t': (0.0, 3.0),
        'ang_dist_range_b2t': (-np.pi / 2, np.pi / 2),
        'insight': True,
        'noblock': True,
        'target_speed_limit': 2.5,
    },
    {  # Discovery
        'lin_dist_range_a2b': (3.0, 10.0),
        'ang_dist_range_a2b': (-np.pi, np.pi),
        'lin_dist_range_b2t': (10.0, 15.0),
        'ang_dist_range_b2t': (-np.pi / 2, np.pi / 2),
        'insight': False,
        'noblock': True,
        'target_move': 0.0,
    },
    {  # Navigation
        'lin_dist_range_a2b': (35.0, 40.0),
        'ang_dist_range_a2b': (-np.pi / 4, np.pi / 4),
        'lin_dist_range_b2t': (0.0, 3.0),
        'ang_dist_range_b2t': (-np.pi / 4, np.pi / 4),
        'insight': False,
        'noblock': True,
        'target_move': 0.0,
    },
]

TTENV_EVAL_MULTI_SET = [
    {
        'lin_dist_range_a2b': (3.0, 10.0),
        'ang_dist_range_a2b': (-np.pi, np.pi),
        'lin_dist_range_b2t': (0.0, 3.0),
        'ang_dist_range_b2t': (-np.pi / 2, np.pi / 2),
        'blocked': None,
        'target_speed_limit': 1.0,
        'const_q': 0.02,
    }
]

# METADATA.update(TTENV_EVAL_SET[2])
print('test')
