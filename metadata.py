import numpy as np

"""
gather the param to be used
ready to rewrite
"""

METADATA_v1 = {
    'version': 1,
    # init the scenario's param
    'size': [40, 40, 20],
    'bottom_corner': [-20, -20, -20],
    'fix_depth': -5,
    # init agent's param
    'sensor_r': 10.0,
    'fov': 100,
    'sensor_r_sd': 0.2,  # sensor range noise.
    'sensor_b_sd': 0.01,  # sensor bearing noise.
    'p_prior': 0.5,  # Prior occupancy probability
    'p_occ': 0.9,  # Probability that cell is occupied with total confidence
    'p_free': 0.35,  # Probability that cell is free with total confidence
    'resolution': 0.1,  # Grid resolution in [m]


    # init target's param
    'measurement_disfactor': 0.9,
    'target_init_cov': 50.0,  # initial target diagonal Covariance.
    'lin_dist_range_a2t': (3.0, 8.0),
    'ang_dist_range_a2t': (-np.pi / 5, np.pi / 5),
    'lin_dist_range_t2b': (0.0, 10.0),
    'ang_dist_range_t2b': (-np.pi / 2, np.pi / 2),
    'insight': True,
    'noblock': True,
    # init target's param
    'margin': 1.0,  # a marginal distance btw targets and the agent.
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

# Designate a metadata version to be used throughout the target tracking env.
METADATA = METADATA_v1

TTENV_EVAL_SET = [
    {   # Tracking
        'sensor_r_sd': 0.5,  # sensor range noise.
        'sensor_b_sd': 0.01,  # sensor bearing noise.
        'lin_dist_range_a2b': (3.0, 10.0),
        'ang_dist_range_a2b': (-np.pi, np.pi),
        'lin_dist_range_b2t': (0.0, 3.0),
        'ang_dist_range_b2t': (-np.pi / 2, np.pi / 2),
        'insight': True,
        'noblock': True,
        'target_speed_limit': 2.5,
        'const_q': 0.1,
    },
    {   # Discovery
        'sensor_r_sd': 0.5,  # sensor range noise.
        'sensor_b_sd': 0.01,  # sensor bearing noise.
        'lin_dist_range_a2b': (3.0, 10.0),
        'ang_dist_range_a2b': (-np.pi, np.pi),
        'lin_dist_range_b2t': (0.0, 3.0),
        'ang_dist_range_b2t': (-np.pi / 2, np.pi / 2),
        'insight': False,
        'noblock': True,
        'target_speed_limit': 2.5,
        'const_q': 0.1,
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
