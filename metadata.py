import numpy as np

# choice_list = ["METADATA_v0", "METADATA_v1", "METADATA_v1_SAMPLE", "METADATA_v2"]
choice = "METADATA_v2"

METADATA_v0 = {
    'version': 1,
    # init the scenario's param
    'size': [40, 40, 20],
    'bottom_corner': [-20, -20, -20],
    'fix_depth': -5,
    'use_sonar': False,

    # init agent's param
    'sensor_r': 10.0,
    'fov': 100,
    'sensor_r_sd': 0.2,  # sensor range noise.
    'sensor_b_sd': 0.01,  # sensor bearing noise.
    'p_prior': 0.5,  # Prior occupancy probability
    'p_occ': 0.8,  # Probability that cell is occupied with total confidence
    'p_free': 0.25,  # Probability that cell is free with total confidence
    'resolution': 0.1,  # Grid resolution in [m]

    # init target's param
    'measurement_disfactor': 0.9,
    'target_init_cov': 50.0,  # initial target diagonal Covariance.
    'lin_dist_range_a2t': (3.0, 8.0),
    'ang_dist_range_a2t': (-np.pi / 5, np.pi / 5),
    'lin_dist_range_t2b': (0.0, 2.0),
    'ang_dist_range_t2b': (-np.pi / 2, np.pi / 2),
    'insight': True,
    'noblock': True,
    'margin': 1.0,  # a marginal distance btw targets and the agent.
    'margin2wall': 1.0,  # a marginal distance from a wall.
    'const_q': 0.5,  # target noise constant in beliefs.
    'const_q_true': 0.01,  # target noise constant of actual targets.
    'lqr_l_p': 50,  # control the target's veocity

    # reinforcement learning setting.
    'random': False,  # for domain randomization.
    'task_random': False,  # if False, according to 'insight' to determine
    'action_range_high': [1, 1, 1],
    'action_range_low': [-1, -1, -1],
    'action_range_scale': [3 / 2, np.pi / 2, np.pi / 4],
    'action_dim': 3,
    'target_num': 1,
    'target_dim': 4,  # x, y, xdot, ydot
    # reward setting
    'reward_param': {
        'c_mean': 0.2,
        'c_std': 0.0,
        'c_penalty': 5.0,
        'k_3': 0.0,  # 0.3,
        'k_4': 0.0,  # 0.01,
        'k_5': 0.0,  # 0.0002,
    },
    # render setting
    'render': True,
    # control_period
    'control_period': 0.5,
    # eval setting
    'eval_fixed': False
}

"""
    used for RGB env
"""
METADATA_v1 = {
    'version': 1,
    'render': True,
    'eval_fixed': False,
    # env set
    'env': {
        'task_random': False,  # insight or not
        'insight': [True, [True, False]],
        'noblock': True,
        'control_period': 0.5,
        'margin': 1.0,  # a marginal distance btw targets and the agent.
        'margin2wall': 1.0,  # a marginal distance from a wall.

        'target_num': 1,
        'target_dim': 4,  # x, y, xdot, ydot
    },

    # init the scenario's param
    'scenario': {
        'size': [40, 40, 20],
        'bottom_corner': [-20, -20, -20],
        'fix_depth': -5,
    },

    # init agent's param
    'agent': {
        'random': False,
        'use_sonar': False,
        'grid': {
            'use_grid': False,
            'p_prior': 0.5,  # Prior occupancy probability
            'p_occ': 0.8,  # Probability that cell is occupied with total confidence
            'p_free': 0.25,  # Probability that cell is free with total confidence
            'resolution': 0.1,  # Grid resolution in [m]
        },
        'sensor_r': 10.0,
        'fov': 100,
        'sensor_r_sd': 0.2,  # sensor range noise.
        'sensor_b_sd': 0.01,  # sensor bearing noise.
    },

    # init target's param
    'target': {
        'target_dim': 4,

        'random': False,  # from const_q set
        # target noise constant in beliefs.
        'const_q': [0.5, [0.01, 0.02, 0.05, 0.1, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]],

        'measurement_disfactor': 0.9,
        'target_init_cov': 50.0,  # initial target diagonal Covariance.
        'lin_dist_range_a2t': (3.0, 8.0),
        'ang_dist_range_a2t': (-np.pi / 5, np.pi / 5),
        'lin_dist_range_t2b': (0.0, 2.0),
        'ang_dist_range_t2b': (-np.pi / 2, np.pi / 2),
        'lqr_l_p': 50,  # control the target's veocity
    },

    # reinforcement learning setting.
    'is_training': True,
    'algorithms': 'PPO',  # PPO、SAC
    'policy': 'CNN',  # CNN、MLP 记得同时修改环境
    # # action set
    'action_range_high': [1, 1, 1],
    'action_range_low': [0, -1, -1],
    'action_range_scale': [3/2, np.pi/2, np.pi/4],
    'action_dim': 6,

    # # reward setting
    'reward_param': {
        'c_mean': 0.2,
        'c_std': 0.0,
        'c_penalty': 5.0,
        'k_3': 0.0,  # 0.3,
        'k_4': 0.0,  # 0.01,
        'k_5': 0.0,  # 0.0002,
    },
}

METADATA_v1_SAMPLE = {
    'version': 1,
    'render': True,
    'eval_fixed': True,
    # env set
    'env': {
        'task_random': False,  # insight or not
        'insight': [True, [True, False]],
        'noblock': True,
        'control_period': 0.5,
        'margin': 1.0,  # a marginal distance btw targets and the agent.
        'margin2wall': 1.0,  # a marginal distance from a wall.

        'target_num': 1,
        'target_dim': 4,  # x, y, xdot, ydot
    },

    # init the scenario's param
    'scenario': {
        'size': [1000, 1000, 100],
        'bottom_corner': [-500, -500, -100],
        'fix_depth': -45,
        # 'size': [40, 40, 20],
        # 'bottom_corner': [-20, -20, -20],
        # 'fix_depth': -5,
    },

    # init agent's param
    'agent': {
        'random': False,
        'use_sonar': False,
        'grid': {
            'use_grid': False,
            'p_prior': 0.5,  # Prior occupancy probability
            'p_occ': 0.8,  # Probability that cell is occupied with total confidence
            'p_free': 0.25,  # Probability that cell is free with total confidence
            'resolution': 0.1,  # Grid resolution in [m]
        },
        'sensor_r': 10.0,
        'fov': 100,
        'sensor_r_sd': 0.2,  # sensor range noise.
        'sensor_b_sd': 0.01,  # sensor bearing noise.
    },

    # init target's param
    'target': {
        'target_dim': 4,

        'random': False,  # from const_q set
        # target noise constant in beliefs.
        'const_q': [0.5, [0.01, 0.02, 0.05, 0.1, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]],

        'measurement_disfactor': 0.9,
        'target_init_cov': 50.0,  # initial target diagonal Covariance.
        'lin_dist_range_a2t': (3.0, 8.0),
        'ang_dist_range_a2t': (-np.pi / 5, np.pi / 5),
        'lin_dist_range_t2b': (0.0, 2.0),
        'ang_dist_range_t2b': (-np.pi / 2, np.pi / 2),
        'lqr_l_p': 50,  # control the target's veocity
    },

    # reinforcement learning setting.
    'is_training': True,
    'algorithms': 'PPO',  # PPO、SAC
    'policy': 'CNN',  # CNN、MLP 记得同时修改环境
    # # action set
    'action_range_high': [1, 1, 1],
    'action_range_low': [0, -1, -1],
    'action_range_scale': [3/2, np.pi/2, np.pi/4],
    'action_dim': 6,

    # # reward setting
    'reward_param': {
        'c_mean': 0.2,
        'c_std': 0.0,
        'c_penalty': 5.0,
        'k_3': 0.0,  # 0.3,
        'k_4': 0.0,  # 0.01,
        'k_5': 0.0,  # 0.0002,
    },
}

METADATA_v2 = {
    'version': 1,
    'render': True,
    'eval_fixed': False,
    # env set
    'env': {
        'task_random': False,  # insight or not
        'insight': [True, [True, False]],
        'noblock': True,
        'control_period': 0.5,
        'margin': 1.0,  # a marginal distance btw targets and the agent.
        'margin2wall': 1.0,  # a marginal distance from a wall.

        'target_num': 1,
        'target_dim': 4,  # x, y, xdot, ydot
    },

    # init the scenario's param
    'scenario': {
        'size': [40, 40, 20],
        'bottom_corner': [-20, -20, -20],
        'fix_depth': -5,
    },

    # init agent's param
    'agent': {
        'random': False,
        'use_sonar': False,
        'grid': {
            'use_grid': False,
            'p_prior': 0.5,  # Prior occupancy probability
            'p_occ': 0.8,  # Probability that cell is occupied with total confidence
            'p_free': 0.25,  # Probability that cell is free with total confidence
            'resolution': 0.1,  # Grid resolution in [m]
        },
        'sensor_r': 10.0,
        'fov': 100,
        'sensor_r_sd': 0.2,  # sensor range noise.
        'sensor_b_sd': 0.01,  # sensor bearing noise.
    },

    # init target's param
    'target': {
        'target_dim': 4,
        'random': False,  # from const_q set
        # target noise constant in beliefs.
        'const_q': [0.5, [0.01, 0.02, 0.05, 0.1, 0.4, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]],

        'measurement_disfactor': 0.9,
        'target_init_cov': 50.0,  # initial target diagonal Covariance.
        'lin_dist_range_a2t': (3.0, 8.0),
        'ang_dist_range_a2t': (-np.pi / 5, np.pi / 5),
        'lin_dist_range_t2b': (0.0, 2.0),
        'ang_dist_range_t2b': (-np.pi / 2, np.pi / 2),
        'lqr_l_p': 50,  # control the target's veocity
    },

    # reinforcement learning setting.
    'is_training': True,
    'algorithms': 'PPO',  # PPO、SAC
    'policy': 'CNN',  # CNN、MLP 记得同时修改环境
    # action set
    'action_range_high': [1, 1],
    'action_range_low': [-1, -1],
    'action_range_scale': [0.5, 0.2],

    # # reward setting
    'reward_param': {
        'c_mean': 0.2,
        'c_std': 0.0,
        'c_penalty': 5.0,
        'k_3': 0.0,  # 0.3,
        'k_4': 0.0,  # 0.01,
        'k_5': 0.0,  # 0.0002,
    },
}

# Designate a metadata version to be used throughout the target tracking env.
METADATA = {"METADATA_v0": METADATA_v0, "METADATA_v1": METADATA_v1, "METADATA_v1_SAMPLE": METADATA_v1_SAMPLE,
            "METADATA_v2": METADATA_v2}
METADATA = METADATA[choice]

TTENV_EVAL_SET = {
    'Tracking':
        {
            'is_training': False,
            'sensor_r_sd': 0.2,  # sensor range noise.
            'sensor_b_sd': 0.01,  # sensor bearing noise.
            'lin_dist_range_a2t': (3.0, 8.0),
            'ang_dist_range_a2t': (-np.pi / 4, np.pi / 4),
            'lin_dist_range_t2b': (0.0, 3.0),
            'ang_dist_range_t2b': (-np.pi / 2, np.pi / 2),
            'insight': True,
            'noblock': True,
            'const_q': 0.5,
        },
    'Discovery':
        {
            'is_training': False,
            'sensor_r_sd': 0.5,  # sensor range noise.
            'sensor_b_sd': 0.01,  # sensor bearing noise.
            'lin_dist_range_a2b': (3.0, 10.0),
            'ang_dist_range_a2b': (-np.pi, np.pi),
            'lin_dist_range_b2t': (0.0, 3.0),
            'ang_dist_range_b2t': (-np.pi / 2, np.pi / 2),
            'insight': False,
            'noblock': True,
            'const_q': 0.1,
        },
    'Navigation':
        {
            'is_training': False,
            'lin_dist_range_a2b': (35.0, 40.0),
            'ang_dist_range_a2b': (-np.pi / 4, np.pi / 4),
            'lin_dist_range_b2t': (0.0, 3.0),
            'ang_dist_range_b2t': (-np.pi / 4, np.pi / 4),
            'insight': False,
            'noblock': True,
            'target_move': 0.0,
        }
}

TTENV_EVAL_MULTI_SET = [
    {
        'lin_dist_range_a2b': (3.0, 10.0),
        'ang_dist_range_a2b': (-np.pi, np.pi),
        'lin_dist_range_b2t': (0.0, 3.0),
        'ang_dist_range_b2t': (-np.pi / 2, np.pi / 2),
        'blocked': None,
        'const_q': 0.02,
    }
]

# METADATA.update(TTENV_EVAL_SET[2])
print('test')
