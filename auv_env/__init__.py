import gymnasium as gym
from .envs.base import TargetTrackingBase
from .envs.world_auv_v0 import WorldAuvV0
from .envs.world_auv_v1 import WorldAuvV1
from .envs.world_auv_v2 import WorldAuvV2
from .envs.world_auv_v0_3d import WorldAuv3DV0
from .envs.world_auv_v1_3d import WorldAuv3DV1
from .envs.world_auv_v0_sample import WorldAuvV0Sample

def make(env_name, config=None, eval:bool=False, t_steps=100, show_viewport=True, **kwargs):
    """
    Parameters:
    ----------
    env_name : str
        name of an environment. (e.g. 'TargetTracking-v0')
    figID : int
        figure ID for rendering and/or recording.
    record : bool
        whether to record a video.
    eval : bool
        whether to show the running process.
    ros : bool
        whether to use ROS.
    directory :str
        a path to store a video file if record is True.
    T_steps : int
        the number of steps per episode.
    """
    world_map = {
        'AUVTracking_v0': (WorldAuvV0, "SimpleUnderwater-Bluerov2", "../configs/envs/v0_config.yml"),
        'AUVTracking_v1': (WorldAuvV1, "SimpleUnderwater-Bluerov2_RGB", "../configs/envs/v1_config.yml"),
        'AUVTracking_v2': (WorldAuvV2, "SimpleUnderwater-Bluerov2_sonar", "../configs/envs/v2_config.yml"),
        'AUVTracking3D_v0': (WorldAuv3DV0, "SimpleUnderwater-Bluerov2", "../configs/envs/3d_v0_config.yml"),
        'AUVTracking3D_v1': (WorldAuv3DV1, "SimpleUnderwater-Bluerov2_RGB", "../configs/envs/3d_v1_config.yml"),
        'AUVTracking_v0_sample': (WorldAuvV0Sample, "SimpleUnderwater-Bluerov2_RGB", "../configs/envs/v0_sample_config.yml"),
    }

    if env_name not in world_map:
        raise ValueError('No such environment exists.')

    world_class, default_map, default_config_path = world_map[env_name]
    if config is None:
        from config_loader import load_config
        config = load_config(default_config_path)
    if 'map' not in config:
        map_name = default_map
    else:
        map_name = config['map']
    env0 = TargetTrackingBase(world_class, map_name, show_viewport, config, **kwargs)

    env = gym.wrappers.TimeLimit(env0, max_episode_steps=t_steps)
    if eval:
        print(eval)
        if '3D' in env_name:
            from auv_env.wrappers.display_wrapper import Display3D
            env = Display3D(env)
            print("Using Display3D wrapper for 3D environment.")
        else:
            from auv_env.wrappers.display_wrapper import Display2D
            env = Display2D(env)
    # if record:
    #     from auv_env.wrappers.display_wrapper import Video2D
    #     env = Video2D(env, dirname=directory)
    return env


##
# Register Gym environments.
# v0 and v1
##
# from .wrappers import TeachObsWrapper, StudentObsWrapper
# v1_teacher_fns = lambda: TeachObsWrapper(make(env_name='AUVTracking_v1',
#                                            record=False,
#                                            show_viewport=True,
#                                            num_targets=1,
#                                            eval=False,
#                                            t_steps=200,
#                                            ))
# v1_teacher_fns_norender = lambda: TeachObsWrapper(make(env_name='AUVTracking_v1',
#                                                   record=False,
#                                                   show_viewport=False,
#                                                   num_targets=1,
#                                                   eval=False,
#                                                   t_steps=200,
#                                                   ))
# v1_teacher_fns_render = lambda: TeachObsWrapper(make(env_name='AUVTracking_v1',
#                                                   record=False,
#                                                   show_viewport=True,
#                                                   num_targets=1,
#                                                   eval=True,
#                                                   t_steps=200,
#                                                   ))

# v1_student_fns = lambda: StudentObsWrapper(make(env_name='AUVTracking_v1',
#                                              record=False,
#                                              show_viewport=True,
#                                              num_targets=1,
#                                              eval=False,
#                                              t_steps=200,
#                                              ))
# v1_student_fns_norender = lambda: StudentObsWrapper(make(env_name='AUVTracking_v1',
#                                                     record=False,
#                                                     show_viewport=False,
#                                                     num_targets=1,
#                                                     is_training=False,
#                                                     eval=False,
#                                                     t_steps=200,
#                                                     ))
# v1_sample_fns = lambda: make(env_name='AUVTracking_v1_sample',
#                         record=False,
#                         num_targets=1,
#                         is_training=False,
#                         eval=False,
#                         t_steps=200,
#                         )
# v1_sample_fns_teacher = lambda: TeachObsWrapper(make(env_name='AUVTracking_v1_sample',
#                         record=False,
#                         num_targets=1,
#                         is_training=False,
#                         eval=False,
#                         t_steps=200,
#                         ))
# gym.register(
#     id="v1-state",
#     entry_point=v1_teacher_fns,
#     disable_env_checker=True,
# )
# gym.register(
#     id="v1-state-norender",
#     entry_point=v1_teacher_fns_norender,
#     disable_env_checker=True,
# )
# gym.register(
#     id="v1-Teacher-render",
#     entry_point=v1_teacher_fns_render,
#     disable_env_checker=True,
# )
# gym.register(
#     id="v1-Student",
#     entry_point=v1_student_fns,
#     disable_env_checker=True,
# )
# gym.register(
#     id="v1-Student-norender",
#     entry_point=v1_student_fns_norender,
#     disable_env_checker=True,
# )
# gym.register(
#     id="v1-Student-sample",
#     entry_point=v1_sample_fns,
#     disable_env_checker=True,
# )
# gym.register(
#     id="v1-Student-sample-teacher",
#     entry_point=v1_sample_fns_teacher,
#     disable_env_checker=True,
# )

# v2_teacher_fns = lambda: TeachObsWrapper(make(env_name='AUVTracking_v2',
#                                            record=False,
#                                            show_viewport=True,
#                                            num_targets=1,
#                                            is_training=False,
#                                            eval=False,
#                                            t_steps=200,
#                                            ))
# v2_teacher_fns_norender = lambda: TeachObsWrapper(make(env_name='AUVTracking_v2',
#                                                   record=False,
#                                                   show_viewport=False,
#                                                   num_targets=1,
#                                                   is_training=True,
#                                                   eval=False,
#                                                   t_steps=200,
#                                                   ))
# v2_teacher_fns_render = lambda: TeachObsWrapper(make(env_name='AUVTracking_v2',
#                                                   record=False,
#                                                   show_viewport=True,
#                                                   num_targets=1,
#                                                   is_training=True,
#                                                   eval=True,
#                                                   t_steps=200,
#                                                   ))
# v2_sample_fns = lambda: make(env_name='AUVTracking_v2_sample',
#                         record=False,
#                         show_viewport=True,
#                         num_targets=1,
#                         is_training=False,
#                         eval=False,
#                         t_steps=200,
#                         map="AUV_RGB_Dam_sonar",
#                         # map="AUV_RGB_OpenWater_sonar",
#                         )
# v2_sample_custom_fns = lambda: make(env_name='AUVTracking_v2_sample_custom',
#                         record=False,
#                         num_targets=1,
#                         is_training=False,
#                         eval=False,
#                         t_steps=200,
#                         map="AUV_RGB_sonar",
#                         )
# v2_sample_teacher_fns = lambda: TeachObsWrapper(make(env_name='AUVTracking_v2_sample',
#                         record=False,
#                         show_viewport=False,
#                         num_targets=1,
#                         is_training=False,
#                         eval=False,
#                         t_steps=200,
#                         map="AUV_RGB_Dam_sonar",
#                         ))
# gym.register(
#     id="v2-Teacher",
#     entry_point=v2_teacher_fns,
#     disable_env_checker=True,
# )
# gym.register(
#     id="v2-Teacher-norender",
#     entry_point=v2_teacher_fns_norender,
#     disable_env_checker=True,
# )
# gym.register(
#     id="v2-Teacher-render",
#     entry_point=v2_teacher_fns_render,
#     disable_env_checker=True,
# )
# gym.register(
#     id="v2-sample-render",
#     entry_point=v2_sample_fns,
#     disable_env_checker=True,
# )
# gym.register(
#     id="v2-sample-custom-render",
#     entry_point=v2_sample_custom_fns,
#     disable_env_checker=True,
# )
# gym.register(
#     id="v2-sample-teacher-wrapper",
#     entry_point=v2_sample_teacher_fns,
#     disable_env_checker=True,
# )



