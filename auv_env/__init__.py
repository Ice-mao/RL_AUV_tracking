import gymnasium as gym
from .envs.base import TargetTrackingBase
from .envs.world_auv import World_AUV
from .envs.world_auv_map import WorldAuvMap
from .envs.world_auv_rgb_v1 import WorldAuvRGBV1
from .envs.world_auv_rgb_v0 import WorldAuvRGBV0
from .envs.world_auv_rgb_v1_sample import WorldAuvRGBV1Sample
from .envs.world_auv_v2 import WorldAuvV2
from .envs.world_auv_v2_sample import WorldAuvV2Sample
from .envs.world_auv_v2_sample_custom import WorldAuvV2SampleCustom


class TargetTracking1(TargetTrackingBase):
    """
    target is an auv.
    """

    def __init__(self, map="TestMap", num_targets=1, show=True, verbose=True, is_training=False, **kwargs):
        super().__init__(World_AUV, map, num_targets, show, verbose, is_training, **kwargs)

class TargetTracking2(TargetTrackingBase):
    """
    target is an auv with map.
    """

    def __init__(self, map="TestMap_AUV", num_targets=1, show=True, verbose=True, is_training=False, **kwargs):
        super().__init__(WorldAuvMap, map, num_targets, show, verbose, is_training, **kwargs)

class AUVTracking_v0(TargetTrackingBase):
    """
    target is an auv with map.
    """

    def __init__(self, map="AUV_RGB", num_targets=1, show_viewport=True, verbose=True, is_training=False, **kwargs):
        super().__init__(WorldAuvRGBV0, map, num_targets, show_viewport, verbose, is_training, **kwargs)

class AUVTracking_v1(TargetTrackingBase):
    """
    target is an auv with map.
    """

    def __init__(self, map="AUV_RGB", num_targets=1, show_viewport=True, verbose=True, is_training=False, **kwargs):
        super().__init__(WorldAuvRGBV1, map, num_targets, show_viewport, verbose, is_training, **kwargs)

class AUVTracking_v1_sample(TargetTrackingBase):
    """
    target is an auv with map.
    """

    def __init__(self, map="AUV_RGB_Dam", num_targets=1, show_viewport=True, verbose=True, is_training=False, **kwargs):
        super().__init__(WorldAuvRGBV1Sample, map, num_targets, show_viewport, verbose, is_training, **kwargs)

class AUVTracking_v2(TargetTrackingBase):
    """
    target is an auv with map.
    """

    def __init__(self, map="AUV_RGB", num_targets=1, show_viewport=True, verbose=True, is_training=False, **kwargs):
        super().__init__(WorldAuvV2, map, num_targets, show_viewport, verbose, is_training, **kwargs)

class AUVTracking_v2_sample(TargetTrackingBase):
    """
    target is an auv with map.
    """

    def __init__(self, map="AUV_RGB_Dam", num_targets=1, show_viewport=True, verbose=True, is_training=False, **kwargs):
        super().__init__(WorldAuvV2Sample, map, num_targets, show_viewport, verbose, is_training, **kwargs)

class AUVTracking_v2_sample_custom(TargetTrackingBase):
    """
    target is an auv with map.
    """

    def __init__(self, map="AUV_RGB_sonar", num_targets=1, show_viewport=True, verbose=True, is_training=False, **kwargs):
        super().__init__(WorldAuvV2SampleCustom, map, num_targets, show_viewport, verbose, is_training, **kwargs)

def make(env_name, record=False, eval=False, ros=False, directory='../',
         t_steps=100, num_targets=1, **kwargs):
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
    num_targets : int
        the number of targets
    """
    # if T_steps is None:
    #     if num_targets > 1:
    #         T_steps = 150
    #     else:
    #         T_steps = 100
    # T_steps = 200

    local_view = 0
    if env_name == 'TargetTracking1':
        env0 = TargetTracking1(num_targets=num_targets, **kwargs)
    elif env_name == 'TargetTracking2':
        env0 = TargetTracking2(num_targets=num_targets, **kwargs)
    elif env_name == 'AUVTracking_v0':
        env0 = AUVTracking_v0(num_targets=num_targets, **kwargs)
    elif env_name == 'AUVTracking_v1':
        env0 = AUVTracking_v1(num_targets=num_targets, **kwargs)
    elif env_name == 'AUVTracking_v1_sample':
        env0 = AUVTracking_v1_sample(num_targets=num_targets, **kwargs)
    elif env_name == 'AUVTracking_v2':
        env0 = AUVTracking_v2(num_targets=num_targets, **kwargs)
    elif env_name == 'AUVTracking_v2_sample':
        env0 = AUVTracking_v2_sample(num_targets=num_targets, **kwargs)
    elif env_name == 'AUVTracking_v2_sample_custom':
        env0 = AUVTracking_v2_sample_custom(num_targets=num_targets, **kwargs)
    else:
        raise ValueError('No such environment exists.')
    # 使用gym中对episode进行timestep限制的wrapper进行封装，保证环境的更新
    env = gym.wrappers.TimeLimit(env0, max_episode_steps=t_steps)
    if ros:
        from auv_env.wrappers.ros_wrapper import Ros
        env = Ros(env)
    if eval:
        from auv_env.wrappers.display_wrapper import Display2D
        env = Display2D(env)
    if record:
        from auv_env.wrappers.display_wrapper import Video2D
        env = Video2D(env, dirname=directory, local_view=local_view)
    #
    return env


##
# Register Gym environments.
# v0 and v1
##
from .wrappers import TeachObsWrapper, StudentObsWrapper
v0_fns = lambda: TeachObsWrapper(make(env_name='AUVTracking_v0',
                                           record=False,
                                           show_viewport=True,
                                           num_targets=1,
                                           is_training=False,
                                           eval=False,
                                           t_steps=200,
                                           ))
v1_teacher_fns = lambda: TeachObsWrapper(make(env_name='AUVTracking_v1',
                                           record=False,
                                           show_viewport=True,
                                           num_targets=1,
                                           is_training=False,
                                           eval=False,
                                           t_steps=200,
                                           ))
v1_teacher_fns_norender = lambda: TeachObsWrapper(make(env_name='AUVTracking_v1',
                                                  record=False,
                                                  show_viewport=False,
                                                  num_targets=1,
                                                  is_training=True,
                                                  eval=False,
                                                  t_steps=200,
                                                  ))
v1_teacher_fns_render = lambda: TeachObsWrapper(make(env_name='AUVTracking_v1',
                                                  record=False,
                                                  show_viewport=True,
                                                  num_targets=1,
                                                  is_training=True,
                                                  eval=True,
                                                  t_steps=200,
                                                  ))
v1_student_fns = lambda: StudentObsWrapper(make(env_name='AUVTracking_v1',
                                             record=False,
                                             show_viewport=True,
                                             num_targets=1,
                                             is_training=False,
                                             eval=False,
                                             t_steps=200,
                                             ))
v1_student_fns_norender = lambda: StudentObsWrapper(make(env_name='AUVTracking_v1',
                                                    record=False,
                                                    show_viewport=False,
                                                    num_targets=1,
                                                    is_training=False,
                                                    eval=False,
                                                    t_steps=200,
                                                    ))
v1_sample_fns = lambda: make(env_name='AUVTracking_v1_sample',
                        record=False,
                        num_targets=1,
                        is_training=False,
                        eval=False,
                        t_steps=200,
                        )
v1_sample_fns_teacher = lambda: TeachObsWrapper(make(env_name='AUVTracking_v1_sample',
                        record=False,
                        num_targets=1,
                        is_training=False,
                        eval=False,
                        t_steps=200,
                        ))
gym.register(
    id="auv-v0",
    entry_point=v0_fns,
    disable_env_checker=True,
)
gym.register(
    id="v1-Teacher",
    entry_point=v1_teacher_fns,
    disable_env_checker=True,
)
gym.register(
    id="v1-Teacher-norender",
    entry_point=v1_teacher_fns_norender,
    disable_env_checker=True,
)
gym.register(
    id="v1-Teacher-render",
    entry_point=v1_teacher_fns_render,
    disable_env_checker=True,
)
gym.register(
    id="v1-Student",
    entry_point=v1_student_fns,
    disable_env_checker=True,
)
gym.register(
    id="v1-Student-norender",
    entry_point=v1_student_fns_norender,
    disable_env_checker=True,
)
gym.register(
    id="v1-Student-sample",
    entry_point=v1_sample_fns,
    disable_env_checker=True,
)
gym.register(
    id="v1-Student-sample-teacher",
    entry_point=v1_sample_fns_teacher,
    disable_env_checker=True,
)

v2_teacher_fns = lambda: TeachObsWrapper(make(env_name='AUVTracking_v2',
                                           record=False,
                                           show_viewport=True,
                                           num_targets=1,
                                           is_training=False,
                                           eval=False,
                                           t_steps=200,
                                           ))
v2_teacher_fns_norender = lambda: TeachObsWrapper(make(env_name='AUVTracking_v2',
                                                  record=False,
                                                  show_viewport=False,
                                                  num_targets=1,
                                                  is_training=True,
                                                  eval=False,
                                                  t_steps=200,
                                                  ))
v2_teacher_fns_render = lambda: TeachObsWrapper(make(env_name='AUVTracking_v2',
                                                  record=False,
                                                  show_viewport=True,
                                                  num_targets=1,
                                                  is_training=True,
                                                  eval=True,
                                                  t_steps=200,
                                                  ))
v2_sample_fns = lambda: make(env_name='AUVTracking_v2_sample',
                        record=False,
                        num_targets=1,
                        is_training=False,
                        eval=False,
                        t_steps=200,
                        map="AUV_RGB_Dam_sonar",
                        # map="AUV_RGB_OpenWater_sonar",
                        )
v2_sample_custom_fns = lambda: make(env_name='AUVTracking_v2_sample_custom',
                        record=False,
                        num_targets=1,
                        is_training=False,
                        eval=False,
                        t_steps=200,
                        map="AUV_RGB_sonar",
                        )
v2_sample_teacher_fns = lambda: TeachObsWrapper(make(env_name='AUVTracking_v2_sample',
                        record=False,
                        show_viewport=False,
                        num_targets=1,
                        is_training=False,
                        eval=False,
                        t_steps=200,
                        map="AUV_RGB_Dam_sonar",
                        ))
gym.register(
    id="v2-Teacher",
    entry_point=v2_teacher_fns,
    disable_env_checker=True,
)
gym.register(
    id="v2-Teacher-norender",
    entry_point=v2_teacher_fns_norender,
    disable_env_checker=True,
)
gym.register(
    id="v2-Teacher-render",
    entry_point=v2_teacher_fns_render,
    disable_env_checker=True,
)
gym.register(
    id="v2-sample-render",
    entry_point=v2_sample_fns,
    disable_env_checker=True,
)
gym.register(
    id="v2-sample-custom-render",
    entry_point=v2_sample_custom_fns,
    disable_env_checker=True,
)
gym.register(
    id="v2-sample-teacher-wrapper",
    entry_point=v2_sample_teacher_fns,
    disable_env_checker=True,
)



