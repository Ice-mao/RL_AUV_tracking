import gymnasium as gym
from .envs.base import TargetTrackingBase
from .envs.world_auv import World_AUV
from .envs.world_auv_map import WorldAuvMap
from .envs.world_auv_rgb import WorldAuvRGB
from .envs.world_auv_rgb_v0 import WorldAuvRGBV0
from .envs.world_auv_rgb_sample import WorldAuvRGBSample


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


class AUVTracking_rgb(TargetTrackingBase):
    """
    target is an auv with map.
    """

    def __init__(self, map="AUV_RGB", num_targets=1, show_viewport=True, verbose=True, is_training=False, **kwargs):
        super().__init__(WorldAuvRGB, map, num_targets, show_viewport, verbose, is_training, **kwargs)

class AUVTracking_rgb_v0(TargetTrackingBase):
    """
    target is an auv with map.
    """

    def __init__(self, map="AUV_RGB", num_targets=1, show_viewport=True, verbose=True, is_training=False, **kwargs):
        super().__init__(WorldAuvRGBV0, map, num_targets, show_viewport, verbose, is_training, **kwargs)

class AUVTracking_rgb_sample(TargetTrackingBase):
    """
    target is an auv with map.
    """

    def __init__(self, map="AUV_RGB_Dam", num_targets=1, show_viewport=True, verbose=True, is_training=False, **kwargs):
        super().__init__(WorldAuvRGBSample, map, num_targets, show_viewport, verbose, is_training, **kwargs)


def make(env_name, render=False, record=False, eval=False, ros=False, directory='../',
         t_steps=100, num_targets=1, **kwargs):
    """
    Parameters:
    ----------
    env_name : str
        name of an environment. (e.g. 'TargetTracking-v0')
    render : bool
        wether to render the ue5 .
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
    elif env_name == 'AUVTracking_rgb':
        env0 = AUVTracking_rgb(num_targets=num_targets, **kwargs)
    elif env_name == 'AUVTracking_rgb_v0':
        env0 = AUVTracking_rgb_v0(num_targets=num_targets, **kwargs)
    elif env_name == 'AUVTracking_rgb_sample':
        env0 = AUVTracking_rgb_sample(num_targets=num_targets, **kwargs)
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
##
from .wrappers import TeachObsWrapper, StudentObsWrapper

fns = lambda: make(env_name='AUVTracking_rgb',
                   render=False,
                   record=False,
                   num_targets=1,
                   is_training=False,
                   eval=False,
                   t_steps=200,
                   )
teacher_fns_v0 = lambda: TeachObsWrapper(make(env_name='AUVTracking_rgb_v0',
                                           render=False,
                                           record=False,
                                           show_viewport=True,
                                           num_targets=1,
                                           is_training=False,
                                           eval=False,
                                           t_steps=200,
                                           ))
teacher_fns = lambda: TeachObsWrapper(make(env_name='AUVTracking_rgb',
                                           render=False,
                                           record=False,
                                           show_viewport=True,
                                           num_targets=1,
                                           is_training=False,
                                           eval=False,
                                           t_steps=200,
                                           ))
teacher_fns_norender = lambda: TeachObsWrapper(make(env_name='AUVTracking_rgb',
                                                  render=False,
                                                  record=False,
                                                  show_viewport=False,
                                                  num_targets=1,
                                                  is_training=False,
                                                  eval=False,
                                                  t_steps=200,
                                                  ))
student_fns = lambda: StudentObsWrapper(make(env_name='AUVTracking_rgb',
                                             render=False,
                                             record=False,
                                             show_viewport=True,
                                             num_targets=1,
                                             is_training=False,
                                             eval=False,
                                             t_steps=200,
                                             ))
student_fns_norender = lambda: StudentObsWrapper(make(env_name='AUVTracking_rgb',
                                                    render=False,
                                                    record=False,
                                                    show_viewport=False,
                                                    num_targets=1,
                                                    is_training=False,
                                                    eval=False,
                                                    t_steps=200,
                                                    ))
sample_fns = lambda: make(env_name='AUVTracking_rgb_sample',
                        render=True,
                        record=False,
                        num_targets=1,
                        is_training=False,
                        eval=False,
                        t_steps=200,
                        )
sample_fns_teacher = lambda: TeachObsWrapper(make(env_name='AUVTracking_rgb_sample',
                        render=False,
                        record=False,
                        num_targets=1,
                        is_training=False,
                        eval=False,
                        t_steps=200,
                        ))
gym.register(
    id="auv_rgb-v0",
    entry_point=fns,
    disable_env_checker=True,
)
gym.register(
    id="Teacher-v0",
    entry_point=teacher_fns_v0,
    disable_env_checker=True,
)
gym.register(
    id="Teacher-v1",
    entry_point=teacher_fns,
    disable_env_checker=True,
)
gym.register(
    id="Teacher-v1-norender",
    entry_point=teacher_fns_norender,
    disable_env_checker=True,
)
gym.register(
    id="Student-v0",
    entry_point=student_fns,
    disable_env_checker=True,
)
gym.register(
    id="Student-v0-norender",
    entry_point=student_fns_norender,
    disable_env_checker=True,
)
gym.register(
    id="Student-v0-sample",
    entry_point=sample_fns,
    disable_env_checker=True,
)
gym.register(
    id="Student-v0-sample-teacher",
    entry_point=sample_fns_teacher,
    disable_env_checker=True,
)

