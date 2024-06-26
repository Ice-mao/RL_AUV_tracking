from gymnasium import wrappers
from .base import TargetTrackingBase, TargetTracking1
def make(env_name, render=False, record=False, eval=False, ros=False, directory='',
         t_steps=100, num_targets=1, **kwargs):
    """
    Parameters:
    ----------
    env_name : str
        name of an environment. (e.g. 'TargetTracking-v0')
    render : bool
        wether to render.
    figID : int
        figure ID for rendering and/or recording.
    record : bool
        whether to record a video.
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
    if env_name == 'TargetTracking':
        env0 = TargetTrackingBase(num_targets=num_targets, **kwargs)
    elif env_name == 'TargetTracking1':
        env0 = TargetTracking1(num_targets=num_targets, **kwargs)
    else:
        raise ValueError('No such environment exists.')
    # 使用gym中对episode进行timestep限制的wrapper进行封装，保证环境的更新
    env = wrappers.TimeLimit(env0, max_episode_steps=t_steps)
    if ros:
        from auv_env.ros_wrapper import Ros
        env = Ros(env)
    if eval:
        from auv_env.display_wrapper import Display2D
        env = Display2D(env)
    if record:
        from auv_env.display_wrapper import Video2D
        env = Video2D(env, dirname=directory, local_view=local_view)

    return env
