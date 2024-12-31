import gymnasium as gym
from gymnasium.spaces import flatten_space, flatten


class TeachObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper extract the state of observation space for RL
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space["state"]

    def observation(self, obs):
        return obs["state"]


class StudentObsWrapper(gym.ObservationWrapper):
    """
    This observation wrapper extract the image of observation space for RL
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = self.env.observation_space["images"]

    def observation(self, obs):
        return obs["images"]
