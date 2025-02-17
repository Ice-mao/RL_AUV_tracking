from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("Pendulum-v1")
expert = SAC(
    policy=MlpPolicy,
    env=env
)
expert.learn(1000000)
print("debug end")