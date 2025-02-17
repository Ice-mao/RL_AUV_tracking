import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.sac.policies import Actor
env = gym.make("BipedalWalker-v3", render_mode="rgb_array")

model = SAC.load("bipedal_trained.zip", env, device='cuda')

vec_env = model.get_env()
obs = vec_env.reset()
total_reward = 0
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    total_reward += reward
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
print(total_reward)