import gymnasium as gym

# tools


# observation_space = spaces.Dict({
#             "images": spaces.Dict(
#                 {
#                     "t-4": spaces.Box(-3, 3, shape=(3, 224, 224), dtype=np.float32),
#                     "t_3": spaces.Box(-3, 3, shape=(3, 224, 224), dtype=np.float32),
#                     "t_2": spaces.Box(-3, 3, shape=(3, 224, 224), dtype=np.float32),
#                     "t_1": spaces.Box(-3, 3, shape=(3, 224, 224), dtype=np.float32),
#                     "t": spaces.Box(-3, 3, shape=(3, 224, 224), dtype=np.float32),
#                 }
#             ),
#             "state": spaces.Box(low=-1, high=1, dtype=np.float32),
#         })
# env = SubprocVecEnv([lambda: gym.make('Teacher-v0') for _ in range(1)])
env = gym.make('Student-v0')
# env = StudentObsWrapper(auv_env.make(env_name='AUVTracking_rgb',
#                                              render=False,
#                                              record=False,
#                                              num_targets=1,
#                                              is_training=False,
#                                              eval=False,
#                                              t_steps=200,
#                                              ))
obs= env.reset()

# a = env.reset()
while True:
    action = env.action_space.sample()
    print(action)
    # action = np.array([0.0, 0.0, 0.0])
    obs, reward, done, _, inf = env.step(action)


