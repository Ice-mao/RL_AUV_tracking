import auv_env
import numpy as np

env = auv_env.make("AUVTracking_rgb",
      render=1,
      num_targets=1,
      map="AUV_RGB",
      # map="TestMap_AUV",
      is_training=True,
      t_steps=200,
      )

obs, _ = env.reset()
while True:
    action = env.action_space.sample()
    print(action)
    # action = np.array([0.0, 0.0, 0.0])
    obs, reward, done, _, inf = env.step(action)