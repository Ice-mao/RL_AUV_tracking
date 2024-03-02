from auv_env.util import transform_2d_inv
import numpy as np
rand_xy_global = transform_2d_inv([0, 5], np.pi/2, [0, 0])
print(rand_xy_global)