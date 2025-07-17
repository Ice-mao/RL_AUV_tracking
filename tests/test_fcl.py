import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import numpy as np
import fcl

g1 = fcl.Box(1,2,3)
t1 = fcl.Transform()
o1 = fcl.CollisionObject(g1, t1)

g2 = fcl.Cone(1,3)
t2 = fcl.Transform()
o2 = fcl.CollisionObject(g2, t2)

request = fcl.CollisionRequest()
result = fcl.CollisionResult()

ret = fcl.collide(o1, o2, request, result)
print(ret)