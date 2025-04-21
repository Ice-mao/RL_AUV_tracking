import holoocean, cv2

env = holoocean.make("Dam-HoveringCamera")
env.act('auv0', [10,10,10,10,0,0,0,0])

for _ in range(200):
    state = env.tick()

    if "LeftCamera" in state:
        pixels = state["LeftCamera"]
        cv2.namedWindow("Camera Output")
        cv2.imshow("Camera Output", pixels[:, :, 0:3])
        cv2.waitKey(0)
        cv2.destroyAllWindows()