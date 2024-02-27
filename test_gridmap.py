#!/usr/bin/env python

from time import perf_counter

import holoocean
import matplotlib.pyplot as plt

from gridmap.grid_map import *

from auv_env.tools import KeyBoardCmd, ImagingSonar, PoseLocation

P_prior = 0.5  # Prior occupancy probability
P_occ = 0.9  # Probability that cell is occupied with total confidence
P_free = 0.35  # Probability that cell is free with total confidence

RESOLUTION = 0.2  # Grid resolution in [m]


if __name__ == '__main__':

    ##################### Init section #####################

    # Init env
    scenario = "Demonstration-HoveringCamera"
    env = holoocean.make(scenario)

    # Init keyboard control
    kb_cmd = KeyBoardCmd(force=50)
    # Init gridmap
    map_x_lim = [-12, 12]
    map_y_lim = [-12, 12]
    dir_pointer_len = 1

    # Create grid map
    gridMap = GridMap(X_lim=map_x_lim,
                        Y_lim=map_y_lim,
                        resolution=RESOLUTION,
                        p=P_prior)

    # Init time
    t_start = perf_counter()
    sim_time = 0
    step = 0
    init_flag = 1

    ##################### Main loop #####################
    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(111)
    # env.spawn_prop(prop_type='cylinder', location=[31.5, 95, -37], scale=[2,2,2], sim_physics=0, material='steel')
    # env.draw_box(center=[31.5, 95, -37],extent=[1,1,1],lifetime=0)
    # env.weather.stop_day_cycle()
    imagingsonar = ImagingSonar(scenario)
    poselocation = PoseLocation()
    while True:
        if 'q' in kb_cmd.pressed_keys:
            break
        command = kb_cmd.parse_keys()

        # send to holoocean
        if init_flag == 1:
            for i in range(10):
                env.act("auv0", command)
                env.tick()
            init_flag = 0

        env.act("auv0", command)
        state = env.tick()
        imagingsonar.draw_pic(state)
        poselocation.update(state)


        # get the agent's pose
        x_odom, y_odom = poselocation.locationxy  # x,y in [m]
        theta_odom = poselocation.direction  # rad

        distances_x, distances_y, distances, nearest_x, nearest_y, nearest_dist = imagingsonar.scan(state, [x_odom, y_odom], poselocation.angle)
        filtered_distances_x = []
        filtered_distances_y = []

        ##################### Grid map update section #####################
        # 机器人当前坐标(x1, y1)
        x1, y1 = gridMap.discretize(x_odom, y_odom)
        # for BGR image of the grid map
        X2 = []
        Y2 = []

        # 类似lidar的原理更新free space
        for (dist_x, dist_y, dist) in zip(nearest_x, nearest_y, nearest_dist):
            # 障碍物的坐标(x2, y2)
            x2, y2 = gridMap.discretize(dist_x, dist_y)
            # draw a discrete line of free pixels, [robot position -> laser hit spot), 确定测量范围内的free space
            for (x_bres, y_bres) in bresenham(gridMap, x1, y1, x2, y2):
                gridMap.update(x=x_bres, y=y_bres, p=P_free)
            # for BGR image of the grid map
            X2.append(x2)
            Y2.append(y2)


        # 更新occ space
        for (dist_x, dist_y, dist) in zip(nearest_x, nearest_y, nearest_dist):
            if dist < imagingsonar.zmax:
                # 障碍物的坐标(x2, y2)
                x2, y2 = gridMap.discretize(dist_x, dist_y)

                # 检测到障碍物，更新occ grid map
                gridMap.update(x=x2, y=y2, p=P_occ)

                # filtered distances in X-Y plane for Ploting
                filtered_distances_x.append(dist_x)
                filtered_distances_y.append(dist_y)
        ############# 整体更新的方法，待调试 ############
        # if imagingsonar.getimage is True:
        #     for i in range(imagingsonar.binsR):
        #         for j in range(imagingsonar.binsA):
        #             x2, y2 = gridMap.discretize(imagingsonar.s2map[i, j][0], imagingsonar.s2map[i, j][1])
        #             if imagingsonar.s[i, j] == 0:  # free space
        #                 gridMap.update(x=x2, y=y2, p=P_free)
        #             else:
        #                 # 检测到障碍物，更新occ grid map
        #                 gridMap.update(x=x2, y=y2, p=P_occ)
        #
        #                 # filtered distances in X-Y plane for Ploting
        #                 filtered_distances_x.append(x2)
        #                 filtered_distances_y.append(y2)


        # for (dist_x, dist_y, dist) in zip(distances_x, distances_y, distances):

        #     # 障碍物的坐标(x2, y2)
        #     x2, y2 = gridMap.discretize(dist_x, dist_y)

        #     # 检测到障碍物，更新occ grid map
        #     gridMap.update(x=x2, y=y2, p=P_occ)

        #     # filtered distances in X-Y plane for Ploting
        #     filtered_distances_x.append(dist_x)
        #     filtered_distances_y.append(dist_y)


        # converting grip map to BGR image
        bgr_image = gridMap.to_BGR_image()

        # marking robot position with blue pixel value
        set_pixel_color(bgr_image, x1, y1, 'BLUE')

        # marking neighbouring pixels with blue pixel value
        for (x, y) in gridMap.find_neighbours(x1, y1):
            set_pixel_color(bgr_image, x, y, 'BLUE')
            for (x, y) in gridMap.find_neighbours(x, y):
                set_pixel_color(bgr_image, x, y, 'BLUE')

        # marking laser hit spots with green value
        # for (x, y) in zip(X2, Y2):
        #     set_pixel_color(bgr_image, x, y, 'GREEN')

        resized_image = cv2.resize(src=bgr_image,
                                    dsize=(500, 500),
                                    interpolation=cv2.INTER_AREA)

        rotated_image = cv2.rotate(src=resized_image,
                                    rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)

        cv2.imshow("Grid map", rotated_image)
        cv2.waitKey(1)

