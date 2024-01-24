#!/usr/bin/env python

import rospy

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

import sys

SCRIPTS_PATH = '/home/maestro/catkin_ws/src/grid_mapping/scripts'
MAPS_PATH = '/home/maestro/catkin_ws/src/grid_mapping/maps'
sys.path.insert(0, SCRIPTS_PATH)

from grid_map import *
from utils import *

P_prior = 0.5	# Prior occupancy probability
P_occ = 0.9	# Probability that cell is occupied with total confidence
P_free = 0.3	# Probability that cell is free with total confidence 

RESOLUTION = 0.03 # Grid resolution in [m]

MAP_NAME  = 'world' # map name without extension

if __name__ == '__main__':

	try:

		# Init map parameters
		if MAP_NAME[:5] == 'stage':

			map_x_lim = [-3, 3]
			map_y_lim = [-3, 3]

		elif MAP_NAME[:5] == 'world':

			map_x_lim = [-4, 4]
			map_y_lim = [-4, 4]

		else:

			map_x_lim = [-10, 10]
			map_y_lim = [-6, 6]

		# Init ROS node
		rospy.init_node('gmapping_node', anonymous = False)
		rate = rospy.Rate(10)

		# Create grid map 
		gridMap = GridMap(X_lim = map_x_lim, 
				  Y_lim = map_y_lim, 
				  resolution = RESOLUTION, 
				  p = P_prior)

		# Init time
		t_start = perf_counter()
		sim_time = 0
		step = 0

		# Main loop
		while not rospy.is_shutdown():

			# Lidar measurements
			msgScan = rospy.wait_for_message('/scan', LaserScan)
			distances, angles, information = lidar_scan(msgScan)  # distances in [m], angles in [radians]

			# Odometry measurements
			msgOdom = rospy.wait_for_message('/odom', Odometry)
			x_odom, y_odom = get_odom_position(msgOdom)   # x,y in [m]
			theta_odom = get_odom_orientation(msgOdom)    # theta in [radians]

			# Lidar measurements in X-Y plane
			distances_x, distances_y = lidar_scan_xy(distances, angles, x_odom, y_odom, theta_odom)

			# x1 and y1 for Bresenham's algorithm
			x1, y1 = gridMap.discretize(x_odom, y_odom)

			# for BGR image of the grid map
			X2 = []
			Y2 = []

			for (dist_x, dist_y, dist) in zip(distances_x, distances_y, distances):

				# x2 and y2 for Bresenham's algorithm
				x2, y2 = gridMap.discretize(dist_x, dist_y)

				# draw a discrete line of free pixels, [robot position -> laser hit spot)
				for (x_bres, y_bres) in bresenham(gridMap, x1, y1, x2, y2):

					gridMap.update(x = x_bres, y = y_bres, p = P_free)

				# mark laser hit spot as ocuppied (if exists)
				if dist < msgScan.range_max:
					
					gridMap.update(x = x2, y = y2, p = P_occ)

				# for BGR image of the grid map
				X2.append(x2)
				Y2.append(y2)

			# converting grip map to BGR image
			bgr_image = gridMap.to_BGR_image()

			# marking robot position with blue pixel value
			set_pixel_color(bgr_image, x1, y1, 'BLUE')
			
			# marking neighbouring pixels with blue pixel value 
			for (x, y) in gridMap.find_neighbours(x1, y1):
				set_pixel_color(bgr_image, x, y, 'BLUE')

			# marking laser hit spots with green value
			for (x, y) in zip(X2,Y2):
				set_pixel_color(bgr_image, x, y, 'GREEN')

			resized_image = cv2.resize(src = bgr_image, 
						   dsize = (500, 500), 
						   interpolation = cv2.INTER_AREA)

			rotated_image = cv2.rotate(src = resized_image, 
						   rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)

			cv2.imshow("Grid map", rotated_image)
			cv2.waitKey(1)

			# Calculate step time in [s]
			t_step = perf_counter()
			step_time = t_step - t_start
			sim_time += step_time
			t_start = t_step
			step += 1 

			print('Step %d ==> %d [ms]' % (step, step_time * 1000))

			rate.sleep()

	except rospy.ROSInterruptException:

		print('\r\nSIMULATION TERMINATED!')
		print('\nSimulation time: %.2f [s]' % sim_time)
		print('Average step time: %d [ms]' % (sim_time * 1000 / step))
		print('Frames per second: %.1f' % (step / sim_time))

		# Saving Grid Map
		resized_image = cv2.resize(src = gridMap.to_BGR_image(), 
					   dsize = (500, 500), 
					   interpolation = cv2.INTER_AREA)

		rotated_image = cv2.rotate(src = resized_image, 
					   rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)

		flag_1 = cv2.imwrite(img = rotated_image * 255.0, 
				     filename = MAPS_PATH + '/' + MAP_NAME + '_grid_map_TEST.png')

		# Calculating Maximum likelihood estimate of the map
		gridMap.calc_MLE()

		# Saving MLE of the Grid Map
		resized_image_MLE = cv2.resize(src = gridMap.to_BGR_image(), 
					       dsize = (500, 500), 
					       interpolation = cv2.INTER_AREA)

		rotated_image_MLE = cv2.rotate(src = resized_image_MLE, 
					       rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE)

		flag_2 = cv2.imwrite(img = rotated_image_MLE * 255.0, 
				     filename = MAPS_PATH + '/' + MAP_NAME + '_grid_map_TEST_mle.png')

		if flag_1 and flag_2:
			print('\nGrid map successfully saved!\n')

		if cv2.waitKey(0) == 27:
			cv2.destroyAllWindows()

		pass
