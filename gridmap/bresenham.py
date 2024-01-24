#!/usr/bin/env python

import numpy as np

def bresenham(gridMap, x1, y1, x2, y2):
	"""
	Bresenham's line drawing algorithm - working for all 4 quadrants!
	"""
	
	# Output pixels
	X_bres = []
	Y_bres = []

	x = x1
	y = y1
	
	delta_x = np.abs(x2 - x1)
	delta_y = np.abs(y2 - y1)
	
	s_x = np.sign(x2 - x1)
	s_y = np.sign(y2 - y1)

	if delta_y > delta_x:

		delta_x, delta_y = delta_y, delta_x
		interchange = True

	else:

		interchange = False

	A = 2 * delta_y
	B = 2 * (delta_y - delta_x)
	E = 2 * delta_y - delta_x

	# mark output pixels
	X_bres.append(x)
	Y_bres.append(y)

	# point (x2,y2) must not be included
	for i in range(1, delta_x):

		if E < 0:

			if interchange:

				y += s_y
			
			else:

				x += s_x

			E = E + A

		else:

			y += s_y
			x += s_x
			E = E + B

		# mark output pixels
		X_bres.append(x)
		Y_bres.append(y)

	return zip(X_bres, Y_bres)