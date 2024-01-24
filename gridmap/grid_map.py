#!/usr/bin/env python

import numpy as np
import sys
import cv2

from gridmap.bresenham import *

TRESHOLD_P_FREE = 0.3
TRESHOLD_P_OCC = 0.6

def log_odds(p):
	"""
	Log odds ratio of p(x):

					p(x)
	 l(x) = log ----------
		     	1 - p(x)

	"""
	return np.log(p / (1 - p))


def retrieve_p(l):
	"""
	Retrieve p(x) from log odds ratio:

	 		   1
	 p(x) = 1 - ---------------
		     1 + exp(l(x))

	"""
	return 1 - 1 / (1 + np.exp(l))

class GridMap:
	"""
	Grid map
	"""
	def __init__(self, X_lim, Y_lim, resolution, p):

		self.X_lim = X_lim
		self.Y_lim = Y_lim
		self.resolution = resolution

		x = np.arange(start = X_lim[0], stop = X_lim[1] + resolution, step = resolution)
		y = np.arange(start = Y_lim[0], stop = Y_lim[1] + resolution, step = resolution)
		
		# probability matrix in log-odds scale:
		self.l = np.full(shape = (len(x), len(y)), fill_value = log_odds(p))

	def get_shape(self):
		"""
		Get dimensions
		"""
		return np.shape(self.l)

	def calc_MLE(self):
		"""
		Calculate Maximum Likelihood estimate of the map
		"""
		for x in range(self.l.shape[0]):

			for y in range(self.l.shape[1]):

				# cell is free
				if self.l[x][y] < log_odds(TRESHOLD_P_FREE):

					self.l[x][y] = log_odds(0.01)

				# cell is occupied
				elif self.l[x][y] > log_odds(TRESHOLD_P_OCC):

					self.l[x][y] = log_odds(0.99)

				# cell state uncertain
				else:

					self.l[x][y] = log_odds(0.5)

	def to_BGR_image(self):
		"""
		Transformation to BGR image format
		"""
		# grayscale image
		gray_image = self.to_grayscale_image()

		# repeat values of grayscale image among 3 axis to get BGR image
		rgb_image = np.repeat(a = gray_image[:,:,np.newaxis], 
							  repeats = 3,
							  axis = 2)

		return rgb_image

	def to_grayscale_image(self):
		"""
		Transformation to GRAYSCALE image format
		"""
		return 1 - retrieve_p(self.l)

	def discretize(self, x_cont, y_cont):
		"""
		Discretize continious x and y 
		并取整表示栅格
		"""
		x = int((x_cont - self.X_lim[0]) / self.resolution)
		y = int((y_cont - self.Y_lim[0]) / self.resolution)
		return (x,y)

	def update(self, x, y, p):
		"""
		Update x and y coordinates in discretized grid map
		"""
		# update probability matrix using inverse sensor model
		if self.check_pixel(x,y):
			self.l[x][y] += log_odds(p)

	def check_pixel(self, x, y):
		"""
		Check if pixel (x,y) is within the map bounds
		"""
		if x >= 0 and x < self.get_shape()[0] and y >= 0 and y < self.get_shape()[1]:
			
			return True 

		else:

			return False

	def find_neighbours(self, x, y):
		"""
		Find neighbouring pixels to pixel (x,y)
		"""
		X_neighbours = []
		Y_neighbours = []

		if self.check_pixel(x + 1, y):

			X_neighbours.append(x + 1)
			Y_neighbours.append(y)

		if self.check_pixel(x + 1, y + 1):

			X_neighbours.append(x + 1)
			Y_neighbours.append(y + 1)

		if self.check_pixel(x + 1, y - 1):

			X_neighbours.append(x + 1)
			Y_neighbours.append(y - 1)

		if self.check_pixel(x, y + 1):

			X_neighbours.append(x)
			Y_neighbours.append(y + 1)

		if self.check_pixel(x, y - 1):

			X_neighbours.append(x)
			Y_neighbours.append(y - 1)

		if self.check_pixel(x - 1, y):

			X_neighbours.append(x - 1)
			Y_neighbours.append(y)

		if self.check_pixel(x - 1, y + 1):

			X_neighbours.append(x - 1)
			Y_neighbours.append(y + 1)

		if self.check_pixel(x - 1, y - 1):

			X_neighbours.append(x - 1)
			Y_neighbours.append(y - 1)

		return zip(X_neighbours, Y_neighbours)


def set_pixel_color(bgr_image, x, y, color):
    """
    Set 'color' to the given pixel (x,y) on 'bgr_image'
    """

    if x < 0 or y < 0 or x >= bgr_image.shape[0] or y >= bgr_image.shape[1]:
        return 

    if color == 'BLUE':

        bgr_image[x, y, 0] = 1.0
        bgr_image[x, y, 1] = 0.0
        bgr_image[x, y, 2] = 0.0

    elif color == 'GREEN':

        bgr_image[x, y, 0] = 0.0
        bgr_image[x, y, 1] = 1.0
        bgr_image[x, y, 2] = 0.0

    elif color == 'RED':

        bgr_image[x, y, 0] = 0.0
        bgr_image[x, y, 1] = 0.0
        bgr_image[x, y, 2] = 1.0
