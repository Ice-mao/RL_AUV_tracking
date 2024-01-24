#!/usr/bin/env python

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

class MsgStack:
	"""
	Stack of messages
	"""
	def __init__(self):
		self.messages = []
 
	def push(self, msg):
		"""
		Push method
		"""
		self.messages.append(msg)

	def top(self):
		"""
		Top method
		"""
		return self.messages[-1]

	def pop(self):
		"""
		Pop method
		"""
		return self.messages.pop()

	def empty(self):
		"""
		Is empty method
		"""
		return len(self.messages) == 0


class MsgHandler:
	"""
	Message handler
	"""
	def __init__(self):
		self.scanMsgStack = MsgStack()
		self.odomMsgStack = MsgStack()

	def accept_message(self, topic, msg):
		"""
		Accepting message method
		"""
		if topic == '/scan':
			self.scanMsgStack.push(msg)
		elif topic == '/odom':
			self.odomMsgStack.push(msg)

	def pair_check(self):
		"""
		Checking if /odom and /scan messages are paired
		"""
		if self.scanMsgStack.empty() or self.odomMsgStack.empty():
			return False
		else:
			return True

	def process_paired_messages(self):
		"""
		Processing paired messages
		"""
		msgOdom = self.odomMsgStack.pop()
		msgScan = self.scanMsgStack.pop()
		return (msgOdom, msgScan)