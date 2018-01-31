#!/usr/bin/env python
# pylint: disable=C0103
# pylint: disable=W0312
# pylint: disable=C0111
# pylint: disable=C0301
# pylint: disable=E0211
# pylint: disable=E0602
# pylint: disable=E1121

import os
import random
import numpy as np

class Memory:

	def __init__(self, capacity):
		self.capacity = capacity
		self.transition_list = []
		self.savefile_name = 'memory.npy'


	def add(self, state, next_state, action, reward):
		temp = np.hstack([state, next_state, action, reward])
		self.transition_list.append(temp)
		if len(self.transition_list) > self.capacity:
			self.transition_list.pop()


	def sample(self, n):
		if n > len(self.transition_list):
			return np.array(self.transition_list)
		return np.array(random.sample(self.transition_list, n))


	def save_memory(self):
		if len(self.transition_list) < 14000:
			np.save(self.savefile_name, self.transition_list)
		else:
			np.save(self.savefile_name, np.array(random.sample(self.transition_list, n)))

	def load_memory(self):
		if os.path.isfile(self.savefile_name):
			self.transition_list = list(np.load(self.savefile_name))
			print "Memory Loaded"
