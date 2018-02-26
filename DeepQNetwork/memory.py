import os
import random
import numpy as np

class Memory:

	def __init__(self, capacity):
		self.capacity = capacity
		self.transition_list = []
		self.savefile_name = 'memory.npy'


	def add(self, action_replay):
		self.transition_list.append(action_replay)
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
			np.save(self.savefile_name, np.array(random.sample(self.transition_list, 14000)))

	def load_memory(self):
		if os.path.isfile(self.savefile_name):
			self.transition_list = list(np.load(self.savefile_name))
			print "Memory Loaded"
