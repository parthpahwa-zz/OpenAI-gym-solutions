import os
import random
import numpy as np

class Memory:

	def __init__(self, capacity):
		self.load = False
		self.capacity = capacity
		self.transition_list = []
		self.under_sampled_rewards = []
		self.under_sampled_zeros = []
		self.length = [1.0, 1.0]
		self.savefile_name = 'memory.npy'


	def add(self, action_replay, perform_under_sampling=False):
		if perform_under_sampling:
			if action_replay[len(action_replay) - 1] :
				if self.length[0]/self.length[1] <= 1.2:
					self.under_sampled_rewards.append(action_replay)
					self.length[0] += 1
				else:
					self.under_sampled_rewards.append(action_replay)
					self.under_sampled_rewards.pop()

			elif action_replay[len(action_replay) - 1] == 0:
				if self.length[1]/self.length[0] <= 1.2:
					self.under_sampled_zeros.append(action_replay)
					self.length[1] += 1
				else:
					self.under_sampled_zeros.append(action_replay)
					self.under_sampled_zeros.pop()
			
			if sum(self.length) > self.capacity + 2:
				self.under_sampled_rewards.pop()
				self.length[0] -= 1

				self.under_sampled_zeros.pop()
				self.length[1] -= 1
			return

		self.transition_list.append(action_replay)
		if len(self.transition_list) > self.capacity:
			self.transition_list.pop()


	def sample(self, n, perform_under_sampling=False):
		if perform_under_sampling:
			if self.load:
				self.load = False
				for val in self.transition_list:
					if val[len(val)-1]:
						self.under_sampled_rewards.append(val)
						self.length[0] += 1
					else:
						self.under_sampled_zeros.append(val)
						self.length[1] += 1
			
			self.transition_list = []
			if n + 2 > sum(self.length):
				tmpList = []
				tmpList.extend(self.under_sampled_rewards)
				tmpList.extend(self.under_sampled_zeros)
				random.shuffle(tmpList)
				return np.array(tmpList)

			self.transition_list.extend(random.sample(self.under_sampled_rewards, min(n/2, int(self.length[0])-1)))
			self.transition_list.extend(random.sample(self.under_sampled_zeros, min(n/2, int(self.length[1])-1)))

			random.shuffle(self.transition_list)
			return np.array(self.transition_list)

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
			self.load = True
