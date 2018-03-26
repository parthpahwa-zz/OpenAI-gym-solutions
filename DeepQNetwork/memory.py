import os
import random
import numpy as np

class Memory:

	def __init__(self, capacity):
		self.load = False
		self.capacity = capacity
		self.transition_list = []
		self.length = [0.0, 0.0, 0.0]
		self.savefile_name = 'memory.npy'
		self.reward_memory = [[], [], []]

	def add(self, action_replay, perform_under_sampling=False):
		if perform_under_sampling:
			if action_replay[len(action_replay) - 1] > 0:
				if self.length[0]/self.capacity <= 0.334:
					self.reward_memory[0].append(action_replay)
					self.length[0] += 1
				else:
					self.reward_memory[0].append(action_replay)
					self.reward_memory[0].pop()

			elif action_replay[len(action_replay) - 1] == 0:
				if self.length[1]/self.capacity <= 0.334:
					self.reward_memory[1].append(action_replay)
					self.length[1] += 1
				else:
					self.reward_memory[1].append(action_replay)
					self.reward_memory[1].pop()

			else:
				if self.length[2]/self.capacity <= 0.334:
					self.reward_memory[2].append(action_replay)
					self.length[2] += 1
				else:
					self.reward_memory[2].append(action_replay)
					self.reward_memory[2].pop()
			print self.length, sum(self.length)
			return

		self.transition_list.append(action_replay)
		if len(self.transition_list) > self.capacity:
			self.transition_list.pop()


	def sample(self, n, perform_under_sampling=False):
		if perform_under_sampling:
			if self.load:
				self.load = False
				for val in self.transition_list:
					if val[len(val)-1] > 0:
						self.reward_memory[0].append(val)
						self.length[0] += 1
					elif val[len(val)-1] == 0:
						self.reward_memory[1].append(val)
						self.length[1] += 1
					else:
						self.reward_memory[2].append(val)
						self.length[2] += 1

			self.transition_list = []

			if n > sum(self.length):
				tmpList = []
				for i in range(0, len(self.length)):
					tmpList.extend(self.reward_memory[i])
				random.shuffle(tmpList)
				return np.array(tmpList)

			remainder = n
			counter = 0
			flag = 1
			while remainder > 0:
				for i in range(0, len(self.length)):
					if self.length[i] < int(n/len(self.length)):
						if flag:
							self.transition_list.extend(random.sample(self.reward_memory[i], int(self.length[i])))
							remainder -= int(self.length[i])
					else:
						if remainder >= int(n/len(self.length)):
							self.transition_list.extend(random.sample(self.reward_memory[i], int(n/len(self.length))))
							remainder -= int(n/len(self.length))
							counter += 1
						elif remainder > 0:
							if int(remainder/counter) == 0:
								self.transition_list.extend(random.sample(self.reward_memory[i], 1))
								remainder -= 1
							else:
								self.transition_list.extend(random.sample(self.reward_memory[i], int(remainder/counter)))
								remainder -= int(remainder/counter)
				flag = 0

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
