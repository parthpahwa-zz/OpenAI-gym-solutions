import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


class Agent:
	BATCH_SIZE = 64
	GAMMA = 0.99
	def __init__(self, n_actions, input_dim, n_hidden, lr=0.001, eps=0.1, memory=100000):
		self.n_states = input_dim
		self.n_actions = n_actions
		self.eps = eps
		self.model = self.build_model(input_dim, n_actions, n_hidden, lr)
		self.memory = Memory(memory, input_dim)


	def build_model(self, input_dim, n_actions, n_hidden, lr):
		model = Sequential()
		model.add(Dense(input_dim=input_dim, units=n_hidden, activation='relu'))
		model.add(Dense(units=n_actions, activation='linear'))
		optimizer = RMSprop(lr=lr)
		model.compile(loss='logcosh', optimizer=optimizer)

		return model


	def train(self, x, y, verbose=0):
		self.model.fit(x, y, verbose=verbose)


	def predict(self, state):
		return self.model.predict(state)


	def predict_single(self, state):
		if random.random() > self.eps:
			return np.argmax(self.predict(state.reshape(1, self.n_states)).flatten())
		else:
			return random.randint(0, self.n_actions-1)


	def replay(self):
		batch = self.memory.sample(BATCH_SIZE)

		x = np.empty(0).reshape(0, self.n_states)
		y = np.empty(0).reshape(0, self.n_actions)

		zeros = np.zeros(4)
		none_list = [None for itr in range(0, self.n_states)]
		next_state = np.array([(zeros if np.array_equal(state, none_list) else state) for state in batch[:, 4:8]])
		target_Q = self.predict(next_state)

		for indx, element in enumerate(batch):
			state = element[:4]
			next_state = element[4:8]
			reward = element[8]

			if np.array_equal(next_state, none_list):
				y = np.vstack([y, reward])
			else:
				y = np.vstack([y, reward + GAMMA * np.amax(target_Q[indx])])
			x = np.vstack([x, state])
		self.train(x, y, verbose=1)


class Memory:

	def __init__(self, capacity, n_state):
		self.capacity = capacity
		self.transition_list = np.empty(0).reshape(0, n_state * 2 + 1)

	def add(self, state, reward, next_state):
		temp = np.hstack([state, reward, next_state])
		self.transition_list = np.vstack([self.transition_list, temp])
		if len(self.transition_list) > self.capacity:
			self.transition_list = np.delete(self.transition_list, 0, 0)

	def sample(self, n):
		n = min(n, len(self.transition_list))
		return np.array(random.sample(self.transition_list, n))
