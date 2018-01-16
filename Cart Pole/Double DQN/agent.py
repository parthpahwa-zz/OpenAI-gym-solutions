import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop


class Agent:
	def __init__(self, input_dim, n_actions, n_hidden, lr=0.0005, eps=1.0, memory=100000):
		self.n_states = input_dim
		self.n_actions = n_actions
		self.eps = eps
		self.BATCH_SIZE = 64
		self.GAMMA = 0.99
		self.UPDATE_FREQUENCY = 10000
		self.timestep = 0
		self.model = self.build_model(input_dim, n_actions, n_hidden, lr)
		self.target_model = self.build_model(input_dim, n_actions, n_hidden, lr)
		self.memory = Memory(memory, input_dim)


	def build_model(self, input_dim, n_actions, n_hidden, lr):
		model = Sequential()
		model.add(Dense(input_dim=input_dim, units=n_hidden, activation='relu'))
		model.add(Dense(units=n_actions, activation='linear'))
		optimizer = RMSprop(lr=lr)
		model.compile(loss='mse', optimizer=optimizer)

		return model


	def train(self, x, y, verbose=0):
		self.model.fit(x, y, verbose=verbose, batch_size=64)


	def predict(self, state, target=False):
		if target:
			return self.target_model.predict(state)
		return self.model.predict(state)


	def predict_single(self, state):
		q_val = self.predict(state.reshape(1, self.n_states))
		if random.random() > self.eps:
			return [np.argmax(q_val.flatten()), q_val]
		else:
			return [random.randint(0, self.n_actions-1), q_val]


	def save(self, state, next_state, action, reward):
		if self.timestep != 0 and self.timestep % self.UPDATE_FREQUENCY == 0:
			self.update_target()
		self.memory.add(np.array(state), np.array(next_state), np.array(action), np.array(reward))
		self.timestep += 1


	def replay(self):
		batch = self.memory.sample(self.BATCH_SIZE)

		x = np.empty(0).reshape(0, self.n_states)
		y = np.empty(0).reshape(0, self.n_actions)
		zeros = np.zeros(4)

		none_list = [None for itr in range(0, self.n_states)]
		next_state = np.array([(zeros if np.array_equal(state, none_list) else state) for state in batch[:, 4:8]])
		state = batch[:, :4]
		target_Q = self.predict(next_state)
		Q_value = self.predict(state)
		DDQN_Q_value = self.predict(state, target=True)

		for indx, element in enumerate(batch):
			state = element[:4]
			next_state = element[4:8]
			action = int(element[8])
			reward = element[9]
			q_val = Q_value[indx]

			if np.array_equal(next_state, none_list):
				q_val[action] = reward
			else:
				q_val[action] = reward + self.GAMMA * DDQN_Q_value[indx][np.argmax(np.array(target_Q[indx]))]

			y = np.vstack([y, q_val])
			x = np.vstack([x, state])
		self.train(x, y)


	def update_target(self):
		self.target_model.set_weights(self.model.get_weights())


	def decay(self):
		self.eps = self.eps*0.99

class Memory:

	def __init__(self, capacity, n_state):
		self.capacity = capacity
		self.transition_list = np.empty(0).reshape(0, n_state * 2 + 1 + 1)

	def add(self, state, next_state, action, reward):
		temp = np.hstack([state, next_state, action, reward])
		self.transition_list = np.vstack([self.transition_list, temp])
		if len(self.transition_list) > self.capacity:
			self.transition_list = np.delete(self.transition_list, 0, 0)

	def sample(self, n):
		n = min(n, len(self.transition_list))
		return np.array(random.sample(self.transition_list, n))
