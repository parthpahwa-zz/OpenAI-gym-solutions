import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from memory import Memory

class Agent:
	def __init__(self, input_dim, n_actions, lr=0.00025, eps=1.0, memory=150000):
		self.n_states = input_dim
		self.n_actions = n_actions
		self.eps = eps
		self.BATCH_SIZE = 32
		self.GAMMA = 0.99
		self.MIN_EPS = 0.1
		self.model = self.build_model(input_dim, n_actions, lr)
		self.memory = Memory(memory)
		self.zeros = np.zeros(self.n_states)


	def build_model(self, input_dim, n_actions, lr):
		model = Sequential()
		model.add(Dense(input_dim=input_dim, units=256, activation='relu'))
		model.add(Dense(input_dim=input_dim, units=1024, activation='relu'))
		model.add(Dropout(0.3))
		model.add(Dense(input_dim=input_dim, units=2048, activation='relu'))
		model.add(Dropout(0.4))
		model.add(Dense(input_dim=input_dim, units=48, activation='relu'))
		model.add(Dense(units=n_actions, activation='linear'))
		optimizer = RMSprop(lr=lr)
		model.compile(loss='mse', optimizer=optimizer)

		return model


	def train(self, x, y, verbose=0):
		self.model.fit(x, y, verbose=verbose, batch_size=64)


	def predict(self, state):
		return self.model.predict(state)


	def predict_single(self, state):
		q_val = self.predict(state.reshape(1, self.n_states))
		if random.random() > self.eps:
			return [np.argmax(q_val.flatten()), q_val]
		else:
			return [random.randint(0, self.n_actions-1), q_val]


	def save(self, state, next_state, action, reward):
		self.memory.add(np.array(state, dtype=np.uint8), np.array(next_state), int(action), int(reward))


	def replay(self):
		batch = self.memory.sample(self.BATCH_SIZE)

		x = np.empty(0).reshape(0, self.n_states)
		y = np.empty(0).reshape(0, self.n_actions)

		none_list = [None for itr in range(0, self.n_states)]

		next_state = np.array([(self.zeros if np.array_equal(state, none_list) else state) for state in batch[:, self.n_states:2*self.n_states]])
		state = batch[:, :self.n_states]

		target_Q = self.predict(next_state)
		Q_value = self.predict(state)

		for indx, element in enumerate(batch):
			state = element[:self.n_states]
			next_state = element[self.n_states:2*self.n_states]
			action = int(element[2*self.n_states])
			reward = element[2*self.n_states + 1]
			q_val = Q_value[indx]

			if np.array_equal(next_state, none_list):
				q_val[action] = reward
			else:
				q_val[action] = reward + self.GAMMA * np.amax(np.array(target_Q[indx]))

			y = np.vstack([y, q_val])
			x = np.vstack([x, state])
		self.train(x, y)


	def decay(self):
		if self.eps > self.MIN_EPS:
			self.eps = self.eps*0.99
