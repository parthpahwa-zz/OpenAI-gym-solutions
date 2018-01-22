import random
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import RMSprop


class Agent:
	def __init__(self, input_dim, n_actions, n_hidden, lr=0.0005, eps=1.0, memory=60000):
		self.n_states = input_dim
		self.n_actions = n_actions
		self.counter = 1
		self.eps = eps
		self.BATCH_SIZE = 64
		self.GAMMA = 0.99
		self.MIN_EPS = 0.1
		self.model = self.build_model(input_dim, n_actions, n_hidden, lr)
		self.memory = Memory(memory, input_dim)


	def build_model(self, input_dim, n_actions, n_hidden, lr):

		model = Sequential()
		model.add(Conv2D(32, kernel_size=11, activation='relu', input_shape=[80, 80, 1]))
		model.add(Conv2D(24, 5, activation='relu'))
		model.add(MaxPooling2D(pool_size=3, strides=2))
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dense(32, activation='relu'))
		model.add(Dense(n_actions, activation='linear'))
		optimizer = RMSprop(lr=lr)
		model.compile(loss='mse', optimizer=optimizer)
		return model


	def train(self, x, y):
		self.model.fit(x, y, batch_size=self.BATCH_SIZE, verbose=0)


	def predict(self, state):
		return self.model.predict(state)


	def predict_single(self, state):
		q_val = self.predict(state)
		if random.random() > self.eps:
			return [np.argmax(q_val.flatten()), q_val]
		else:
			return [random.randint(0, self.n_actions-1), q_val]


	def save(self, state, next_state, action, reward):
		if self.counter % 101 == 0:
			self.decay()
			self.counter = 1
		else:
			self.counter += 1
		self.memory.add((np.array(state, dtype=float), np.array(next_state), np.array(action), np.array(reward)))


	def replay(self):
		batch = self.memory.sample(self.BATCH_SIZE)

		x = []
		y = np.empty(0).reshape(0, self.n_actions)
		zeros = np.zeros(80 * 80).reshape(80, 80)
		next_state = []
		for state in batch[:, 1]:
			if state.shape == (80, 80):
				next_state.append(state)
			else:
				next_state.append(zeros)

		next_state = np.vstack(next_state).reshape(-1, 80, 80, 1)
		state = np.vstack(batch[:, 0]).reshape(-1, 80, 80, 1)

		target_Q = self.predict(next_state)
		Q_value = self.predict(state)

		for indx, element in enumerate(batch):
			state = element[0]
			next_state = element[1]
			action = int(element[2])
			reward = element[3]
			q_val = Q_value[indx]

			if next_state is None:
				q_val[action] = reward
			else:
				q_val[action] = reward + self.GAMMA * np.amax(np.array(target_Q[indx]))
				
			y = np.vstack([y, q_val])
			x.append(state)
		x = np.array(x).reshape(-1, 80, 80, 1)
		self.train(x, y)


	def decay(self):
		if self.eps > self.MIN_EPS:
			self.eps = self.eps*0.99

class Memory:

	def __init__(self, capacity, n_state):
		self.capacity = capacity
		self.transition_list = np.empty(0).reshape(0, 4)

	def add(self, action_replay):
		
		self.transition_list = np.vstack([self.transition_list, action_replay])
		if len(self.transition_list) > self.capacity:
			self.transition_list = np.delete(self.transition_list, 0, 0)

	def sample(self, n):
		n = min(n, len(self.transition_list))
		return np.array(random.sample(self.transition_list, n))
