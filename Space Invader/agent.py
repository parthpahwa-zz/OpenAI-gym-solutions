import random
import numpy as np
from keras.models import Sequential
from keras.layers.core import Activation
from keras.layers import Dense, Flatten, Convolution2D
from keras.optimizers import RMSprop
from memory import Memory


class Agent:

	def __init__(self, height, width, n_actions, nframes=3, lr=0.0005, eps=1.0, memory=150000):
		self.n_actions = n_actions
		self.height = height
		self.width = width
		self.n_frames = nframes
		self.counter = 1
		self.eps = eps
		self.BATCH_SIZE = 32
		self.GAMMA = 0.99
		self.MIN_EPS = 0.1
		self.model = self.build_model(lr)
		self.memory = Memory(memory)
		self.zeros = np.zeros(self.height * self.width).reshape(self.height, self.width)


	def build_model(self, lr):

		model = Sequential()
		model.add(Convolution2D(32, 8, 8, subsample=(4, 4), input_shape=(self.height, self.width, self.n_frames)))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 4, 4, subsample=(2, 2)))
		model.add(Activation('relu'))
		model.add(Convolution2D(64, 3, 3))
		model.add(Activation('relu'))
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation('relu'))
		model.add(Dense(self.n_actions))
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
		self.memory.add((np.array(state, dtype=np.uint8), np.array(next_state), int(action), int(reward)))


	def replay(self):
		batch = self.memory.sample(self.BATCH_SIZE)

		x = []
		y = np.empty(0).reshape(0, self.n_actions)
		next_state = []

		for state in batch[:, 1]:
			if state[self.n_frames - 1] is not None:
				next_state.append(state)
			else:
				temp = []
				for i in range(0, self.n_frames-1):
					temp.append(state[i])
				temp.append(self.zeros)
				next_state.append(temp)

		next_state = np.vstack(next_state).reshape(-1, self.height, self.width, self.n_frames)
		state = np.vstack(batch[:, 0]).reshape(-1, self.height, self.width, self.n_frames)

		target_Q = self.predict(next_state)
		Q_value = self.predict(state)

		for indx, element in enumerate(batch):
			state = element[0]
			next_state = element[1]
			action = int(element[2])
			reward = element[3]
			q_val = Q_value[indx]

			if next_state[self.n_frames - 1] is None:
				q_val[action] = reward
			else:
				q_val[action] = reward + self.GAMMA * np.amax(np.array(target_Q[indx]))

			y = np.vstack([y, q_val])
			x.append(state)
		x = np.array(x).reshape(-1, self.height, self.width, self.n_frames)
		self.train(x, y)


	def decay(self):
		if self.eps > self.MIN_EPS:
			self.eps = self.eps*0.99
