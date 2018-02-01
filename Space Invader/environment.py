import gym
import numpy as np
import cv2

WIDTH = 84
HEIGHT = 84
N_FRAMES = 3

class Environment:
	def __init__(self, environment_name):
		self.env = gym.make(environment_name)
		self.q_val = []

	def train(self, agent):
		state = self.env.reset()
		total_reward = 0
		state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
		state = cv2.resize(state, (HEIGHT, WIDTH))
		current_frame_buffer = [state, state, state]
		next_frame_buffer = [state, state, state]

		frame_count = 1
		while True:
			action, self.q_val = agent.predict_single(np.array(current_frame_buffer).reshape(1, HEIGHT, WIDTH, N_FRAMES))
			next_state, reward, done, _ = self.env.step(action)

			if done: # terminal state
				next_state = None
			else:
				next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
				next_state = cv2.resize(next_state, (HEIGHT, WIDTH))

			next_frame_buffer.pop()
			next_frame_buffer.append(next_state)

			agent.save(current_frame_buffer, next_frame_buffer, action, reward)
			agent.replay()

			state = next_state
			current_frame_buffer.pop()
			current_frame_buffer.append(state)

			if reward == 0:
				reward = -0.05
			total_reward += reward
			frame_count += 1
			if done:
				break

		return total_reward, self.q_val, frame_count

	def test(self, agent):
		state = self.env.reset()
		total_reward = 0
		state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
		state = cv2.resize(state, (HEIGHT, WIDTH))
		current_frame_buffer = [state, state, state]
		next_frame_buffer = [state, state, state]
		while True:
			action, self.q_val = agent.predict_single(np.array(current_frame_buffer).reshape(1, HEIGHT, WIDTH, N_FRAMES), True)
			next_state, reward, done, _ = self.env.step(action)

			if done: # terminal state
				next_state = None
			else:
				next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
				next_state = cv2.resize(next_state, (HEIGHT, WIDTH))

			next_frame_buffer.pop()
			next_frame_buffer.append(next_state)

			state = next_state
			current_frame_buffer.pop()
			current_frame_buffer.append(state)

			total_reward += reward

			if done:
				break

		return total_reward, self.q_val
