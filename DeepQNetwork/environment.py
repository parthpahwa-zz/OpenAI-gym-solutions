import gym
import numpy as np
from .utils import preprocess

class Environment:
	def __init__(self, environment_name, WIDTH, HEIGHT, N_FRAMES):
		self.env = gym.make(environment_name)
		self.WIDTH = WIDTH
		self.HEIGHT = HEIGHT
		self.N_FRAMES = N_FRAMES

	def train(self, agent, undersampled=False):
		q_val_list = []
		state = self.env.reset()
		total_reward = 0

		state = preprocess(state, self.HEIGHT, self.WIDTH)
		current_frame_buffer = [state, state, state, state]
		next_frame_buffer = [state, state, state, state]

		while True:
			clipped_reward = 0
			action, q_val = agent.predict_single(np.array(current_frame_buffer).reshape(1, self.HEIGHT, self.WIDTH, self.N_FRAMES))
			next_state, reward, done, _ = self.env.step(action)

			if done: # terminal state
				next_state = None
			else:
				next_state = preprocess(next_state, self.HEIGHT, self.WIDTH)

			next_frame_buffer.pop()
			next_frame_buffer.append(next_state)

			q_val_list.append(np.mean(q_val))
			if reward > 0:
				clipped_reward = 1
			elif reward < 0:
				clipped_reward = -1

			agent.save(current_frame_buffer, next_frame_buffer, action, clipped_reward, undersampled)
			agent.replay(undersampled)

			state = next_state
			current_frame_buffer.pop()
			current_frame_buffer.append(state)

			total_reward += reward
			agent.counter += 1
			if done:
				break

		return total_reward, q_val_list, q_val

	def test(self, agent):
		state = self.env.reset()
		total_reward = 0

		state = preprocess(state, self.HEIGHT, self.WIDTH)
		current_frame_buffer = [state, state, state, state]
		next_frame_buffer = [state, state, state, state]
		while True:
			action, self.q_val = agent.predict_single(np.array(current_frame_buffer).reshape(1, self.HEIGHT, self.WIDTH, self.N_FRAMES), True)
			next_state, reward, done, _ = self.env.step(action)

			self.env.render()
			if done: # terminal state
				next_state = None
			else:
				next_state = preprocess(next_state, self.HEIGHT, self.WIDTH)

			next_frame_buffer.pop()
			next_frame_buffer.append(next_state)

			state = next_state
			current_frame_buffer.pop()
			current_frame_buffer.append(state)

			total_reward += reward

			if done:
				break

		return total_reward, self.q_val
