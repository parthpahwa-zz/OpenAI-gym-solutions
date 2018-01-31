#!/usr/bin/env python
# pylint: disable=C0103
# pylint: disable=W0312
# pylint: disable=C0111
# pylint: disable=C0301
# pylint: disable=E0211
# pylint: disable=E0602
# pylint: disable=E1121


import gym
import numpy as np

class Environment:
	def __init__(self, environment_name):
		self.env = gym.make(environment_name)
		self.q_val = []

	def train(self, agent):
		state = self.env.reset()
		total_reward = 0

		while True:

			action, self.q_val = agent.predict_single(state)
			next_state, reward, done, _ = self.env.step(action)

			if done: # terminal state
				next_state = [None for itr in range(0, self.env.observation_space.shape[0])]

			agent.save(state, next_state, action, reward)
			agent.replay()

			state = next_state
			total_reward += reward

			if done:
				break

		return total_reward, self.q_val

	def test(self, agent):
		state = self.env.reset()
		total_reward = 0

		while True:
			action, self.q_val = agent.predict_single(state)
			next_state, reward, done, _ = self.env.step(action)

			state = next_state
			total_reward += reward

			if done:
				break

		return total_reward, self.q_val
