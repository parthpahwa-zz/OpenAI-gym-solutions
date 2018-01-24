import os.path
import gym
from agent import Agent
import numpy as np
import cv2
from keras.models import load_model

WIDTH = 80
HEIGHT = 80

class Environment:
	def __init__(self, environment_name):
		self.env = gym.make(environment_name)
		self.q_val = []

	def run(self, agent):
		state = self.env.reset()
		total_reward = 0
		state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
		state = cv2.resize(state, (HEIGHT, WIDTH))
		while True:

			action, self.q_val = agent.predict_single(np.array(state).reshape(1, HEIGHT, WIDTH, 1))
			next_state, reward, done, _ = self.env.step(action)

			if done: # terminal state
				next_state = None
			else:
				next_state = cv2.cvtColor(next_state, cv2.COLOR_BGR2GRAY)
				next_state = cv2.resize(next_state, (80, 80))

			agent.save(state, next_state, action, reward)
			agent.replay()

			state = next_state
			total_reward += reward

			if done:
				break

		return total_reward, self.q_val


space_invader = Environment('SpaceInvaders-v0')
agent = Agent(HEIGHT, WIDTH, space_invader.env.action_space.n)

itr = 0
reward_list = []
try:
	if os.path.isfile("weights.h5"):
		agent.model = load_model("weights.h5")
		print "Weights loaded"

	while True:
		reward, q_val = space_invader.run(agent)
		print "Reward:", reward, "Q value:", q_val, "Iteration:", itr
		agent.decay()
		reward_list.append(reward)
		if itr % 100 == 0 and itr != 0:
			print np.mean(reward_list[-100:])
			agent.model.save("weights.h5")
		itr += 1
finally:
	agent.model.save("weights.h5")
