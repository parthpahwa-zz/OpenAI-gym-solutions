import os.path
import gym
from agent import Agent
import numpy as np
from keras.models import load_model
import ast

class Environment:
	def __init__(self, environment_name):
		self.env = gym.make(environment_name)
		self.q_val = []

	def run(self, agent):
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

	def perform_quicksave(self, itr, agent):
		agent.model.save("weights.h5")
		config_file = open('config.txt', 'w')
		config = {"itr":itr, "eps":agent.eps}
		config_file.write(str(config))
		config_file.close()
		agent.memory.save_memory()


space_invader = Environment('MsPacman-ram-v0')
agent = Agent(space_invader.env.observation_space.shape[0], space_invader.env.action_space.n, 64)

itr = 0
reward_list = []
try:
	if os.path.isfile("weights.h5"):
		agent.model = load_model("weights.h5")
		agent.memory.load_memory()
		print "Weights loaded"

	if os.path.isfile("config.txt"):
		config_file = open('config.txt', 'r')
		config = ast.literal_eval(config_file.read())
		itr = int(config["itr"])
		agent.eps = float(config["eps"])
		config_file.close()
		print "Config loaded"

	while itr < 150000:
		target = open('score.txt', 'a')
		reward, q_val = space_invader.run(agent)
		string = "Reward: " + str(reward) + " Q value: " + str(q_val) + " Iteration: " + str(itr) + "\n"
		target.write(string)
		reward_list.append(reward)
		if itr % 20 == 0:
			agent.decay()
		if itr % 100 == 0 and itr != 0:
			target.write(str(np.mean(reward_list[-100:]) + "\n"))
			space_invader.perform_quicksave(itr, agent)
		itr += 1
		target.close()
finally:
	target.close()
	space_invader.perform_quicksave(itr, agent)
