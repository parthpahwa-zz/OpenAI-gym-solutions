import gym
from agent import Agent
import numpy as np
class Environment:
	def __init__(self, environment_name):
		self.env = gym.make(environment_name)
		self.q_val = []

	def run(self, agent):
		state = self.env.reset()
		total_reward = 0

		while True:
			# self.env.render()

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


cart_pole = Environment('CartPole-v1')
agent = Agent(cart_pole.env.observation_space.shape[0], cart_pole.env.action_space.n, 64)

itr = 0
reward_list = []
varbose = 0
while True:
	reward, q_val = cart_pole.run(agent)
	if verbose:
		print "Reward:", reward, "Q value:", q_val, "Iteration:", itr
	agent.decay()
	reward_list.append(reward)
	if itr % 100 == 0 and itr != 0:
		print np.mean(reward_list[-100:])
	itr += 1
