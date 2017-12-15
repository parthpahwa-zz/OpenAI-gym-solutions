import gym
import numpy as np
from frozenlakeNN import NeuralNet

def modify_reward(rwd, flag):
	if flag:
		if rwd < 1:
			return -0.95
	if rwd == 0:
		return -0.01
	return rwd

env = gym.make('FrozenLake-v0')
n_actions = env.action_space.n
n_states = env.observation_space.n
num_episodes = 2000
agent = NeuralNet(n_actions=n_actions, n_states=n_states, eps=0.1, discount=0.90, lr=0.2)
counter = 0
rList = []
for i in range(0, num_episodes):
	state = env.reset()
	state_vector = np.identity(16)[state:state+1]
	done = False
	rAll = 0
	j = 0

	while j < 99:
		j += 1
		action = agent.getGreedyAction(state_vector)
		if action[0] == -1:
			action[0] = env.action_space.sample()
		next_state, reward, done, _ = env.step(action[0])
		next_state_vector = np.identity(16)[next_state:next_state+1]
		new_reward = modify_reward(reward, done)

		agent.learn(state_vector, action, next_state_vector, new_reward)

		state = next_state
		state_vector = next_state_vector

		rAll += reward
		if done:
			if reward == 1:
				counter += 1
				print i, "      ", counter
			agent.decayEps(i)
			break
	rList.append(rAll)

print "Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%"
