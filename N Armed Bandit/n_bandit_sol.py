import tensorflow as tf
import numpy as np
from n_bandit_agent import Agent
from n_bandit import Bandit

bandit = Bandit()
agent = Agent(n_actions=bandit.n_actions, n_states=bandit.n_bandits)
num_iterations = 10000

total_reward = np.zeros([bandit.n_bandits, bandit.n_actions])
i = 0

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	while i < num_iterations:
		state = bandit.select_bandit()
		action = agent.getGreedyAction(state)
		reward = bandit.pull_arm(action)
		_, ww = agent.learn(state, action, reward)
		total_reward[state, action] += reward
		if i % 500 == 0:
			print "Mean reward for each of the " + str(bandit.n_bandits) + " bandits: " + str(np.mean(total_reward, axis=1))
			print ww
		i += 1
	print "Mean reward for each of the " + str(bandit.n_bandits) + " bandits: " + str(np.mean(total_reward, axis=1))
