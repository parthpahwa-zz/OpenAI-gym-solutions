import gym
import tensorflow as tf
import numpy as np

env = gym.make('CartPole-v0')
n_actions = env.action_space.n
# n_states = env.observation_space.n
num_episodes = 2000
max_eps = 500

def discounted_reward(r):
	d_reward = np.zeros_like(r)
	r_reward = 0
	for t in reversed(xrange(0, r.size)):
		r_reward = r_reward * gamma + r[t]
		d_reward[t] = r_reward
	return d_reward

with tf.Session() as sess:
	i = 0
	total_reward = []
	total_lenght = []
	while i < num_episodes:
		running_reward = 0
		current_state = env.reset()
		j = 0
		ep_history = []
		while j < max_eps:
			action = np.random.randint(0, n_actions)
			next_state, reward, done, _ = env.step(action)
			running_reward += reward
			ep_history.append([current_state, action, reward, next_state])
			current_state = next_state

			if done:
				ep_history = np.array(ep_history)
				ep_history[:, 2] = discounted_reward(ep_history[:, 2])
				break
			j += 1

		print "Reward: " + str(running_reward) + "Steps: " + str(j)
		total_reward.append(running_reward)
		total_lenght.append(j)
		i += 1
