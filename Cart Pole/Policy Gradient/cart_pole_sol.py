import gym
import tensorflow as tf
import numpy as np
from cart_pole_agent import Agent

env = gym.make('CartPole-v0')
n_actions = env.action_space.n

gamma = 0.95
num_episodes = 1400
max_eps = 500
update_freq = 8
agent = Agent(n_actions, 4, 16)


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
	gradBuffer = agent.get_trainable_var()

	for ix, grad in enumerate(gradBuffer):
		gradBuffer[ix] = grad * 0

	while i < num_episodes:
		current_state = env.reset()
		j = 0
		running_reward = 0
		ep_history = []
		while j < max_eps:
			action = agent.predict(current_state)
			next_state, reward, done, _ = env.step(action)
			running_reward += reward
			ep_history.append([current_state, action, reward, next_state])
			current_state = next_state

			if done:
				ep_history = np.array(ep_history)
				ep_history[:, 2] = discounted_reward(ep_history[:, 2])
				grads = np.array(agent.compute_gradient(ep_history))

				for ix, grad in enumerate(grads):
					gradBuffer[ix] += grad

				if i % update_freq == 0 and i != 0:
					feed_dict = dictionary = dict(zip(agent.grad_list, gradBuffer))
					agent.learn(feed_dict)
					for ix, grad in enumerate(grads):
						gradBuffer[ix] = grad * 0
				total_reward.append(running_reward)
				total_lenght.append(j)
				break
			j += 1
		if i % 100 == 0:
			print np.mean(total_reward[-100:])
		i += 1
