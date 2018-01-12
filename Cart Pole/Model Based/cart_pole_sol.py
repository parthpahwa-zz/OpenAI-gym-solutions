import gym
import tensorflow as tf
import numpy as np
from cart_pole_agent import Agent
from cart_pole_model import Model


def discount(r):
	discounted = np.array([val * (gamma ** itr) for itr, val in enumerate(r)])
	discounted -= np.mean(discounted)
	return discounted

def step_model(x_list, action_l):
	x = x_list[-1].reshape(1, -1)
	x = np.hstack([x, [[action_l]]])

	output = model.predict(x)[0]
	output_next_state = output[:, :4]
	output_reward = output[:, 4]
	output_done = output[:, 5]

	output_next_state[:, 0] = np.clip(output_next_state[:, 0], -2.4, 2.4)

	output_next_state[:, 2] = np.clip(output_next_state[:, 2], -0.4, 0.4)
	output_done = True if output_done > 0.01 or len(x_list) > 500 else False

	return output_next_state, output_reward, output_done


env = gym.make('CartPole-v0')
n_actions = env.action_space.n
gamma = 0.99
num_episodes = 5000
max_eps = 500

agent = Agent(n_actions, 4, 10)
model = Model(4, 256)

train_from_model = False
train_first_steps = 500
state_list = np.empty(0).reshape(0, 4)
rewards = np.empty(0).reshape(0, 1)
action_list = np.empty(0).reshape(0, 1)
i = 0

model_batch_size = 4
policy_batch_size = 4

origianl_reward = []
real_rewards = []
with tf.Session() as sess:
	grads = np.array([np.zeros(var.get_shape().as_list()) for var in agent.trainable_vars])
	current_state = env.reset()
	while i < num_episodes:

		current_state = current_state.reshape(1, -1)
		action = agent.predict(current_state)

		state_list = np.vstack([current_state, state_list])
		action_list = np.vstack([action, action_list])

		if train_from_model:
			current_state, reward, done = step_model(state_list, action)
		else:
			current_state, reward, done, _ = env.step(action)

		rewards = np.vstack([reward, rewards])
		if done or len(state_list) > 300:
			done_list = np.zeros(shape=(len(state_list), 1))
			if done:
				done_list[len(state_list) - 1] = 1

			if not train_from_model:
				prev_state_l = np.hstack([state_list[:-1, :], action_list[:-1, :]])
				next_state_l = state_list[1:, :]
				next_reward = rewards[1:, :]
				next_done = done_list[1:, :]
				loss, _ = model.learn(prev_state_l, next_state_l, next_reward, next_done)

			real_rewards.append(sum(rewards))
			disc_reward = discount(rewards)
			grads += agent.compute_gradient(state_list, action_list, disc_reward)
			i += 1
			current_state = env.reset()
			state_list = np.empty(0).reshape(0, 4)
			rewards = np.empty(0).reshape(0, 1)
			action_list = np.empty(0).reshape(0, 1)

			if i > train_first_steps:
				train_from_model = not train_from_model

			if i % policy_batch_size == 0 and i != 0:
				agent.learn(grads)
				grads = np.array([np.zeros(var.get_shape().as_list()) for var in agent.trainable_vars])

				if i % 500 == 0:
					print "Game:", str(i), "Average of 100 plays:", np.mean(real_rewards[-100: ])
