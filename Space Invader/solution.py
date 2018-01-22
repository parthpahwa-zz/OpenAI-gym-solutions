import gym
from agent import Agent
import numpy as np
import cv2
from keras.models import load_model

class Environment:
	def __init__(self, environment_name):
		self.env = gym.make(environment_name)
		self.q_val = []

	def run(self, agent):
		state = self.env.reset()
		total_reward = 0
		state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
		state = cv2.resize(state, (80, 80))
		while True:
			action, self.q_val = agent.predict_single(np.array(state).reshape(1, 80, 80, 1))
			next_state, reward, done, _ = self.env.step(action)
      
			if done:
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
# print cart_pole.env.observation_space.shape[0], cart_pole.env.action_space.n
agent = Agent(space_invader.env.observation_space.shape[0], space_invader.env.action_space.n, 64)

itr = 0
reward_list = []
try:
	agent.model = load_model("my_model_itr_0.h5")
	while True:
		reward, q_val = space_invader.run(agent)
		print "Reward:", reward, "Q value:", q_val, "Iteration:", itr
		reward_list.append(reward)
		if itr % 50 == 0 and itr != 0:
			print np.mean(reward_list[-50:])
			model_name = "my_model_itr_" + str(itr) + ".h5"
			agent.model.save(model_name)
		itr += 1
finally:
	model_name = "my_model_itr_" + str(itr) + ".h5"
	agent.model.save(model_name)
