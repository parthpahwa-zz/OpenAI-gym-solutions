import gym
from agent import Agent

class Environment:
	def __init__(self, environment_name):
		self.env = gym.make(environment_name)

	def run(self, agent):
		state = self.env.reset()
		total_reward = 0

		while True:
			self.env.render()

			action = agent.predict_single(state)
			next_state, reward, done, _ = self.env.step(action)

			if done:
				next_state = [None for itr in range(0, self.env.observation_space.shape[0])]

			agent.save(state, next_state, action, reward)
			agent.replay()

			state = next_state
			total_reward += reward

			if done:
				break

		print("Total reward:", total_reward)


cart_pole = Environment('CartPole-v0')
print cart_pole.env.observation_space.shape[0], cart_pole.env.action_space.n
agent = Agent(cart_pole.env.observation_space.shape[0], cart_pole.env.action_space.n, 64)

while True:
	cart_pole.run(agent)
	agent.decay()
