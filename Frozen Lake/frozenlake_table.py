import gym
import numpy as np

env = gym.make("FrozenLake-v0")

Q = np.zeros([env.observation_space.n, env.action_space.n])

lr = .8
y = .95
num_episodes = 5
rList = []
for i in range(0, num_episodes):
	envState = env.reset()
	overFlag = False
	j = 0
	rAll = 0
	while j < 99:
		j += 1
		action = np.argmax(Q[envState, :] + np.random.rand(1, env.action_space.n)*(1./(1 + 0.5*i)))
		newState, reward, done, _ = env.step(action)
		Q[envState, action] = Q[envState, action]+lr*(reward+y*np.max(Q[newState, :])-Q[envState, action])
		envState = newState
		rAll += reward
		if done:
			break
	rList.append(rAll)
print Q
print "Score over time: " +  str(sum(rList)/num_episodes)
