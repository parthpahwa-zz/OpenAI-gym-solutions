import tensorflow as tf
import numpy as np

"""
Class: NeuralNet

Member Variables:
	1) n_actions : (int) number of available actions for the agent
	2) n_states : (int) number of states in the environment
	3) discount : (float) discount value for the bellman equation
	4) epsilon : (float) greedy epsilon value to choose random action
	5) lr : (float) learning rate
	6) input_tensor : (tf tensor) input to Neural network of shape (1, n_states)
	7) Qvalue : (tf tensor) output tensor of shape (1, n_actions)
	8) weights1 : (tf tensor) weight matrix

Member Functions:
	1) __init__ : Constructor
	2) build_model : initialzes the neural network
	3) predict : predicts the Q values for the input state
	4) getGreedyAction : returns the action using epsilon greedy method
	5) learn : Trains the neural network
"""
class NeuralNet:

	def __init__(self, n_actions, n_states, eps=0.1, discount=0.99, lr=0.1):
		self.n_actions = n_actions
		self.n_states = n_states
		self.discount = discount
		self.epsilon = eps
		self.lr = lr
		tf.reset_default_graph()
		self.input_tensor, self.Qvalue, self.TargetQval, self.trainer, self.weight1 = self.build_model()
		self.loss = tf.reduce_sum(tf.square(self.TargetQval - self.Qvalue))
		self.updateModel = self.trainer.minimize(self.loss)
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())


	# No need to use bias as the input is one-hot encoded
	def build_model(self):
		input_tensor = tf.placeholder(tf.float32, shape=(1, self.n_states))

		# Xavier Initialization for weights
		weights1 = tf.Variable(tf.random_uniform([16, 4], 0, 1.) * np.sqrt(2./4))
		Qvalue_tensor = tf.matmul(input_tensor, weights1)

		TargetQval_tensor = tf.placeholder(shape=[1, 4], dtype=tf.float32)
		trainer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
		return input_tensor, Qvalue_tensor, TargetQval_tensor, trainer, weights1


	def predict(self, state):
		return self.sess.run(self.Qvalue, feed_dict={self.input_tensor: state})


	def getGreedyAction(self, state):
		if np.random.rand(1) < self.epsilon:
			return [-1]
		else:
			return np.argmax(self.predict(state), 1)

	def learn(self, state, action, next_state, reward):
		newQVal = self.predict(next_state)

		TargetQValue = self.predict(state)
		# Update Q values using Bellman equation
		TargetQValue[0, action[0]] = reward + self.discount * np.max(newQVal)

		nn_dict = {self.input_tensor:state, self.TargetQval:TargetQValue}

		self.sess.run([self.updateModel, self.weight1], feed_dict=nn_dict)

	def print_weights(self):
		w1 = self.sess.run(self.weight1)
		print w1


	def decayEps(self, i):
		self.epsilon = 1./((i/50) + 10)
