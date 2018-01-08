import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class Agent:

	def __init__(self, n_actions, n_states, lr=0.1, eps=0.1):
		self.n_actions = n_actions
		self.n_states = n_states
		self.lr = lr
		self.epsilon = eps
		tf.reset_default_graph()
		self.current_state = tf.placeholder(shape=[1], dtype=tf.int32)
		self.input_tensor, self.output, self.trainer = self.build_model()
		self.chosen_action = tf.argmax(self.output, 0)

		self.reward = tf.placeholder(shape=[1], dtype=tf.float32)
		self.action = tf.placeholder(shape=[1], dtype=tf.int32)
		self.policy = tf.slice(self.output, self.action, [1])

		self.loss = -(tf.log(self.policy)*self.reward)
		self.updateModel = self.trainer.minimize(self.loss)
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())


	def build_model(self):
		input_tensor_OH = slim.one_hot_encoding(self.current_state, self.n_states)

		output = slim.fully_connected(input_tensor_OH, self.n_actions, biases_initializer=None, activation_fn=tf.nn.sigmoid, weights_initializer=tf.random_normal_initializer)
		output = tf.reshape(output, [-1])
		trainer = tf.train.AdagradOptimizer(learning_rate=self.lr)
		return input_tensor_OH, output, trainer


	def predict(self, state):
		return self.sess.run(self.action, feed_dict={self.input_tensor: state})


	def getGreedyAction(self, state):
		if np.random.rand(1) < self.epsilon:
			return [-1]
		else:
			return np.argmax(self.predict(state), 1)


	def learn(self, state, action, reward):
		weights = tf.trainable_variables()
		feed_dict = {self.current_state:state, self.reward:reward, self.action:action}
		return self.sess.run([self.updateModel, weights], feed_dict=feed_dict)


	def decayEps(self, i):
		self.epsilon = 1./((i/10) + 10)
