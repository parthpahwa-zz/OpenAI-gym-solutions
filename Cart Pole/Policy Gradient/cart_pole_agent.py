import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


class Agent:

	def __init__(self, n_actions, input_dim, n_hidden, lr=0.03, eps=0.1):
		self.n_actions = n_actions
		self.lr = lr
		self.n_hidden = n_hidden
		self.epsilon = eps
		self.dim = input_dim
		tf.reset_default_graph()
		self.input_tensor, self.output, self.trainer = self.build_model()
		self.chosen_action = tf.argmax(self.output, 1)

		self.reward_list = tf.placeholder(shape=[None], dtype=tf.float32)
		self.action_list = tf.placeholder(shape=[None], dtype=tf.int32)

		self.index_list = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_list
		self.policy_list = tf.gather(tf.reshape(self.output, [-1]), self.index_list)

		self.loss = -1.0 * tf.reduce_mean(tf.log(self.policy_list)*self.reward_list)
		self.trainable_vars = tf.trainable_variables()

		self.gradients = tf.gradients(self.loss, self.trainable_vars)
		self.grad_list = []
		for indx, var in enumerate(self.trainable_vars):
			self.grad_list.append(tf.placeholder(dtype=tf.float32, name=str(indx)+"_holder"))

		self.update_grads = self.trainer.apply_gradients(zip(self.grad_list, self.trainable_vars))

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())



	# No need to use bias as the input is one-hot encoded
	def build_model(self):
		self.input_tensor = tf.placeholder(shape=[None, self.dim], dtype=tf.float32)
		hidden = slim.fully_connected(self.input_tensor, self.n_hidden, biases_initializer=None, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
		output = slim.fully_connected(hidden, self.n_actions, activation_fn=tf.nn.softmax, biases_initializer=None)
		trainer = tf.train.AdamOptimizer(learning_rate=self.lr)
		return self.input_tensor, output, trainer


	def predict(self, state):
		a_dist = self.sess.run(self.output, feed_dict={self.input_tensor:[state]})
		action = np.random.choice(a_dist[0], p=a_dist[0])
		action = np.argmax(a_dist == action)
		return action

	def compute_gradient(self, history):
		feed_dict = {self.reward_list:history[:, 2], self.action_list:history[:, 1], self.input_tensor:np.vstack(history[:, 0])}
		return self.sess.run(self.gradients, feed_dict=feed_dict)

	def learn(self, feed_dict):
		# feed_dict = dict(zip(self.grad_list, gradient_buffer))
		# print feed_dict
		return self.sess.run(self.update_grads, feed_dict=feed_dict)

	def get_trainable_var(self):
		return self.sess.run(self.trainable_vars)

	def decayEps(self, i):
self.epsilon = 1./((i/10) + 10)
