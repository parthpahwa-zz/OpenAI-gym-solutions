import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

class Agent:
	def __init__(self, n_actions, input_dim, n_hidden, lr=0.01):
		tf.reset_default_graph()
		self.lr = lr
		self.input_tensor, self.output, self.trainer = self.build_model(input_dim, n_hidden, n_actions)
		self.policy_list = tf.placeholder(tf.float32, shape=[None, 1], name="policy_list")
		self.advantage = tf.placeholder(tf.float32, shape=[None, 1], name="reward_list")
		self.gradients = self.define_grds(self.policy_list, self.advantage, self.output)
		self.update_grads = self.trainer.apply_gradients(zip([self.Weight_1, self.Weight_2], self.trainable_vars))
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())


	def build_model(self, input_dim, n_hidden, n_actions):
		self.input_tensor = tf.placeholder(shape=[None, input_dim], dtype=tf.float32)
		hidden = slim.fully_connected(self.input_tensor, n_hidden, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
		output = slim.fully_connected(hidden, 1, activation_fn=tf.nn.sigmoid)
		trainer = tf.train.AdamOptimizer(learning_rate=self.lr)
		return self.input_tensor, output, trainer


	def define_grds(self, policy_list, advantage, output):
		self.Weight_1 = tf.placeholder(tf.float32, name="W1_grad")
		self.Weight_2 = tf.placeholder(tf.float32, name="W2_grad")

		mod_policy = tf.log(policy_list*(policy_list - output) + (1 -policy_list)*(policy_list + output))
		loss = tf.reduce_mean(mod_policy * advantage)
		var_temp = tf.trainable_variables()

		self.trainable_vars = [var_temp[0], var_temp[1]]
		grads = tf.gradients(loss, self.trainable_vars)
		return grads


	def predict(self, state):
		policy = self.sess.run(self.output, feed_dict={self.input_tensor:state})
		action = 0 if policy > np.random.uniform() else 1
		return action

	def compute_gradient(self, history, action, advantage):
		feed_dict = {self.input_tensor:history, self.policy_list:action, self.advantage:advantage}
		return self.sess.run(self.gradients, feed_dict=feed_dict)

	def learn(self, grads):
		feed_dict = {self.Weight_1: grads[0], self.Weight_2: grads[1]}
		return self.sess.run(self.update_grads, feed_dict=feed_dict)

	def get_trainable_var(self):
		return self.sess.run(self.trainable_vars)
