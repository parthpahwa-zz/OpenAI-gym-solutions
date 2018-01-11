import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


class Model:
	def __init__(self, input_dim, n_hidden, lr=0.01):
		tf.reset_default_graph()
		self.lr = lr
		self.input_tensor, self.output, self.trainer = self.build_model(input_dim, n_hidden)
		self.loss = self.define_loss(input_dim)
		self.update_grads = self.trainer.minimize(self.loss)
		self.predicted_output = tf.concat([output[0], output[1], output[2]], axis=1),
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())


	def build_model(self, input_dim, n_hidden):
		self.input_tensor = tf.placeholder(shape=[None, input_dim + 1], dtype=tf.float32)

		hidden_1 = slim.fully_connected(self.input_tensor, n_hidden, biases_initializer=None, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())
		hidden_2 = slim.fully_connected(hidden_1, n_hidden, biases_initializer=None, activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer())

		output_state = slim.fully_connected(hidden_2, input_dim + 1, biases_initializer=None)
		output_reward = slim.fully_connected(hidden_2, 1, biases_initializer=None)
		output_done = slim.fully_connected(hidden_2, 1, activation_fn=tf.nn.sigmoid, biases_initializer=None)

		trainer = tf.train.AdamOptimizer(learning_rate=self.lr)
		return self.input_tensor, [output_state, output_reward, output_done], trainer


	def define_loss(self, input_dim):
		self.obs_state = tf.placeholder(tf.float32, shape=[None, input_dim + 1])
		self.obs_reward = tf.placeholder(tf.float32, shape=[None, 1])
		self.obs_done = tf.placeholder(tf.float32, shape=[None, 1])

		loss_state = tf.square(self.obs_state - self.output[0])
		loss_reward = tf.square(self.obs_reward - self.output[1])
		loss_done = -1.0*tf.log(self.obs_done * self.output[2] + (1 - self.obs_done) * (1 - self.output[2]))

		loss = tf.reduce_max(loss_state + loss_reward + loss_done)
		return loss


	def predict(self, state):
		output = self.sess.run(self.predicted_output, feed_dict={self.input_tensor:state})
		return output


	def learn(self, input_tensor, obs_state, obs_reward, obs_done):
		feed_dict = {self.input_tensor: input_tensor.astype(np.float32), self.obs_state: obs_state.astype(np.float32),
		self.obs_reward: obs_reward.astype(np.float32), self.obs_done: obs_done.astype(np.float32)}
		return self.sess.run([self.loss, self.update_grads], feed_dict=feed_dict)
