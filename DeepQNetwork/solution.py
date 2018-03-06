import os.path
import ast
from agent import Agent
from keras.models import load_model
from environment import Environment

WIDTH = 84
HEIGHT = 84
N_FRAMES = 4

class Player:
	def __init__(self):
		self.game = Environment('SpaceInvaders-v0', WIDTH, HEIGHT, N_FRAMES)
		self.agent = Agent(HEIGHT, WIDTH, self.game.env.action_space.n, nframes=N_FRAMES)
		self.weights_filename = "Records/weights.h5"
		self.config_filename = 'Records/config.txt'
		self.score_filename = 'Records/score.txt'
		self.qval_filename = 'Records/q_val.txt'
		if not os.path.exists("Records"):
			os.makedirs("Records")

	def train(self, early_evaluate=False, undersampled=False):
		itr, avg, max_score = self.load_config()

		try:
			while self.agent.counter < 50000000:
				reward, q_val_list, q_val = self.game.train(self.agent, undersampled)

				if reward > max_score:
					max_score = reward

				if itr % 1000 == 0 and itr != 0:
					if itr % 3500 == 0:
						self.perform_quicksave(itr, avg, max_score, save_memory=True)
					else:
						self.perform_quicksave(itr, avg, max_score)

				string = str(reward) + " Itr: " + str(itr) + " Qval: " + str(q_val) + " Max: " + str(max_score) + "\n"
				self.quick_write(string, q_val_list)

				if early_evaluate:
					if reward >= 0.7*avg:
						avg = self.perform_earlystop(avg)
				itr += 1
		finally:
			self.perform_quicksave(itr, avg, max_score, save_memory=True)


	def perform_earlystop(self, avg):
		itr = 0
		total_reward = 0
		while itr < 20:
			reward, q_val = self.game.test(self.agent)
			total_reward += reward
			itr += 1

		if avg <= total_reward/20.0:
			if os.path.isfile("Records/best_performace_" + str(avg) + ".h5"):
				os.remove("Records/best_performace_" + str(avg) + ".h5")
			avg = total_reward/20.0
			self.agent.model.save("Records/best_performace_" + str(avg) + ".h5")
		return avg


	def test(self):
		self.load_config()
		print "LOADED"
		itr = 0
		try:
			while itr < 100:
				reward, q_val = self.game.test(self.agent)
				string = "Reward: " + str(reward) + " Q value: " + str(q_val) + " Iteration: " + str(itr)
				print string
				itr += 1

		finally:
			pass


	def perform_quicksave(self, itr, avg, max_score, save_memory=False):
		self.agent.model.save(self.weights_filename)

		config_file = open(self.config_filename, 'w')
		config = {"itr":itr, "eps":self.agent.eps, "avg":avg, "frame":self.agent.counter, "max":max_score}

		config_file.write(str(config))
		config_file.close()

		if save_memory:
			self.agent.memory.save_memory()


	def load_config(self, test=False):
		if os.path.isfile(self.weights_filename) and not test:
			self.agent.model = load_model(self.weights_filename)
			self.agent.memory.load_memory()
			print "Weights loaded"

		if os.path.isfile(self.config_filename):
			config_file = open(self.config_filename, 'r')

			config = ast.literal_eval(config_file.read())
			itr = int(config["itr"])
			avg = float(config["avg"])
			max_score = float(config["max"])
			self.agent.counter = float(config["frame"])
			self.agent.eps = float(config["eps"])
			config_file.close()
			print "Config loaded"

			if test:
				self.agent.model = load_model("Records/best_performace_" + str(avg) + ".h5")
				print "Weights loaded"
			return itr, avg, max_score

		return 0, -1.0, -1.0


	def quick_write(self, string, q_val):
		target = open(self.score_filename, 'a')
		target.write(string)
		target.close()

		target = open(self.qval_filename, 'a')
		target.write(str(q_val) + "\n")
		target.close()
