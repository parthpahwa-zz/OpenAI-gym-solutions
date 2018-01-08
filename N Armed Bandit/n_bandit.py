import numpy as np

class Bandit:

    def __init__(self):
        self.state = -1
        self.bandits = np.array([[1.0, 0.90, 0.4, 0.6], [0.1, 0.09, 0.06, 0.25], [0.45, .55, .25, .1], [0.14, .2, .3, .12]])
        self.n_bandits = self.bandits.shape[0]
        self.n_actions = self.bandits.shape[1]


    def pull_arm(self, action):
        value = self.bandits[self.state, action]
        result = np.random.randn(1)

        if result > value:
            return 1
        return -1

    def select_bandit(self):
        self.state = np.random.randint(0, len(self.bandits))
        return self.state
