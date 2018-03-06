from DeepQNetwork import solution

PERFORM_UNDER_SAMPLING = True

player = solution.Player()
player.train(undersampled=PERFORM_UNDER_SAMPLING)
