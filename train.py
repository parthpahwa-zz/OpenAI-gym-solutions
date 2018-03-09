from DeepQNetwork import solution

PERFORM_UNDER_SAMPLING = True
ENV_NAME = "SpaceInvaders-v0"

player = solution.Player(ENV_NAME)
player.train(early_evaluate=True, undersampled=PERFORM_UNDER_SAMPLING)
