import numpy as np


class RandomAgent:
    def __init__(self, action_space):
        self.action_space_size = action_space.shape[0]

    def act(self, obs):
        a = np.random.normal(0, 0.75, self.action_space_size)
        return a