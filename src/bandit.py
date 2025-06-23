import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, arms, epsilon=0.1):
        self.arms = arms
        self.epsilon = epsilon
        self.counts = np.zeros(len(arms))
        self.values = np.zeros(len(arms))

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.arms))
        else:
            return np.argmax(self.values)

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward 