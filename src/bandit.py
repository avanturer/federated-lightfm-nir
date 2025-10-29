import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, arms, epsilon=0.1, rng: np.random.Generator | None = None):
        self.arms = arms
        self.epsilon = epsilon
        self.counts = np.zeros(len(arms))
        self.values = np.zeros(len(arms))
        self.rng = rng if rng is not None else np.random.default_rng()

    def select_arm(self):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(len(self.arms)))
        else:
            return int(np.argmax(self.values))

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.values[chosen_arm]
        self.values[chosen_arm] = ((n - 1) / n) * value + (1 / n) * reward 

    def set_epsilon(self, epsilon: float):
        self.epsilon = float(epsilon)