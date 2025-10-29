from typing import Dict, Optional
import flwr as fl
import numpy as np


class EpsilonGreedyBandit:
    def __init__(self, arms: list, epsilon: float = 0.2):
        self.arms = arms
        self.epsilon = epsilon
        self.counts = np.zeros(len(arms), dtype=np.int64)
        self.values = np.zeros(len(arms), dtype=np.float32)

    def select_arm(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.arms))
        return int(np.argmax(self.values))

    def update(self, arm_idx: int, reward: float) -> None:
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]
        v = self.values[arm_idx]
        self.values[arm_idx] = ((n - 1) / n) * v + (1 / n) * reward


class BanditStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *,
        algo_arms: list[str] = ("ncf",),
        epoch_arms: list[int] = (1, 2, 3),
        epsilon: float = 0.2,
        dp_clip_norm: Optional[float] = None,
        dp_sigma: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.algo_bandit = EpsilonGreedyBandit(list(algo_arms), epsilon)
        self.epoch_bandit = EpsilonGreedyBandit(list(epoch_arms), epsilon)
        self.dp_clip_norm = dp_clip_norm
        self.dp_sigma = dp_sigma
        self._last_algo_idx: int | None = None
        self._last_epoch_idx: int | None = None

    def configure_fit(self, server_round: int, parameters, client_manager):
        algo_arm_idx = self.algo_bandit.select_arm()
        epochs_arm_idx = self.epoch_bandit.select_arm()
        self._last_algo_idx = algo_arm_idx
        self._last_epoch_idx = epochs_arm_idx
        algo = self.algo_bandit.arms[algo_arm_idx]
        epochs = int(self.epoch_bandit.arms[epochs_arm_idx])
        config = {
            "algo": algo,
            "epochs": epochs,
            "dp_clip_norm": 0.0 if self.dp_clip_norm is None else float(self.dp_clip_norm),
            "dp_sigma": 0.0 if self.dp_sigma is None else float(self.dp_sigma),
        }
        fit_ins = fl.server.strategy.FedAvg.configure_fit(self, server_round, parameters, client_manager)
        # Inject config into all client fit instructions
        fit_cfg = []
        for ins in fit_ins:
            ins.config.update(config)
            fit_cfg.append(ins)
        return fit_cfg

    def aggregate_evaluate(self, server_round: int, results, failures):
        agg = super().aggregate_evaluate(server_round, results, failures)
        # Use average loss as negative reward for bandit
        if agg is not None:
            _, metrics = agg
            loss = metrics.get("loss") if isinstance(metrics, dict) else None
            if loss is not None and self._last_algo_idx is not None and self._last_epoch_idx is not None:
                reward = max(0.0, 1.0 - float(loss))
                # Update the bandits with the chosen arms from this round
                self.algo_bandit.update(self._last_algo_idx, reward)
                self.epoch_bandit.update(self._last_epoch_idx, reward)
        return agg


def start_server(num_rounds: int = 3, dp_clip_norm: float | None = 1.0, dp_sigma: float | None = 0.01) -> None:
    strategy = BanditStrategy(
        algo_arms=["ncf"],
        epoch_arms=[1, 2, 3],
        epsilon=0.2,
        dp_clip_norm=dp_clip_norm,
        dp_sigma=dp_sigma,
        min_fit_clients=3,
        min_available_clients=3,
        min_evaluate_clients=3,
    )
    fl.server.start_server(server_address="0.0.0.0:8080", config=fl.server.ServerConfig(num_rounds=num_rounds), strategy=strategy)


if __name__ == "__main__":
    start_server()


