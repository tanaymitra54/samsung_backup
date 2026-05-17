import yaml
import numpy as np
from copy import deepcopy


class SimulatedAnnealingSolver:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        sa_cfg = self.config["solver"]["sa"]
        self.initial_temp = sa_cfg["initial_temp"]
        self.final_temp = sa_cfg["final_temp"]
        self.cooling_rate = sa_cfg["cooling_rate"]
        self.iterations = sa_cfg["iterations"]
        self.num_reads = sa_cfg["num_reads"]

    def _compute_energy(self, state: np.ndarray, Q: np.ndarray) -> float:
        return state @ Q @ state

    def solve(self, Q: np.ndarray) -> tuple[np.ndarray, float]:
        n = Q.shape[0]
        best_state = None
        best_energy = float("inf")

        for _ in range(self.num_reads):
            state = np.random.randint(0, 2, size=n)
            current_energy = self._compute_energy(state, Q)
            temp = self.initial_temp

            for step in range(self.iterations):
                flip_idx = np.random.randint(0, n)
                state[flip_idx] = 1 - state[flip_idx]
                new_energy = self._compute_energy(state, Q)

                delta = new_energy - current_energy
                if delta < 0 or np.random.random() < np.exp(-delta / temp):
                    current_energy = new_energy
                else:
                    state[flip_idx] = 1 - state[flip_idx]

                temp = max(self.final_temp, temp * self.cooling_rate)

            if current_energy < best_energy:
                best_energy = current_energy
                best_state = state.copy()

        return best_state, best_energy
