import yaml
import numpy as np


class HyperparameterQUBO:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def _discretize_param(self, param_name: str, param_range: list) -> list[float]:
        if param_name == "temperature" or param_name == "top_p":
            return np.linspace(param_range[0], param_range[1], 5).tolist()
        elif param_name in ("num_answers", "num_reasons", "subset_size"):
            return list(range(param_range[0], param_range[1] + 1, 2))
        return param_range

    def _one_hot_block(self, n_values: int, start_idx: int) -> tuple[np.ndarray, int]:
        block = np.ones((1, n_values))
        return block, start_idx + n_values

    def build_hyperparam_qubo(
        self, param_grid: dict[str, list]
    ) -> np.ndarray:
        blocks = []
        var_offset = 0
        total_vars = 0

        param_mapping = {}
        for param_name, values in param_grid.items():
            discrete = self._discretize_param(param_name, values)
            block_size = len(discrete)
            param_mapping[param_name] = {
                "start_idx": total_vars,
                "values": discrete,
                "block_size": block_size,
            }
            total_vars += block_size
            blocks.append(block_size)

        Q = np.zeros((total_vars, total_vars))

        offset = 0
        for block_size in blocks:
            for i in range(block_size):
                for j in range(block_size):
                    if i != j:
                        Q[offset + i][offset + j] = 10.0
            offset += block_size

        return Q, param_mapping

    def decode_solution(
        self, state: np.ndarray, param_mapping: dict
    ) -> dict:
        selected_params = {}
        for param_name, info in param_mapping.items():
            start = info["start_idx"]
            block = state[start: start + info["block_size"]]
            if np.sum(block) > 0:
                chosen = np.argmax(block)
            else:
                chosen = 0
            selected_params[param_name] = info["values"][chosen]
        return selected_params
