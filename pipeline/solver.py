import yaml
import numpy as np
from copy import deepcopy

from pipeline.device_utils import resolve_device


class SimulatedAnnealingSolver:
    def __init__(self, config_path: str = "config/config.yaml", device: str | None = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        sa_cfg = self.config["solver"]["sa"]
        self.initial_temp = sa_cfg["initial_temp"]
        self.final_temp = sa_cfg["final_temp"]
        self.cooling_rate = sa_cfg["cooling_rate"]
        self.iterations = sa_cfg["iterations"]
        self.num_reads = sa_cfg["num_reads"]

        preferred_device = device or self.config.get("evaluation", {}).get("device")
        self.device = resolve_device(preferred_device)
        gpu_cfg = self.config["solver"].get("gpu", {})
        self.gpu_enabled = gpu_cfg.get("enabled", False) and self.device.type == "cuda" and self._cuda_available()
        self.num_parallel_reads = gpu_cfg.get("num_parallel_reads", 1024)
        self.use_parallel_tempering = gpu_cfg.get("use_parallel_tempering", False)
        self.use_counterdiabatic = gpu_cfg.get("use_counterdiabatic", False)

    def _cuda_available(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _compute_energy(self, state: np.ndarray, Q: np.ndarray) -> float:
        return state @ Q @ state

    def solve(self, Q: np.ndarray) -> tuple[np.ndarray, float]:
        if self.gpu_enabled:
            device = str(self.device)
            import torch
            Q_t = torch.tensor(Q, dtype=torch.float32, device=device)
            if self.use_parallel_tempering:
                return self._solve_parallel_tempering_gpu(Q_t)
            elif self.use_counterdiabatic:
                return self._solve_counterdiabatic_gpu(Q_t)
            else:
                return self._solve_gpu(Q_t)
        return self._solve_cpu(Q)

    def _solve_cpu(self, Q: np.ndarray) -> tuple[np.ndarray, float]:
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

    def _solve_gpu(self, Q: "torch.Tensor") -> tuple[np.ndarray, float]:
        import torch
        n = Q.shape[0]
        num_r = self.num_parallel_reads
        states = torch.randint(0, 2, (num_r, n), device=str(self.device), dtype=torch.float32)
        energies = torch.einsum('ri,ij,rj->r', states, Q, states)

        for step in range(self.iterations):
            t = max(self.final_temp, self.initial_temp * (self.cooling_rate ** step))
            flip_idx = torch.randint(0, n, (num_r,), device=str(self.device))
            states[torch.arange(num_r), flip_idx] = 1.0 - states[torch.arange(num_r), flip_idx]
            new_energies = torch.einsum('ri,ij,rj->r', states, Q, states)
            delta = new_energies - energies
            accept = (delta < 0) | (torch.rand(num_r, device=str(self.device)) < torch.exp(-delta / max(t, 1e-8)))
            revert_idx = flip_idx[~accept]
            states[~accept, revert_idx] = 1.0 - states[~accept, revert_idx]
            energies = torch.where(accept, new_energies, energies)

        best_idx = torch.argmin(energies)
        best_state_np = states[best_idx].cpu().numpy().astype(int)
        best_energy = energies[best_idx].item()
        return best_state_np, best_energy

    def _solve_parallel_tempering_gpu(self, Q: "torch.Tensor") -> tuple[np.ndarray, float]:
        import torch
        n = Q.shape[0]
        n_replicas = min(self.num_parallel_reads, 64)
        n_temps = max(2, n_replicas // 4)
        replicas_per_temp = n_replicas // n_temps
        total = replicas_per_temp * n_temps

        states = torch.randint(0, 2, (total, n), device=str(self.device), dtype=torch.float32)
        temps = torch.logspace(
            np.log10(self.initial_temp), np.log10(max(self.final_temp, 0.01)), n_temps, device=str(self.device)
        )
        replica_temps = temps.repeat_interleave(replicas_per_temp)
        energies = torch.einsum('ri,ij,rj->r', states, Q, states)

        for step in range(self.iterations):
            t_vals = torch.clamp(replica_temps * (self.cooling_rate ** step), min=0.001)
            flip_idx = torch.randint(0, n, (total,), device=str(self.device))
            states[torch.arange(total), flip_idx] = 1.0 - states[torch.arange(total), flip_idx]
            new_energies = torch.einsum('ri,ij,rj->r', states, Q, states)
            delta = new_energies - energies
            accept = (delta < 0) | (torch.rand(total, device=str(self.device)) < torch.exp(-delta / t_vals))
            revert_idx = flip_idx[~accept]
            states[~accept, revert_idx] = 1.0 - states[~accept, revert_idx]
            energies = torch.where(accept, new_energies, energies)

            if step % 10 == 0:
                # Replica exchange, vectorised.
                #
                # This was a Python double loop over temperature pairs and
                # replicas, with .item() calls inside it. Every .item() forces a
                # GPU->CPU sync, and the block runs ~3000 times per solve, which
                # made annealing 22% of benchmark wall time once the pool grew to
                # 20 chains (4.0 min per 20 questions, up from 1.0).
                #
                # Adjacent-pair swaps cannot all be applied at once -- pairs
                # (k, k+1) and (k+1, k+2) would both claim replica k+1 -- so this
                # alternates even and odd pair sets on successive exchange rounds,
                # the standard even-odd scheme. Both views share storage with
                # `states` and `energies`, so writes propagate.
                e_view = energies.view(n_temps, replicas_per_temp)
                s_view = states.view(n_temps, replicas_per_temp, n)
                parity = (step // 10) % 2
                ks = torch.arange(parity, n_temps - 1, 2, device=str(self.device))
                if ks.numel() > 0:
                    beta = 1.0 / temps
                    beta_diff = (beta[ks] - beta[ks + 1]).unsqueeze(1)
                    de = e_view[ks + 1] - e_view[ks]
                    swap = torch.rand(
                        (ks.numel(), replicas_per_temp), device=str(self.device)
                    ) < torch.exp(-beta_diff * de)

                    mask = swap.unsqueeze(-1)
                    s_lo = s_view[ks].clone()
                    s_hi = s_view[ks + 1].clone()
                    s_view[ks] = torch.where(mask, s_hi, s_lo)
                    s_view[ks + 1] = torch.where(mask, s_lo, s_hi)

                    e_lo = e_view[ks].clone()
                    e_hi = e_view[ks + 1].clone()
                    e_view[ks] = torch.where(swap, e_hi, e_lo)
                    e_view[ks + 1] = torch.where(swap, e_lo, e_hi)

        best_idx = torch.argmin(energies)
        return states[best_idx].cpu().numpy().astype(int), energies[best_idx].item()

    def _solve_counterdiabatic_gpu(self, Q: "torch.Tensor") -> tuple[np.ndarray, float]:
        import torch
        n = Q.shape[0]
        num_r = self.num_parallel_reads
        states = torch.randint(0, 2, (num_r, n), device=str(self.device), dtype=torch.float32)
        energies = torch.einsum('ri,ij,rj->r', states, Q, states)
        Q.requires_grad_(True)

        for step in range(self.iterations):
            t = max(self.final_temp, self.initial_temp * (self.cooling_rate ** step))
            dt = self.initial_temp * (self.cooling_rate ** step) * np.log(1.0 / self.cooling_rate) if step > 0 else 0.0
            flip_idx = torch.randint(0, n, (num_r,), device=str(self.device))
            states[torch.arange(num_r), flip_idx] = 1.0 - states[torch.arange(num_r), flip_idx]
            new_energies = torch.einsum('ri,ij,rj->r', states, Q, states)
            delta = new_energies - energies
            cd_correction = abs(dt) * torch.sum(torch.abs(new_energies - energies).unsqueeze(1).expand(-1, n) * (1.0 - 2.0 * states), dim=1)
            delta_cd = delta + cd_correction * 0.1
            accept = (delta_cd < 0) | (torch.rand(num_r, device=str(self.device)) < torch.exp(-delta_cd / max(t, 1e-8)))
            revert_idx = flip_idx[~accept]
            states[~accept, revert_idx] = 1.0 - states[~accept, revert_idx]
            energies = torch.where(accept, new_energies, energies)

        Q.requires_grad_(False)
        best_idx = torch.argmin(energies)
        return states[best_idx].cpu().numpy().astype(int), energies[best_idx].item()
