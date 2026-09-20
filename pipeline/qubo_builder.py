import yaml
import numpy as np
from itertools import combinations
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.answer_groups import normalise_answer
from pipeline.device_utils import resolve_device
from pipeline.qubo_math import pairwise_penalty


class QUBOBuilder:
    def __init__(self, config_path: str = "config/config.yaml", device: str | None = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        qubo_cfg = self.config["qubo"]
        self.max_vars = qubo_cfg["max_vars"]
        self.penalty_weight = qubo_cfg["penalty_weight"]
        self.diversity_bonus = qubo_cfg["diversity_bonus"]
        self.clustering_method = qubo_cfg["clustering_method"]
        self.n_clusters = min(self.max_vars, 50)
        self.hubo_enabled = qubo_cfg.get("hubo_enabled", False)
        self.hubo_triplet_penalty = qubo_cfg.get("hubo_triplet_penalty", 1.0)

        self.answer_sim_weight = qubo_cfg.get("answer_sim_weight", 1.0)
        self.answer_agree_weight = qubo_cfg.get("answer_agree_weight", 0.0)
        self.exact_k = qubo_cfg.get("exact_k", True)
        self.cardinality_penalty = qubo_cfg.get("cardinality_penalty", 0.0)
        self.top_k_per_cluster = qubo_cfg.get("top_k_per_cluster", 2)

        preferred_device = device or self.config.get("evaluation", {}).get("device")
        embedder_device = qubo_cfg.get("embedder_device") or str(resolve_device(preferred_device))
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=embedder_device)

    def _embed_reasons(self, reasons: list[str]) -> np.ndarray:
        return self.embedder.encode(reasons, convert_to_numpy=True)

    def _cluster_reasons(self, embeddings: np.ndarray, reason_indices: list[int]):
        n = min(len(embeddings), self.n_clusters)
        if n < 2:
            return [reason_indices] if reason_indices else []

        kmeans = KMeans(n_clusters=n, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        clusters = []
        for i in range(n):
            cluster_indices = [
                reason_indices[j] for j in range(len(reason_indices)) if labels[j] == i
            ]
            if cluster_indices:
                clusters.append(cluster_indices)
        return clusters

    def build_qubo(self, samples: list[dict]) -> tuple[np.ndarray, list[int]]:
        reasons = [s["reason"] for s in samples]
        embeddings = self._embed_reasons(reasons)
        indices = list(range(len(reasons)))

        clusters = self._cluster_reasons(embeddings, indices)
        selected_indices = []
        for cluster in clusters:
            if cluster:
                best = max(cluster, key=lambda i: samples[i].get("correctness_score", 0.5))
                selected_indices.append(best)
        if len(selected_indices) < 2:
            selected_indices = indices[: min(len(indices), 10)]

        selected_indices = selected_indices[: self.max_vars]
        selected_embeddings = embeddings[selected_indices]
        n = len(selected_indices)

        Q = np.zeros((n, n))

        for i in range(n):
            idx = selected_indices[i]
            correctness = samples[idx].get("correctness_score", 0.5)
            Q[i][i] = -correctness + self.diversity_bonus

        sim_matrix = cosine_similarity(selected_embeddings)
        answers = [
            samples[selected_indices[i]].get("answer", "") or ""
            for i in range(n)
        ]

        for i in range(n):
            for j in range(i + 1, n):
                a_i = normalise_answer(answers[i])
                a_j = normalise_answer(answers[j])
                agree = bool(a_i and a_j and a_i == a_j)
                pen = pairwise_penalty(
                    cos=float(sim_matrix[i][j]) * self.answer_sim_weight,
                    answers_agree=agree,
                    penalty_weight=self.penalty_weight,
                )
                Q[i][j] = pen
                Q[j][i] = pen

        if not self.exact_k and self.cardinality_penalty:
            k_target = self.config["pipeline"]["subset_size"]
            Q = self._apply_cardinality_constraint(
                Q, k=k_target, lambda_c=self.cardinality_penalty
            )

        return Q, selected_indices

    def _apply_cardinality_constraint(
        self, Q: np.ndarray, k: int, lambda_c: float
    ) -> np.ndarray:
        n = Q.shape[0]
        np.fill_diagonal(Q, Q.diagonal() + lambda_c * (1 - 2 * k))
        off_diag_mask = ~np.eye(n, dtype=bool)
        Q[off_diag_mask] += 2 * lambda_c
        return Q

    def build_hubo(self, samples: list[dict]) -> tuple[np.ndarray, list[int], dict]:
        Q, selected_indices = self.build_qubo(samples)
        if not self.hubo_enabled:
            return Q, selected_indices, {}

        selected_embeddings = self.embedder.encode(
            [samples[i]["reason"] for i in selected_indices], convert_to_numpy=True
        )
        n = len(selected_indices)
        n_triplets = min(n, self.max_vars)
        sim_matrix = cosine_similarity(selected_embeddings)

        T = {}
        triplet_count = 0
        max_triplets = 100

        for i, j, k in combinations(range(n_triplets), 3):
            if triplet_count >= max_triplets:
                break
            triple_sim = (sim_matrix[i][j] + sim_matrix[i][k] + sim_matrix[j][k]) / 3.0
            if triple_sim > 0.7:
                T[(i, j, k)] = triple_sim * self.hubo_triplet_penalty
                triplet_count += 1

        return Q, selected_indices, T

    def compute_hubo_energy(
        self, state: np.ndarray, Q: np.ndarray, T: dict
    ) -> float:
        energy = state @ Q @ state
        for (i, j, k), val in T.items():
            energy += val * state[i] * state[j] * state[k]
        return float(energy)
