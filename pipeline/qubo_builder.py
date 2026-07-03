"""
pipeline/qubo_builder.py  —  QUBO Matrix Construction for Reason Selection
===========================================================================

WHAT IS QUBO?
─────────────
QUBO = Quadratic Unconstrained Binary Optimisation.

We want to SELECT a small, high-quality, non-redundant subset of reasoning
traces from a large candidate pool. This is a combinatorial selection problem
that we encode as a QUBO so it can be solved by a simulated annealing solver.

BINARY DECISION VARIABLES:
    x_i ∈ {0, 1}  for each candidate reason i
    x_i = 1  →  reason i IS selected
    x_i = 0  →  reason i is NOT selected

THE QUBO ENERGY FUNCTION:
─────────────────────────
    E(x) = Σ_i     Q[i][i] · x_i              (diagonal  — quality reward)
           + Σ_{i<j} Q[i][j] · x_i · x_j      (off-diag — redundancy penalty)

The SOLVER MINIMISES E(x). We design Q so that:
    • High-quality reasons make Q[i][i] strongly negative → selecting them
      LOWERS energy → solver prefers them.
    • Highly similar (redundant) reason pairs make Q[i][j] strongly positive
      → selecting both RAISES energy → solver avoids redundant pairs.

DIAGONAL TERM  Q[i][i]  — QUALITY REWARD:
──────────────────────────────────────────
    Q[i][i] = −quality_score(i) + diversity_bonus

    WHY NEGATIVE quality_score:
    We need to turn "maximise quality" into "minimise energy".
    A reason with quality_score=0.9 → Q[i][i] = −0.9 + bonus (very attractive).
    A reason with quality_score=0.1 → Q[i][i] = −0.1 + bonus (barely attractive).

    WHY +diversity_bonus:
    Without this offset, Q[i][i] would always be negative (since quality ∈ [0,1]).
    The solver could then minimise energy by selecting ALL reasons (every x_i=1
    drives energy down). The bonus shifts the baseline so that selecting a
    low-quality reason costs more than not selecting it. Think of it as the
    "price of admission" — a reason must have high enough quality to justify
    its inclusion despite the potential redundancy costs it adds.

    Intuition: if diversity_bonus = 0.5, only reasons with quality_score > 0.5
    have negative Q[i][i] and are unconditionally attractive. Reasons below 0.5
    are only selected if the solver needs them for coverage (diversity).

OFF-DIAGONAL TERM  Q[i][j]  — REDUNDANCY PENALTY:
───────────────────────────────────────────────────
    Q[i][j] = cosine_similarity(embed_i, embed_j) × penalty_weight
             (same value for Q[j][i] — the matrix is symmetric)

    WHY POSITIVE:
    Selecting BOTH reasons i and j when they are similar adds Q[i][j] to the
    energy. Higher similarity → bigger penalty for co-selection → solver picks
    one or the other but rarely both.

    WHY COSINE SIMILARITY:
    Cosine similarity of sentence embeddings is a reliable, fast proxy for
    semantic redundancy. Two reasons that say the same thing in different words
    will have high cosine similarity. Two reasons that address different aspects
    of the problem will have low cosine similarity.

    WHY penalty_weight = 2.0 (default):
    The redundancy penalty must be large enough to overcome the quality reward
    of selecting both reasons. If two reasons each have quality_score=0.8,
    selecting both saves 2×0.8=1.6 energy. The penalty for a similarity of
    0.9 is 0.9×2.0=1.8, which just outweighs the reward — the solver correctly
    prefers to pick only the better one.

WORKED EXAMPLE  (3 candidate reasons, quality scores and embeddings known):
────────────────────────────────────────────────────────────────────────────
    Reason A: "12 × 5 = 60, the answer is 60"           quality=0.95
    Reason B: "multiply 12 by 5 to get 60, answer=60"   quality=0.82  (≈same as A)
    Reason C: "there are 60 total items because 3×4=12 sets of 5"  quality=0.78 (different path)

    Q[0][0] = −0.95 + 0.5 = −0.45    (A: strongly attractive)
    Q[1][1] = −0.82 + 0.5 = −0.32    (B: attractive)
    Q[2][2] = −0.78 + 0.5 = −0.28    (C: slightly attractive)

    Cosine similarities (from embeddings):
    sim(A,B) = 0.91  →  Q[0][1] = 0.91 × 2.0 = 1.82  (heavy penalty — A and B are paraphrases)
    sim(A,C) = 0.23  →  Q[0][2] = 0.23 × 2.0 = 0.46  (light penalty — different approach)
    sim(B,C) = 0.25  →  Q[1][2] = 0.25 × 2.0 = 0.50  (light penalty)

    Energy for selecting {A, C} (x=[1,0,1]):
        E = Q[0][0]×1 + Q[2][2]×1 + Q[0][2]×1×1
          = −0.45 + (−0.28) + 0.46 = −0.27   ← LOWER (preferred)

    Energy for selecting {A, B} (x=[1,1,0]):
        E = Q[0][0]×1 + Q[1][1]×1 + Q[0][1]×1×1
          = −0.45 + (−0.32) + 1.82 = +1.05   ← HIGHER (avoided)

    Result: solver correctly selects A and C — the best quality AND different approach.

WHY CLUSTER BEFORE BUILDING THE QUBO?
──────────────────────────────────────
With N=200 candidate reasons, the QUBO has N²/2 ≈ 20,000 off-diagonal terms.
Solving such a large QUBO is computationally expensive.

Instead, we:
  1. K-means cluster all N reasons by semantic embedding.
  2. Pick the BEST-quality reason from each cluster as a representative.
  3. Build the QUBO over only these k≤50 representatives.

This:
  (a) Reduces the QUBO from N variables to k — much faster to solve.
  (b) Guarantees coarse-level diversity BEFORE the penalty term runs:
      representatives from different clusters already address different
      aspects of the problem. The QUBO's off-diagonal penalty then fine-tunes
      across cluster representatives.

HUBO EXTENSION (higher-order terms):
──────────────────────────────────────
QUBO captures pairwise interactions. HUBO adds triplet terms T[(i,j,k)]·x_i·x_j·x_k.
If three reasons are all mutually similar, HUBO adds an extra penalty for
selecting all three simultaneously — catching redundancy that pairwise QUBO misses.
"""

import yaml
import numpy as np
from itertools import combinations
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.device_utils import resolve_device


class QUBOBuilder:
    """
    Constructs the Q matrix encoding the reason-selection problem as a QUBO.

    Diagonal  Q[i][i]: encodes the quality REWARD for selecting reason i.
    Off-diag  Q[i][j]: encodes the redundancy PENALTY for selecting the pair (i, j).

    The SimulatedAnnealingSolver in solver.py minimises E(x) = x^T Q x
    to find the optimal binary selection vector x.
    """

    def __init__(self, config_path: str = "config/config.yaml", device: str | None = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        qubo_cfg = self.config["qubo"]

        # max_vars: caps the QUBO size (number of binary variables).
        # Larger → more candidates considered, but solving is O(max_vars²) harder.
        # Default 200 is a practical balance for GPU-based simulated annealing.
        self.max_vars = qubo_cfg["max_vars"]

        # penalty_weight (λ): multiplier for the redundancy penalty.
        # Higher → solver more aggressively avoids similar reason pairs.
        # This controls the quality-diversity trade-off:
        #   too low  → solver picks redundant high-quality reasons (safe but boring)
        #   too high → solver avoids even moderately similar but complementary reasons
        self.penalty_weight = qubo_cfg["penalty_weight"]

        # diversity_bonus: constant added to every diagonal entry.
        # This is the "entry cost" for including a reason in the selection.
        # Only reasons with quality_score > diversity_bonus have net-negative Q[i][i]
        # and are unconditionally attractive to the solver.
        # See the module docstring for the full rationale.
        self.diversity_bonus = qubo_cfg["diversity_bonus"]

        self.clustering_method = qubo_cfg["clustering_method"]

        # n_clusters: how many K-means clusters (= how many QUBO variables after
        # the clustering step). Capped at max_vars so we never create more
        # clusters than the QUBO can accommodate.
        self.n_clusters = min(self.max_vars, 50)

        self.hubo_enabled = qubo_cfg.get("hubo_enabled", False)
        self.hubo_triplet_penalty = qubo_cfg.get("hubo_triplet_penalty", 1.0)

        # Fix #2 — Answer-aware off-diagonal weights
        self.answer_sim_weight   = qubo_cfg.get("answer_sim_weight",   0.6)
        self.answer_agree_weight = qubo_cfg.get("answer_agree_weight", 0.4)

        # Fix #3 — Soft cardinality constraint coefficient (λ_c)
        self.cardinality_penalty = qubo_cfg.get("cardinality_penalty", 0.1)

        # Fix #6 — Top-k representatives per cluster
        self.top_k_per_cluster = qubo_cfg.get("top_k_per_cluster", 2)

        preferred_device = device or self.config.get("evaluation", {}).get("device")
        embedder_device = qubo_cfg.get("embedder_device") or str(resolve_device(preferred_device))

        # SentenceTransformer: converts reason text to dense semantic vectors.
        # "all-MiniLM-L6-v2" is a 22M-parameter model that produces 384-dim
        # embeddings. It is fast and accurate enough for semantic similarity
        # estimation at our scale. The cosine similarity between two embeddings
        # is our proxy for semantic redundancy between two reasons.
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=embedder_device)

    # =========================================================================
    # ─── EMBEDDING & CLUSTERING ───────────────────────────────────────────────
    # =========================================================================

    def _embed_reasons(self, reasons: list[str]) -> np.ndarray:
        """
        Convert reason strings into fixed-size semantic embedding vectors.

        Output shape: (len(reasons), 384)  for MiniLM-L6-v2.

        WHY EMBEDDINGS:
        Two reasons that say the same thing in different words will produce
        geometrically similar embeddings — their cosine similarity will be high.
        This is more robust than string matching or TF-IDF overlap for capturing
        semantic redundancy in natural language reasoning traces.
        """
        return self.embedder.encode(reasons, convert_to_numpy=True)

    def _cluster_reasons(self, embeddings: np.ndarray, reason_indices: list[int]):
        """
        Group semantically similar reasons into K-means clusters.

        Returns a list of clusters, where each cluster is a list of
        indices into the original `samples` list.

        WHY K-MEANS IN EMBEDDING SPACE:
        K-means on sentence embeddings groups reasons that discuss the same
        aspect of the problem. Representatives from different clusters are
        therefore semantically diverse by construction — the QUBO's redundancy
        penalty then refines selection within the remaining cross-cluster space.

        HOW THE REPRESENTATIVES ARE CHOSEN (see build_qubo):
        Within each cluster, we select the top-k reasons by quality score.
        With k=2 (default), the QUBO can choose between the best AND second-best
        within each semantic group — preserving the optimizer's ability to make
        real tradeoff decisions rather than receiving only greedy pre-filtered reps.
        Singleton clusters (len == 1) contribute exactly 1 representative.

        EDGE CASE:
        If there are fewer than 2 reasons to cluster (degenerate input),
        return the whole list as a single cluster to avoid KMeans failure.
        """
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

    # =========================================================================
    # ─── QUBO MATRIX CONSTRUCTION ─────────────────────────────────────────────
    # =========================================================================

    def build_qubo(self, samples: list[dict]) -> tuple[np.ndarray, list[int]]:
        """
        Build the QUBO Q matrix from a list of scored reasoning traces.

        PIPELINE:
          Step 1 → Embed all reasons into semantic vector space.
          Step 2 → K-means cluster; pick top-k quality representatives per cluster
                   (Fix #6). Reduces from N candidates to at most n_clusters×k variables,
                   subject to the max_vars cap. Singleton clusters yield 1 rep.
          Step 3 → Build Q:
                   Diagonal  Q[i][i] = −quality_score + diversity_bonus
                   Off-diag  Q[i][j] = (α·cosine + β·answer_agree) × penalty_weight
                   where α = answer_sim_weight, β = answer_agree_weight  (Fix #2)
          Step 4 → Apply soft cardinality constraint (Fix #3):
                   Q[i][i] += λ_c × (1 − 2k)   for all i
                   Q[i][j] += 2λ_c              for all i ≠ j

        DEGENERATE CASE (zero-score inputs):
          When all correctness_score values are 0.0 — which happens when no gold
          answer is provided (e.g. during inference on unlabelled data) — all
          diagonals become 0 + diversity_bonus. The QUBO then reduces to pure
          diversity selection: the solver picks the most semantically spread-out
          subset. This is intentional and correct behaviour; the pipeline does not
          require a gold answer to function.

        Returns
        -------
        Q               : (k × k) float64 numpy array — the QUBO matrix.
                          Symmetric: Q[i][j] = Q[j][i].
        selected_indices: list of k indices into `samples`.
                          These are the reasons represented by each row/col of Q.
        """
        # ── Step 1: Embed ─────────────────────────────────────────────────
        reasons = [s["reason"] for s in samples]
        embeddings = self._embed_reasons(reasons)
        indices = list(range(len(reasons)))

        # ── Step 2: Cluster → top-k representatives per cluster ───────────
        # Fix #6: collect top_k_per_cluster best reasons from each cluster.
        # Singleton clusters always contribute exactly 1 representative.
        clusters = self._cluster_reasons(embeddings, indices)
        selected_indices = []
        for cluster in clusters:
            if cluster:
                sorted_cluster = sorted(
                    cluster,
                    key=lambda i: samples[i].get("correctness_score", 0.5),
                    reverse=True,
                )
                k_reps = min(self.top_k_per_cluster, len(sorted_cluster))
                selected_indices.extend(sorted_cluster[:k_reps])

        # Safety fallback: if clustering degenerated (< 2 representatives),
        # take the first N by index to keep the pipeline running.
        if len(selected_indices) < 2:
            selected_indices = indices[: min(len(indices), 10)]

        # Cap at max_vars to keep the QUBO tractable for the solver.
        selected_indices = selected_indices[: self.max_vars]
        selected_embeddings = embeddings[selected_indices]
        n = len(selected_indices)

        # ── Step 3: Build the Q matrix ────────────────────────────────────
        Q = np.zeros((n, n))

        # DIAGONAL: Q[i][i] = −quality_score + diversity_bonus
        for i in range(n):
            idx = selected_indices[i]
            correctness = samples[idx].get("correctness_score", 0.5)
            Q[i][i] = -correctness + self.diversity_bonus

        # OFF-DIAGONAL (Fix #2): Q[i][j] = (α·cosine + β·answer_agree) × penalty_weight
        #
        # Two chains that share the same extracted answer are logically redundant
        # even if their textual embeddings appear diverse. The answer_agree term
        # fires on logical redundancy that cosine similarity would miss.
        # Empty-answer guard: two chains with empty answers are NOT treated as
        # logically identical — we have no evidence of agreement in that case.
        sim_matrix = cosine_similarity(selected_embeddings)
        answers = [
            samples[selected_indices[i]].get("answer", "") or ""
            for i in range(n)
        ]

        def _normalise_ans(a: str) -> str:
            return a.strip().lower()

        for i in range(n):
            for j in range(i + 1, n):
                cos_sim = sim_matrix[i][j]
                a_i = _normalise_ans(answers[i])
                a_j = _normalise_ans(answers[j])
                agree = 1.0 if (a_i and a_j and a_i == a_j) else 0.0
                combined = self.answer_sim_weight * cos_sim + self.answer_agree_weight * agree
                Q[i][j] = combined * self.penalty_weight
                Q[j][i] = Q[i][j]   # symmetric

        # ── Step 4: Soft cardinality constraint (Fix #3) ──────────────────
        k_target = self.config["pipeline"]["subset_size"]
        Q = self._apply_cardinality_constraint(Q, k=k_target, lambda_c=self.cardinality_penalty)

        return Q, selected_indices

    def _apply_cardinality_constraint(
        self, Q: np.ndarray, k: int, lambda_c: float
    ) -> np.ndarray:
        """
        Apply the soft cardinality constraint penalty to Q and return it.

        CONSTRAINT: Penalise solutions that don't select exactly k variables.

        DERIVATION:
          Penalty term = λ_c × (Σ x_i − k)²
          Expanding (using x_i² = x_i for binary x_i):
            Diagonal  += λ_c × (1 − 2k)   for all i
            Off-diag  += 2λ_c              for all i ≠ j  (symmetric)

        WHY SOFT (λ_c = 0.1, NOT a hard constraint):
          Adding 2λ_c = 0.20 flat to every off-diagonal is well below the
          typical cosine-based penalty (penalty_weight × cosine ≈ 1.0–2.0).
          The cardinality nudge steers the solver without dominating diversity.
          λ_c needs empirical tuning per dataset — increase if the solver
          frequently selects too few or too many chains.
        """
        n = Q.shape[0]
        np.fill_diagonal(Q, Q.diagonal() + lambda_c * (1 - 2 * k))
        off_diag_mask = ~np.eye(n, dtype=bool)
        Q[off_diag_mask] += 2 * lambda_c
        return Q



    # =========================================================================
    # ─── HUBO EXTENSION ───────────────────────────────────────────────────────
    # =========================================================================

    def build_hubo(self, samples: list[dict]) -> tuple[np.ndarray, list[int], dict]:
        """
        Build a HUBO (Higher-Order UBO) extension on top of the QUBO.

        WHAT HUBO ADDS:
        QUBO captures PAIRWISE interactions: selecting both i and j when they
        are similar is penalised by Q[i][j] · x_i · x_j.

        But consider three reasons A, B, C where each pair has similarity 0.65
        (below the threshold at which any pair alone triggers a large penalty),
        yet all three together are essentially saying the same thing. QUBO would
        select all three. HUBO adds:

            T[(i,j,k)] · x_i · x_j · x_k

        This TRIPLET PENALTY fires only when ALL THREE are selected simultaneously,
        catching higher-order redundancy that pairwise QUBO misses.

        T[(i,j,k)] = average_pairwise_similarity(i,j,k) × hubo_triplet_penalty
        Only added for triplets where average pairwise sim > 0.7.

        IMPLEMENTATION NOTE:
        HUBO solving requires the compute_hubo_energy() method to be used
        instead of the standard QUBO energy E = x^T Q x.
        This is enabled/disabled via config.yaml → qubo.hubo_enabled.
        """
        Q, selected_indices = self.build_qubo(samples)
        if not self.hubo_enabled:
            # HUBO is off — return empty triplet dict so callers work uniformly
            return Q, selected_indices, {}

        selected_embeddings = self.embedder.encode(
            [samples[i]["reason"] for i in selected_indices], convert_to_numpy=True
        )
        n = len(selected_indices)
        n_triplets = min(n, self.max_vars)
        sim_matrix = cosine_similarity(selected_embeddings)

        T = {}
        triplet_count = 0
        max_triplets = 100   # cap to avoid combinatorial blowup (C(50,3) = 19,600 triplets)

        for i, j, k in combinations(range(n_triplets), 3):
            if triplet_count >= max_triplets:
                break
            # Average pairwise similarity within the triplet (i,j,k).
            # If all three are mutually similar, this will be high.
            triple_sim = (sim_matrix[i][j] + sim_matrix[i][k] + sim_matrix[j][k]) / 3.0
            if triple_sim > 0.7:
                # Only add the triplet term if the three reasons are highly similar
                # (threshold 0.7 empirically chosen to filter noise).
                T[(i, j, k)] = triple_sim * self.hubo_triplet_penalty
                triplet_count += 1

        return Q, selected_indices, T

    def compute_hubo_energy(
        self, state: np.ndarray, Q: np.ndarray, T: dict
    ) -> float:
        """
        Compute total HUBO energy for a binary state vector.

        E(x) = x^T Q x                               (standard QUBO quadratic term)
               + Σ_{(i,j,k) ∈ T} T[(i,j,k)]·x_i·x_j·x_k  (HUBO triplet term)

        The triplet term only contributes when all three variables in a triplet
        are simultaneously set to 1 — this is what makes it "higher order".
        Used by the simulated annealing solver when hubo_enabled=True.
        """
        energy = state @ Q @ state
        for (i, j, k), val in T.items():
            energy += val * state[i] * state[j] * state[k]
        return float(energy)
