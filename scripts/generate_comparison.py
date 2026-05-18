import csv
import time
import sys
import os
from datetime import datetime
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.sampling import DiverseSampler
from pipeline.verifier import ReasonVerifier
from pipeline.qubo_builder import QUBOBuilder
from pipeline.solver import SimulatedAnnealingSolver


QUESTIONS = [
    "If x + y = 10 and x - y = 4, find the values of x and y.",
    "A train travels 120 km in 2 hours. If it speeds up by 20 km/h, what is the new speed?",
    "A rectangle has perimeter 36 cm and its length is twice its width. Find its area.",
    "Solve for z: 3z - 7 = 2z + 5.",
    "A shop sells apples for $0.50 each and oranges for $0.75 each. If John buys 4 apples and 3 oranges, how much does he pay?",
    "All cats are mammals. Some mammals are predators. Can we conclude all cats are predators?",
]

K_SELECTED = 6


def build_embeddings(samples, embedder):
    from sentence_transformers import SentenceTransformer
    if embedder is None:
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    reasons = [s["reason"] for s in samples]
    embs = embedder.encode(reasons, convert_to_numpy=True)
    return embs, embedder


def compute_redundancy(selected_indices, embeddings):
    if len(selected_indices) < 2:
        return 0.0
    selected_embs = embeddings[selected_indices]
    sims = np.dot(selected_embs, selected_embs.T)
    n = len(selected_indices)
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(sims[mask]))


def compute_relevance(selected_indices, samples):
    scores = [samples[i].get("correctness_score", 0.0) for i in selected_indices]
    return float(np.mean(scores)) if scores else 0.0


def compute_qubo_energy(selected_indices, Q, qubo_var_indices):
    idx_map = {orig: pos for pos, orig in enumerate(qubo_var_indices)}
    n = Q.shape[0]
    state = np.zeros(n, dtype=int)
    for orig_idx in selected_indices:
        if orig_idx in idx_map:
            state[idx_map[orig_idx]] = 1
    return float(state @ Q @ state)


def method_base(samples, Q, qubo_var_indices, solver):
    state, energy = solver.solve(Q)
    selected = [qubo_var_indices[i] for i in range(len(state)) if state[i] == 1]
    if len(selected) < K_SELECTED:
        available = [i for i in range(len(samples)) if i not in set(selected)]
        selected.extend(
            sorted(available, key=lambda i: samples[i].get("correctness_score", 0.0), reverse=True)
            [:(K_SELECTED - len(selected))]
        )
    elif len(selected) > K_SELECTED:
        scores = [samples[i].get("correctness_score", 0.0) for i in selected]
        selected = [selected[i] for i in np.argsort(scores)[::-1][:K_SELECTED]]
    return selected[:K_SELECTED]


def method_self_consistency(samples, Q, qubo_var_indices):
    return list(range(min(K_SELECTED, len(samples))))


def method_tree_of_thoughts(samples, Q, qubo_var_indices, embeddings, clusters):
    cluster_reps = []
    for cluster in clusters:
        best = max(cluster, key=lambda i: samples[i].get("correctness_score", 0.5))
        cluster_reps.append(best)
    if len(cluster_reps) >= K_SELECTED:
        return cluster_reps[:K_SELECTED]
    used = set(cluster_reps)
    remaining = [i for i in range(len(samples)) if i not in used]
    remaining.sort(key=lambda i: samples[i].get("correctness_score", 0.0), reverse=True)
    cluster_reps.extend(remaining[:K_SELECTED - len(cluster_reps)])
    return cluster_reps[:K_SELECTED]


def method_ranked_voting(samples, Q, qubo_var_indices, clusters):
    votes = np.zeros(len(samples))
    for cluster in clusters:
        cluster_scores = [samples[i].get("correctness_score", 0.5) for i in cluster]
        total = sum(cluster_scores)
        for i, idx in enumerate(cluster):
            if total > 0:
                votes[idx] += cluster_scores[i] / total
    selected = np.argsort(votes)[::-1][:K_SELECTED]
    return [int(i) for i in selected]


def method_combinatorial(samples, Q, qubo_var_indices, embeddings):
    selected = []
    candidates = list(range(len(samples)))
    best_idx = max(candidates, key=lambda i: samples[i].get("correctness_score", 0.0))
    selected.append(best_idx)
    candidates.remove(best_idx)
    while len(selected) < K_SELECTED and candidates:
        best_score = -float("inf")
        best_candidate = None
        for c in candidates:
            score = samples[c].get("correctness_score", 0.0) * 0.3
            c_emb = embeddings[c]
            for s in selected:
                s_emb = embeddings[s]
                sim = float(np.dot(c_emb, s_emb))
                score -= sim * 0.7
            if score > best_score:
                best_score = score
                best_candidate = c
        if best_candidate is not None:
            selected.append(best_candidate)
            candidates.remove(best_candidate)
        else:
            break
    return selected[:K_SELECTED]


def method_qcr_llm(samples, Q, qubo_var_indices):
    scores = np.array([s.get("correctness_score", 0.0) for s in samples])
    selected = np.argsort(scores)[::-1][:K_SELECTED]
    return [int(i) for i in selected]


def method_dqo_bias(samples, Q, qubo_var_indices, solver):
    state, _ = solver.solve(Q)
    base_selected = set(qubo_var_indices[i] for i in range(len(state)) if state[i] == 1)
    np.random.seed(42)
    for i in range(len(state)):
        if np.random.random() < 0.15:
            state[i] = 1 - state[i]
    perturbed_selected = set(qubo_var_indices[i] for i in range(len(state)) if state[i] == 1)
    combined = list(base_selected | perturbed_selected)
    if len(combined) < K_SELECTED:
        available = [i for i in range(len(samples)) if i not in set(combined)]
        combined.extend(
            sorted(available, key=lambda i: samples[i].get("correctness_score", 0.0), reverse=True)
        )
    combined.sort(key=lambda i: samples[i].get("correctness_score", 0.0), reverse=True)
    return combined[:K_SELECTED]


METHODS = [
    ("BASE", "Textbook QUBO (yours)", method_base),
    ("[5]", "Self-Consistency", method_self_consistency),
    ("[6]", "Tree-of-Thoughts", method_tree_of_thoughts),
    ("[7]", "Ranked-Voting SC", method_ranked_voting),
    ("[8]", "Combinatorial Reasoning", method_combinatorial),
    ("[9]", "QCR-LLM (HUBO)", method_qcr_llm),
    ("[10]", "DQO bias-field", method_dqo_bias),
]


def main():
    print("Initializing pipeline components...")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(repo_root, "config", "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_name = cfg.get("model", {}).get("name", "unknown-model")
    print(f"Using SLM model: {model_name}")

    sampler = DiverseSampler()
    verifier = ReasonVerifier()
    qubo_builder = QUBOBuilder()
    solver = SimulatedAnnealingSolver()

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    outputs_dir = os.path.join(repo_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(outputs_dir, f"method_comparison_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_idx", "query", "method_code", "method_name",
            "k_selected", "selected_indices", "common_qubo_energy",
            "avg_redundancy", "avg_relevance", "runtime_s"
        ])

        for q_idx, question in enumerate(QUESTIONS, 1):
            print(f"\n{'='*60}")
            print(f"Query {q_idx}: {question[:60]}...")
            print(f"{'='*60}")

            print("  Sampling...")
            samples = sampler.sample(question)
            if not samples:
                print(f"  WARNING: No samples generated for query {q_idx}, skipping")
                continue
            print(f"  Got {len(samples)} samples")

            print("  Verifying...")
            samples = verifier.score_batch(samples, task_type="math")

            print("  Building QUBO...")
            Q, qubo_var_indices = qubo_builder.build_qubo(samples)
            n_vars = Q.shape[0]
            print(f"  QUBO matrix: {n_vars}x{n_vars}")

            print("  Computing embeddings...")
            all_embeddings, _ = build_embeddings(samples, embedder)

            clusters = qubo_builder._cluster_reasons(
                qubo_builder._embed_reasons([s["reason"] for s in samples]),
                list(range(len(samples)))
            )
            print(f"  Found {len(clusters)} clusters")

            for method_code, method_name, method_fn in METHODS:
                t_start = time.time()
                try:
                    if method_code == "BASE":
                        selected = method_fn(samples, Q, qubo_var_indices, solver)
                    elif method_code == "[6]":
                        selected = method_fn(samples, Q, qubo_var_indices, all_embeddings, clusters)
                    elif method_code == "[7]":
                        selected = method_fn(samples, Q, qubo_var_indices, clusters)
                    elif method_code == "[8]":
                        selected = method_fn(samples, Q, qubo_var_indices, all_embeddings)
                    elif method_code == "[10]":
                        selected = method_fn(samples, Q, qubo_var_indices, solver)
                    else:
                        selected = method_fn(samples, Q, qubo_var_indices)
                except Exception as e:
                    print(f"  WARNING: Method {method_code} failed: {e}")
                    continue

                runtime = time.time() - t_start

                if not selected:
                    selected = list(range(min(K_SELECTED, len(samples))))

                selected = selected[:K_SELECTED]
                while len(selected) < K_SELECTED and len(selected) < len(samples):
                    for i in range(len(samples)):
                        if i not in selected and len(selected) < K_SELECTED:
                            selected.append(i)

                energy = compute_qubo_energy(selected, Q, qubo_var_indices)
                redundancy = compute_redundancy(selected, all_embeddings)
                relevance = compute_relevance(selected, samples)
                selected_str = ";".join(str(i) for i in selected)

                writer.writerow([
                    q_idx, question, method_code, method_name,
                    len(selected), selected_str,
                    f"{energy:.4f}", f"{redundancy:.4f}", f"{relevance:.4f}", f"{runtime:.4f}"
                ])
                print(f"  [{method_code}] {method_name:<25s} energy={energy:.4f}  runtime={runtime:.4f}s")

    print(f"\nDone! CSV written to {csv_path}")
    print(f"Model used: {model_name}")


if __name__ == "__main__":
    main()
