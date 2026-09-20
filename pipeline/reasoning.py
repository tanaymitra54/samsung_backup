from pipeline.answer_groups import keep_one_answer_group, pick_winning_group
from pipeline.selection import select_best_reasoning


def run_reasoning_pipeline(
    sampler,
    verifier,
    qubo_builder,
    solver,
    inference,
    question: str,
    task_type: str = "math",
    generate_answer: bool = True,
) -> dict | None:
    """Sample, score, select one answer group with exact-K QUBO, then prompt.

    The verifier score does not receive the gold answer. Set generate_answer
    when the current model should also produce the final response.
    """
    samples = sampler.sample(question, task_type=task_type)
    if not samples:
        return None

    samples = verifier.score_batch(samples, task_type=task_type)
    group_idx = pick_winning_group(samples)
    group_samples = [samples[i] for i in group_idx]
    Q, qubo_var_indices = qubo_builder.build_qubo(group_samples)
    k = min(inference.subset_size, len(group_samples))
    state, energy = solver.solve(Q, k=k)
    scores = [s.get("correctness_score", 0.0) for s in group_samples]
    local_selected = select_best_reasoning(
        state, qubo_var_indices, len(group_samples), k, scores=scores
    )
    selected_indices = keep_one_answer_group(
        samples, [group_idx[i] for i in local_selected if i < len(group_idx)]
    )
    final_prompt, ordered_reasons = inference.prepare_final_prompt(
        question, selected_indices, samples
    )
    answer = inference.generate_answer(final_prompt) if generate_answer else ""
    return {
        "selected_traces": ordered_reasons,
        "final_prompt": final_prompt,
        "predicted_answer": answer,
        "answer": answer,
        "energy": float(energy),
        "qubo_energy": float(energy),
        "selected_indices": selected_indices,
    }
