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
    """Sample, score, select with QUBO, then build the final prompt.

    The verifier score does not receive the gold answer. Set generate_answer
    when the current model should also produce the final response.
    """
    samples = sampler.sample(question)
    if not samples:
        return None

    samples = verifier.score_batch(samples, task_type=task_type)
    Q, qubo_var_indices = qubo_builder.build_qubo(samples)
    state, energy = solver.solve(Q)
    if state is None:
        selected_indices = []
    else:
        selected_indices = [
            qubo_var_indices[i] for i, bit in enumerate(state) if int(bit) == 1
        ]
    if not selected_indices:
        selected_indices = list(range(min(inference.subset_size, len(samples))))

    composed = inference.compose(
        question, selected_indices, samples, generate=generate_answer
    )
    composed["energy"] = float(energy)
    composed["selected_indices"] = selected_indices
    return composed
