"""Protocol tests use synthetic outputs; they are not benchmark accuracy evidence."""
import copy
import json
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.diagnostic_data import assigned_role, question_hash, select_examples
from evaluation.diagnostic_report import exact_mcnemar, holm_adjust, paired_inference, summarize, verified_record
from evaluation.diagnostic_runner import evaluate_example
from evaluation.learned_weights import fit_weights, load_weights
from evaluation.selection_protocol import N_TRACES, PROTOCOL_VERSION, select_methods, validate_pool
from evaluation.strict_answers import canonical_answer, grade
from pipeline.preference_labels import chain_matches_gold
from pipeline.qubo_math import exhaustive_solve, qubo_energy
from training.dpo_data import validate_pairs


WEIGHTS = dict(protocol=PROTOCOL_VERSION, intercept=-3.0,
               coefficients={"sc_score": 0.0, "nli_score": 0.0, "conf_score": 6.0, "arith_score": 0.0},
               penalty_weight=0.0, model_identity={"test": "synthetic"})


def example(index=0, role="eval", topic="test_task"):
    question = f"Synthetic test question {index}"
    return dict(question=question, question_hash=question_hash(question), source_id=f"synthetic:{index}",
                gold="1", answer_type="numeric", benchmark="gsm8k", role=role, topic=topic, topic_origin="synthetic")


def pool():
    samples = []
    for i in range(N_TRACES):
        correct = i >= 10
        answer = "1" if correct else "2"
        samples.append(dict(raw_text=f"Final answer: {answer}", canonical_answer=answer, answer=answer,
                            reason="synthetic", sc_score=6/16 if correct else 10/16,
                            nli_score=0.5, conf_score=float(correct), arith_score=0.0,
                            confidence_score=float(correct), input_tokens=1, output_tokens=1, token_limit_reached=False))
    return samples, [[float(i == j) for j in range(N_TRACES)] for i in range(N_TRACES)]


class FakeBackend:
    def pool(self, e, seed):
        assert "gold" not in e
        return pool()

    def synthesize(self, e, samples, indices):
        assert "gold" not in e
        return dict(raw_text=samples[min(indices)]["raw_text"], input_tokens=len(indices), output_tokens=1, token_limit_reached=False)


def record(index=0, seed=42):
    row = evaluate_example(FakeBackend(), example(index), seed, WEIGHTS)
    row["model_identity"] = WEIGHTS["model_identity"]
    return row


@pytest.mark.parametrize("raw,kind,expected", [
    ("A is possible but B is better", "choice", ""),
    ("Choice A is wrong\nFinal answer: (B)", "choice", "B"),
    ("Final answer: yes and no", "boolean", ""),
    ("Final answer: False", "boolean", "no"),
    ("Final answer: 0", "numeric", "0"),
    ("Calculating 12 + 30\n#### 1,200.00", "numeric", "1200"),
    ("Final answer: 1e3", "numeric", "1000"),
    ("42 is plausible; 99 was another estimate", "numeric", ""),
    ("Final answer: apple banana", "text", "apple banana"),
])
def test_strict_parser(raw, kind, expected):
    assert canonical_answer(raw, kind) == expected


def test_no_substring_or_percent_tolerance_in_labels():
    assert not chain_matches_gold({"answer": "Final answer: not yes"}, "yes", "commonsense")
    assert not chain_matches_gold({"answer": "999"}, "1000", "math")
    assert chain_matches_gold({"answer": "0", "reason": "Intermediate 7"}, "0", "math")


def test_shared_pool_exact_k_and_synthesis_control():
    row = record()
    assert not row.get("error")
    assert row["selection"]["indices"]["qubo"] == list(range(10, 16))
    assert row["selection"]["indices"]["greedy"] == list(range(10, 16))
    assert not row["methods"]["cot16"]["correct"]
    assert row["methods"]["qubo6"]["correct"]
    assert "cot16_synthesis" in row["methods"]
    for indices in row["selection"]["indices"].values():
        assert len(indices) == len(set(indices)) == 6


def test_fewer_than_six_in_an_answer_group_does_not_reduce_k():
    samples, sim = pool()
    for i, sample in enumerate(samples):
        sample["canonical_answer"] = str(i)
    selection = select_methods(samples, sim, WEIGHTS, 42)
    assert len(selection["indices"]["qubo"]) == 6


def test_reproducible_random_control_and_gold_independence():
    a = record()
    e = example()
    e["gold"] = "999"
    b = evaluate_example(FakeBackend(), e, 42, WEIGHTS)
    assert a["selection"] == b["selection"]
    assert a["methods"]["qubo6"]["raw_text"] == b["methods"]["qubo6"]["raw_text"]


def test_incomplete_pool_is_error_not_silent_fallback():
    class Broken(FakeBackend):
        def pool(self, e, seed):
            samples, sim = pool()
            return samples[:-1], sim
    row = evaluate_example(Broken(), example(), 42, WEIGHTS)
    assert "exactly 16" in row["error"]


def test_invalid_signal_and_similarity_rejected():
    samples, sim = pool()
    samples[0]["conf_score"] = float("nan")
    with pytest.raises(ValueError):
        validate_pool(samples, sim)
    samples, sim = pool()
    sim[0][1] = 0.5
    with pytest.raises(ValueError):
        validate_pool(samples, sim)


def test_exact_solver_matches_independent_enumeration():
    import itertools
    rng = random.Random(13)
    matrix = [[rng.uniform(-2, 2) for _ in range(7)] for _ in range(7)]
    state, energy = exhaustive_solve(matrix, k=3)
    reference = min(qubo_energy(matrix, [int(i in indices) for i in range(7)]) for indices in itertools.combinations(range(7), 3))
    assert sum(state) == 3
    assert energy == pytest.approx(reference)


def test_saved_prediction_or_selection_tampering_is_detected():
    row = record()
    row["methods"]["cot16"]["correct"] = True
    with pytest.raises(ValueError, match="correctness"):
        verified_record(row, WEIGHTS)
    row = record()
    row["selection"]["indices"]["qubo"] = [0, 1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="selections"):
        verified_record(row, WEIGHTS)


def test_paired_statistics_known_values_and_null():
    assert exact_mcnemar(10, 0) == pytest.approx(2 / 1024)
    assert exact_mcnemar(10, 10) == 1.0
    assert paired_inference([0.0] * 20)["p"] == 1.0
    assert paired_inference([1.0] * 10)["p"] == pytest.approx(2 / 1024)
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_seeds_clustered_by_question_and_missing_seed_rejected():
    rows = [record(0, s) for s in (42, 43, 44)]
    result = summarize(rows, WEIGHTS, [42, 43, 44])
    assert result["n_questions"] == 1
    assert all(c["evidence"] == "inconclusive" for c in result["comparisons"])
    with pytest.raises(ValueError, match="Missing"):
        summarize(rows[:-1], WEIGHTS, [42, 43, 44])


def test_null_no_advantage_and_errors_preserve_denominator():
    row = record()
    row["error"] = "synthetic execution failure"
    summary = summarize([row], WEIGHTS, [42])
    assert summary["errors"] == 1 and summary["n_questions"] == 1
    assert all(c["qubo_accuracy"] == 0 and c["evidence"] == "inconclusive" for c in summary["comparisons"])


def test_stratification_disjoint_partitions_and_deduplication():
    rows = [example(i, topic=str(i % 3)) for i in range(12)]
    chosen = select_examples(rows + [rows[0]], 7, 5)
    assert len(chosen) == 7 and len({r["question_hash"] for r in chosen}) == 7
    assert len({r["topic"] for r in chosen}) == 3
    for row in rows:
        assert assigned_role(row["question"], True) in {"fit", "tune", "eval"}


def calibration_records(role, start):
    records = []
    for i in range(20):
        samples, sim = pool()
        records.append(dict(example=example(start + i, role), samples=samples, similarity=sim,
                            model_identity=WEIGHTS["model_identity"], seed=42, protocol=PROTOCOL_VERSION))
    return records


def test_learn_weights_and_reject_test_leakage(tmp_path):
    fit, tune = calibration_records("fit", 0), calibration_records("tune", 100)
    weights = fit_weights(fit, tune, candidates=2)
    assert weights["coefficients"]["conf_score"] > 0
    assert weights["coefficients"] != WEIGHTS["coefficients"]
    path = tmp_path / "weights.json"
    path.write_text(json.dumps(weights))
    assert load_weights(path, WEIGHTS["model_identity"], [example(999)]) == weights
    with pytest.raises(ValueError, match="overlaps"):
        load_weights(path, WEIGHTS["model_identity"], [example(0)])
    tune[0]["example"]["role"] = "eval"
    with pytest.raises(ValueError):
        fit_weights(fit, tune, candidates=2)


def test_fit_tune_overlap_and_degenerate_learning_rejected():
    fit, tune = calibration_records("fit", 0), calibration_records("tune", 0)
    with pytest.raises(ValueError, match="overlap"):
        fit_weights(fit, tune, candidates=2)
    tune = calibration_records("tune", 100)
    for r in fit:
        for s in r["samples"]:
            s["raw_text"] = "Final answer: 1"
    with pytest.raises(ValueError, match="both correct and incorrect"):
        fit_weights(fit, tune, candidates=2)


def test_dpo_revalidates_completion_and_rejects_eval_pair(tmp_path):
    pair = dict(example(0, "fit"), prompt="training prompt", chosen="Final answer: 1", rejected="Final answer: 2",
                chosen_answer="1", rejected_answer="2")
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(pair) + "\n")
    assert len(validate_pairs(path)) == 1
    pair["chosen"] = "Final answer: 2"
    path.write_text(json.dumps(pair) + "\n")
    with pytest.raises(ValueError, match="Completion"):
        validate_pairs(path)
    pair["role"] = "eval"
    path.write_text(json.dumps(pair) + "\n")
    with pytest.raises(ValueError, match="never evaluation"):
        validate_pairs(path)


def test_loader_schemas_preserve_topics_and_arc_mapping(monkeypatch, tmp_path):
    import types
    from evaluation.diagnostic_data import load_pool
    requests = []
    def fake_load(dataset, name=None, split=None, **kwargs):
        requests.append((dataset, name, split))
        if dataset == "openai/gsm8k":
            return [{"question": "Toy math " + split, "answer": "Worked solution\n#### 0"}]
        if dataset == "cais/mmlu":
            return [{"question": "Toy subject " + split, "subject": "abstract_algebra", "choices": ["x", "y", "z", "w"], "answer": 2}]
        if dataset == "allenai/ai2_arc":
            return [{"question": "Toy science " + split, "id": "toy", "choices": {"label": ["1", "2", "3"], "text": ["x", "y", "z"]}, "answerKey": "2"}]
        if dataset == "voidful/StrategyQA":
            return [{"question": f"Toy boolean {i}", "answer": "false"} for i in range(30)]
        return [{"input": "Toy task (A) x (E) y " + str(i), "target": "(E)"} for i in range(30)]
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load))
    strategy_file = tmp_path / "strategyqa_train.json"
    strategy_file.write_text(json.dumps(fake_load("voidful/StrategyQA")))
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=lambda **kwargs: str(strategy_file)))
    assert grade(load_pool("gsm8k", "eval")[0]["gold"], load_pool("gsm8k", "eval")[0])["correct"]
    mmlu = load_pool("mmlu", "eval")
    assert mmlu[0]["topic"] == "abstract_algebra" and mmlu[0]["gold"] == "C"
    assert ("cais/mmlu", "all", "test") in requests
    arc = load_pool("arc_challenge", "eval")[0]
    assert arc["gold"] == "B" and arc["labels"] == ["A", "B", "C"]
    strategy = load_pool("strategyqa", "eval")
    assert strategy and all(r["gold"] == "no" for r in strategy)
    bbh = load_pool("bbh", "eval")
    assert len({r["topic"] for r in bbh}) == 27
    assert all(grade("Final answer: E", r)["correct"] for r in bbh)


def test_question_overlap_detection_ignores_legacy_wrappers():
    legacy = "Question: Toy science\n1. Mercury\n2. Venus\nAnswer:"
    new = "Toy science\nA. Mercury\nB. Venus"
    assert question_hash(legacy) == question_hash(new)


def test_official_test_duplicates_of_calibration_are_removed(monkeypatch):
    import types
    from evaluation.diagnostic_data import load_pool
    def fake_load(*args, split=None, **kwargs):
        rows = [dict(question="Duplicated official question", subject="physics", choices=["a", "b", "c", "d"], answer=0)]
        if split == "test":
            rows.append(dict(question="Unique test question", subject="physics", choices=["a", "b", "c", "d"], answer=1))
        return rows
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load))
    rows = load_pool("mmlu", "eval")
    assert len(rows) == 1
    assert rows[0]["calibration_source_duplicates_removed"] == 1


def test_cli_write_resume_and_offline_analysis(monkeypatch, tmp_path):
    import yaml
    import evaluation.diagnostic_runner as runner
    config = dict(model={"name": "synthetic", "use_vllm": False, "allow_device_fallback": False, "allow_quantization_fallback": False},
                  pipeline=dict(temperature_range=[0.3, 0.9], top_p=0.95, max_new_tokens=20, sampling_max_new_tokens=20),
                  verifier={"nli_model": "synthetic"})
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    identity = runner.model_identity(config, None)
    weights = dict(WEIGHTS, model_identity=identity, resolved_revisions={}, fit_question_hashes=[], tune_question_hashes=[], benchmark_coverage=["gsm8k"])
    weights_path = tmp_path / "weights.json"
    weights_path.write_text(json.dumps(weights))
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(json.dumps(example(1)) + "\n")
    directory = tmp_path / "output"
    class Backend(FakeBackend):
        revisions = {}
        def __init__(self, *args):
            pass
    monkeypatch.setattr(runner, "GenerationBackend", Backend)
    argv = ["test", "--config", str(config_path), "--manifest", str(manifest_path), "--weights", str(weights_path),
            "--output-dir", str(directory), "--seeds", "42", "43", "44"]
    monkeypatch.setattr(sys, "argv", argv)
    runner.main("gsm8k")
    assert (directory / "diagnosis.md").exists()
    saved = (directory / "results.jsonl").read_text()
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    runner.main("gsm8k")
    assert (directory / "results.jsonl").read_text() == saved
    monkeypatch.setattr(sys, "argv", ["test", "--analyze-only", "--output-dir", str(directory)])
    runner.main("gsm8k")
    # Changing seeds/config/model must not silently reuse old results.
    monkeypatch.setattr(sys, "argv", argv + ["--resume", "--questions", "1"])
    with pytest.raises(ValueError, match="provenance"):
        runner.main("gsm8k")
