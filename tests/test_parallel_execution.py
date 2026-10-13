"""Durable result collection and GPU allocation tests, without fabricated accuracy."""
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.parallel_execution import append_record, execution_devices, merge_records, read_records, recover_records


def row(index, seed=42):
    return dict(example=dict(question_hash=str(index)), seed=seed, raw_text="synthetic transport fixture")


def test_durable_merge_order_and_duplicate_identical_copy(tmp_path):
    append_record(tmp_path / "shards/worker-001.jsonl", row(1))
    append_record(tmp_path / "shards/worker-000.jsonl", row(0))
    expected = {(str(i), 42) for i in range(2)}
    assert len(recover_records(tmp_path, expected)) == 2
    examples = [dict(question_hash="0"), dict(question_hash="1")]
    result = merge_records(tmp_path, examples, [42])
    assert [r["example"]["question_hash"] for r in result] == ["0", "1"]
    assert len(recover_records(tmp_path, expected)) == 2


def test_partial_last_write_is_preserved_and_retried(tmp_path):
    path = tmp_path / "shards/worker-000.jsonl"
    append_record(path, row(0))
    with path.open("ab") as handle:
        handle.write(b'{"interrupted":')
    assert read_records(path, recover_tail=True) == [row(0)]
    assert path.with_suffix(".jsonl.interrupted-tail").read_bytes() == b'{"interrupted":'
    append_record(path, row(1))
    assert read_records(path) == [row(0), row(1)]


def test_complete_corrupt_line_and_conflicting_duplicate_fail(tmp_path):
    path = tmp_path / "shards/worker-000.jsonl"
    path.parent.mkdir()
    path.write_bytes(b'{bad}\n')
    with pytest.raises(ValueError, match="Corrupt"):
        read_records(path, recover_tail=True)
    path.write_text(json.dumps(row(0)) + "\n")
    append_record(tmp_path / "shards/worker-001.jsonl", dict(row(0), raw_text="changed"))
    with pytest.raises(ValueError, match="Conflicting"):
        recover_records(tmp_path, {("0", 42)})


def test_incomplete_merge_never_publishes_success(tmp_path):
    append_record(tmp_path / "shards/worker-000.jsonl", row(0))
    with pytest.raises(ValueError, match="Incomplete"):
        merge_records(tmp_path, [dict(question_hash="0"), dict(question_hash="1")], [42])
    assert not (tmp_path / "results.jsonl").exists()


def test_gpu_ids_are_scheduler_logical_indices(monkeypatch):
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2))
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    assert execution_devices(True, False, [0, 1], None, {}) == ["cuda:0", "cuda:1"]
    with pytest.raises(ValueError, match="logical indices"):
        execution_devices(True, False, [3, 7], None, {})
    with pytest.raises(ValueError, match="pins one"):
        execution_devices(True, False, None, "cuda:0", {})
    with pytest.raises(ValueError):
        execution_devices(True, False, [0, 0], None, {})


def test_config_auto_single_and_explicit_modes(monkeypatch):
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, device_count=lambda: 2))
    monkeypatch.setitem(sys.modules, "torch", fake)
    config = dict(hardware=dict(multi_gpu=True), evaluation=dict(device="cuda:0"))
    assert execution_devices(False, False, None, None, config) == ["cuda:0", "cuda:1"]
    assert execution_devices(False, True, None, None, config) == ["cuda:0"]
    assert execution_devices(False, False, None, "cpu", config) == ["cpu"]


def test_combined_report_corrects_across_datasets(tmp_path):
    from scripts.run_diagnostics import combine_reports
    for benchmark, p in (("gsm8k", 0.03), ("mmlu", 0.04)):
        directory = tmp_path / benchmark / "eval"
        directory.mkdir(parents=True)
        comparison = dict(p=p, holm_p=p, evidence="repeatable_advantage_on_this_panel", group_type="overall",
                          group="all", comparator="cot16", n_questions=100, delta=0.1)
        (directory / "diagnosis.json").write_text(json.dumps(dict(comparisons=[comparison])))
    report = combine_reports(tmp_path, ["gsm8k", "mmlu"])
    assert all(c["holm_p"] == pytest.approx(0.06) for c in report["comparisons"])
    assert all(c["evidence"] == "inconclusive" for c in report["comparisons"])
