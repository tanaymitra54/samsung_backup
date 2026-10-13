"""One model replica per visible GPU; workers own separate durable JSONL files."""
from __future__ import annotations

import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def task_key(record):
    return record["example"]["question_hash"], record["seed"]


def read_records(path: Path, recover_tail=False):
    if not path.exists():
        return []
    data = path.read_bytes()
    rows, offset = [], 0
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            offset += len(line)
            continue
        try:
            rows.append(json.loads(line.decode("utf-8")))
        except (ValueError, UnicodeDecodeError):
            if not recover_tail or index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt complete record in {path}; manual inspection required")
            # Preserve interrupted bytes, then retry that incomplete task on resume.
            path.with_suffix(path.suffix + ".interrupted-tail").write_bytes(data[offset:])
            with path.open("r+b") as handle:
                handle.truncate(offset)
            break
        offset += len(line)
    # A valid record with no trailing newline needs a separator before append.
    if rows and data and not data.endswith(b"\n") and offset == len(data) and recover_tail:
        with path.open("ab") as handle:
            handle.write(b"\n")
    return rows


def recover_records(directory: Path, expected: set[tuple]) -> list[dict]:
    files = [directory / "results.jsonl"] + sorted((directory / "shards").glob("worker-*.jsonl"))
    records = {}
    for path in files:
        for row in read_records(path, recover_tail=True):
            key = task_key(row)
            if key not in expected:
                raise ValueError(f"Unexpected question/seed record in {path}")
            if key in records and row != records[key]:
                raise ValueError(f"Conflicting duplicate record in {path}")
            records[key] = row
    return list(records.values())


def append_record(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def worker_run(index, device, tasks, directory, adapter, weights, identity, code_sha, expected_revisions, backend_factory=None):
    from evaluation.diagnostic_runner import GenerationBackend, evaluate_example
    if device and device.startswith("cuda:"):
        import torch
        torch.cuda.set_device(int(device.split(":")[1]))
    factory = backend_factory or GenerationBackend
    directory = Path(directory)
    backend = factory(directory / "effective_config.yaml", device, adapter)
    if weights and weights.get("resolved_revisions") != backend.revisions:
        raise ValueError("Resolved model/precision differs from calibration; recollect and recalibrate")
    if expected_revisions is not None and expected_revisions != backend.revisions:
        raise ValueError("Model/precision changed during resume")
    path = directory / "shards" / f"worker-{index:03d}.jsonl"
    count = 0
    for example, seed in tasks:
        row = evaluate_example(backend, example, seed, weights)
        row.update(model_identity=identity, implementation_sha256=code_sha,
                   resolved_revisions=backend.revisions, worker_device=device, worker_index=index)
        append_record(path, row)
        count += 1
        print(f"[worker {index} {device}] {count}/{len(tasks)} {example['source_id']} seed={seed} {row.get('error', 'ok')}", flush=True)
    return dict(worker=index, device=device, records=count, revisions=backend.revisions)


def execute_tasks(tasks, devices, directory, adapter, weights, identity, code_sha, expected_revisions=None, backend_factory=None):
    # A question/seed is indivisible: all methods consume its same local trace pool.
    shards = [tasks[index::len(devices)] for index in range(len(devices))]
    arguments = [(index, device, shard, str(directory), adapter, weights, identity, code_sha, expected_revisions, backend_factory)
                 for index, (device, shard) in enumerate(zip(devices, shards)) if shard]
    if len(arguments) == 1:
        statuses = [worker_run(*arguments[0])]
    else:
        # CUDA processes must use spawn, never inherit an initialized CUDA context.
        with ProcessPoolExecutor(max_workers=len(arguments), mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(worker_run, *worker_args) for worker_args in arguments]
            statuses = [future.result() for future in as_completed(futures)]
    if statuses and any(status["revisions"] != statuses[0]["revisions"] for status in statuses):
        raise ValueError("Workers used different model revisions or precision")
    return sorted(statuses, key=lambda status: status["worker"])


def merge_records(directory: Path, examples, seeds):
    expected = {(e["question_hash"], seed) for e in examples for seed in seeds}
    records = recover_records(directory, expected)
    by_key = {task_key(r): r for r in records}
    if set(by_key) != expected:
        raise ValueError(f"Incomplete run: {len(by_key)}/{len(expected)} question/seed records")
    records = [by_key[(e["question_hash"], seed)] for e in examples for seed in seeds]
    temporary = directory / "results.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, directory / "results.jsonl")
    return records


def execution_devices(multi_gpu, no_multi_gpu, gpu_ids, device, config):
    if multi_gpu and no_multi_gpu:
        raise ValueError("Choose --multi-gpu or --no-multi-gpu")
    if device and (multi_gpu or gpu_ids):
        raise ValueError("--device pins one device; use --gpu-ids to select multiple visible devices")
    use_multi = multi_gpu or bool(gpu_ids) or (not no_multi_gpu and not device and config.get("hardware", {}).get("multi_gpu", False))
    preferred = device or config.get("evaluation", {}).get("device")
    if not use_multi:
        return [preferred]
    import torch
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count == 0:
        raise RuntimeError("Multi-GPU evaluation requires CUDA GPUs allocated to this process")
    ids = list(gpu_ids) if gpu_ids else list(range(count))
    if len(set(ids)) != len(ids) or any(index < 0 or index >= count for index in ids):
        raise ValueError(f"--gpu-ids are logical indices among {count} visible GPUs; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    if multi_gpu and len(ids) < 2:
        raise ValueError("--multi-gpu requested, but fewer than two GPUs are visible")
    return [f"cuda:{index}" for index in ids]
