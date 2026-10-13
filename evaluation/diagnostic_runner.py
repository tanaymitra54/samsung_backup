"""CLI and strict generation backend shared by the five benchmark entrypoints."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

from evaluation.diagnostic_data import BENCHMARKS, load_pool, question_hash, read_jsonl, select_examples
from evaluation.diagnostic_report import write_report
from evaluation.learned_weights import artifact_hash, load_weights
from evaluation.selection_protocol import PROTOCOL_VERSION, N_TRACES, select_methods, validate_pool
from evaluation.strict_answers import canonical_answer, grade
from evaluation.parallel_execution import execution_devices, execute_tasks, merge_records, recover_records, task_key

ROOT = Path(__file__).resolve().parents[1]


def _seed(base: int, digest: str) -> int:
    return int(hashlib.sha256(f"{base}:{digest}".encode()).hexdigest()[:8], 16)


def answer_instruction(example: dict) -> str:
    kind = example["answer_type"]
    constraint = {"numeric": "a number only", "boolean": "yes or no only", "text": "the exact target answer, without explanation",
                  "choice": "one option label from " + ", ".join(example.get("labels", []))}[kind]
    return f"End with a separate line: Final answer: <{constraint}>."


class GenerationBackend:
    """Reuse the repository model/verifier; do not silently shrink generation budgets."""
    def __init__(self, config_path, device, adapter):
        import torch
        from pipeline.inference import InferencePipeline
        from pipeline.sampling import DiverseSampler
        from pipeline.verifier import ReasonVerifier
        self.torch = torch
        self.inference = InferencePipeline(str(config_path), device=device, use_vllm=False, adapter_path=adapter)
        self.sampler = DiverseSampler(str(config_path), device=str(self.inference.device),
                                     shared_model=self.inference.model, shared_tokenizer=self.inference.tokenizer)
        self.verifier = ReasonVerifier(str(config_path), device=str(self.inference.device))
        self.config = self.inference.config
        self.revisions = {"model": getattr(self.inference.model.config, "_commit_hash", None),
                          "nli": getattr(self.verifier.nli_model.config, "_commit_hash", None),
                          "embedder": getattr(self.inference.embedder[0].auto_model.config, "_commit_hash", None),
                          "model_dtype": str(self.inference.model.dtype),
                          "quantization": str(getattr(self.inference.model.config, "quantization_config", None))}

    def set_seed(self, seed):
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)

    def generate(self, prompt, sample, temperature=None):
        tokenizer, model = self.inference.tokenizer, self.inference.model
        formatted = self.inference._apply_chat_template(prompt)
        inputs = tokenizer(formatted, return_tensors="pt", truncation=False, return_token_type_ids=False).to(self.inference.generation_input_device)
        input_tokens = int(inputs["input_ids"].shape[1])
        max_new = int(self.config["pipeline"]["sampling_max_new_tokens"] if sample else self.config["pipeline"]["max_new_tokens"])
        context = getattr(model.config, "max_position_embeddings", None)
        if context and input_tokens + max_new > context:
            raise ValueError(f"Context overflow: {input_tokens}+{max_new}>{context}; input was not truncated")
        kwargs = dict(max_new_tokens=max_new, do_sample=sample, return_dict_in_generate=True,
                      output_scores=sample, use_cache=True, pad_token_id=tokenizer.pad_token_id,
                      eos_token_id=tokenizer.eos_token_id)
        if sample:
            kwargs.update(temperature=temperature, top_p=self.config["pipeline"]["top_p"])
        with self.torch.inference_mode():
            output = model.generate(**inputs, **kwargs)
        ids = output.sequences[0, input_tokens:]
        raw = tokenizer.decode(ids, skip_special_tokens=True).strip()
        confidence = self.sampler._sequence_confidence(output, input_tokens) if sample else None
        eos = tokenizer.eos_token_id
        if isinstance(eos, int):
            eos = [eos]
        exhausted = len(ids) >= max_new and (not len(ids) or int(ids[-1]) not in (eos or []))
        return dict(raw_text=raw, input_tokens=input_tokens, output_tokens=int(len(ids)),
                    token_limit_reached=exhausted, confidence_score=confidence)

    def pool(self, example, seed):
        self.set_seed(_seed(seed, example["question_hash"]))
        samples = []
        for prompt in self.sampler.perturb_prompt(example["question"]):
            prompt += "\n" + answer_instruction(example)
            for _ in range(4):
                lo, hi = self.config["pipeline"]["temperature_range"]
                temperature = random.uniform(lo, hi)
                result = self.generate(prompt, True, temperature)
                answer = canonical_answer(result["raw_text"], example["answer_type"], example.get("labels"))
                samples.append(dict(result, reason=result["raw_text"], answer=answer, canonical_answer=answer,
                                    temperature=temperature, prompt_template=prompt))
        samples = self.verifier.score_batch(samples, task_type="math" if example["answer_type"] == "numeric" else "commonsense", gold=None)
        counts = Counter(s["canonical_answer"] for s in samples if s["canonical_answer"])
        for sample in samples:
            # Strict answers drive consistency. Incidental numbers in an invalid
            # reasoning trace must not become a self-consistency answer group.
            sample["sc_score"] = counts[sample["canonical_answer"]] / N_TRACES if sample["canonical_answer"] else 0.0
            sample.pop("correctness_score", None)  # Legacy hand-set fusion is not used in this protocol.
        embeddings = self.inference.embedder.encode([s["reason"] for s in samples], convert_to_numpy=True)
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity(embeddings).tolist()
        validate_pool(samples, similarity)
        return samples, similarity

    def synthesize(self, example, samples, indices):
        # Retain all six selected traces including answers; bypass legacy answer-group filtering.
        evidence = "\n\n".join(f"Trace {rank + 1}:\n{samples[i]['raw_text']}" for rank, i in enumerate(sorted(indices)))
        prompt = (f"Question:\n{example['question']}\n\nCandidate reasoning traces:\n{evidence}\n\n"
                  "Check the candidate reasoning and produce your own final answer.\n" + answer_instruction(example))
        return self.generate(prompt, False)


def evaluate_example(backend, example, seed, weights=None):
    started = time.perf_counter()
    row = dict(protocol=PROTOCOL_VERSION, example=example, seed=seed)
    try:
        generation_example = {k: v for k, v in example.items() if k not in {"gold", "reference_facts", "reference_decomposition"}}
        samples, similarity = backend.pool(generation_example, seed)
        validate_pool(samples, similarity)
        row.update(samples=samples, similarity=similarity)
        if weights is not None:
            selection = select_methods(samples, similarity, weights, seed)
            methods = {name: dict(raw_text=value, **grade(value, example)) for name, value in selection["votes"].items()}
            for name in ("qubo", "greedy", "random"):
                raw = backend.synthesize(generation_example, samples, selection["indices"][name])
                methods[name + "6"] = dict(raw, **grade(raw["raw_text"], example))
            raw = backend.synthesize(generation_example, samples, list(range(N_TRACES)))
            methods["cot16_synthesis"] = dict(raw, **grade(raw["raw_text"], example))
            row.update(selection=selection, methods=methods)
    except Exception as exc:
        # Preserve the denominator and the actual error; block confident conclusions for this run.
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["runtime_s"] = time.perf_counter() - started
    return row


def _adapter_digest(path):
    if not path:
        return None
    directory = Path(path).resolve()
    files = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix in {".json", ".safetensors", ".bin"})
    if not files:
        raise ValueError(f"No adapter files: {directory}")
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(directory)).encode())
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def model_identity(config, adapter):
    return dict(model=config["model"], adapter_sha256=_adapter_digest(adapter),
                generation={k: config["pipeline"][k] for k in ("temperature_range", "top_p", "max_new_tokens", "sampling_max_new_tokens")},
                nli_model=config["verifier"]["nli_model"], embedder="all-MiniLM-L6-v2", protocol=PROTOCOL_VERSION)


def main(benchmark: str):
    parser = argparse.ArgumentParser(description=f"Verified {benchmark}: CoT-16 vs Greedy top-six vs full-pool QUBO-six")
    parser.add_argument("--config", default=str(ROOT / "config/config.yaml"))
    parser.add_argument("--questions", type=int, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--sample-seed", type=int, default=None)
    parser.add_argument("--role", choices=["fit", "tune", "eval"], default="eval")
    parser.add_argument("--collect-only", action="store_true", help="Collect calibration trace features without final selection")
    parser.add_argument("--weights", help="Learned calibration JSON; required for evaluation")
    parser.add_argument("--manifest", help="Explicit JSONL with question, gold, answer_type, source_id, topic and role")
    parser.add_argument("--exclude-manifest", action="append", default=[], help="Exclude questions previously used for training (repeatable)")
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--multi-gpu", action="store_true", help="Spawn one full model replica per visible GPU; shard question/seed tasks")
    parser.add_argument("--no-multi-gpu", action="store_true", help="Disable automatic multi-GPU mode")
    parser.add_argument("--gpu-ids", type=int, nargs="+", help="Logical indices within the scheduler's visible GPU allocation, e.g. 0 1")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Show protocol and settings without datasets/models")
    parser.add_argument("--analyze-only", action="store_true", help="Reverify and report an existing completed run without ML dependencies")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("Evaluation runs one dispatcher with spawned GPU workers; do not launch it through torchrun or multiple srun tasks")
    if args.multi_gpu and args.no_multi_gpu:
        parser.error("Choose --multi-gpu or --no-multi-gpu")
    if args.device and (args.multi_gpu or args.gpu_ids):
        parser.error("--device pins one device; --gpu-ids selects logical GPU workers")
    import yaml
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    diagnostic_config = config.get("diagnostics", {})
    calibration = args.role in {"fit", "tune"}
    if args.questions is None:
        args.questions = diagnostic_config.get("calibration_questions" if calibration else "questions", 200)
    if args.seeds is None:
        args.seeds = diagnostic_config.get("calibration_seeds" if calibration else "seeds", [42] if calibration else [42, 43, 44, 45, 46])
    if args.sample_seed is None:
        args.sample_seed = diagnostic_config.get("sample_seed", 20261013)
    if args.questions <= 0 or len(args.seeds) != len(set(args.seeds)):
        parser.error("questions must be positive and seeds must be unique")
    if args.role != "eval" and not args.collect_only:
        parser.error("fit/tune runs must use --collect-only")
    plan = dict(benchmark=benchmark, role=args.role, questions="all" if args.full else args.questions,
                seeds=args.seeds, traces_per_question=N_TRACES, qubo_k=6, greedy_k=6,
                cot="majority_vote_of_same_16_CoT_chains", topic_source="dataset metadata or explicit annotations",
                scoring="learned on fit; penalty chosen on tune; frozen on eval", protocol=PROTOCOL_VERSION,
                extra_generations="QUBO/Greedy/random each have one six-trace synthesis; CoT-16 synthesis has one all-trace synthesis")
    if args.plan:
        plan["execution"] = dict(multi_gpu=args.multi_gpu or bool(args.gpu_ids) or (not args.no_multi_gpu and not args.device and config.get("hardware", {}).get("multi_gpu", False)),
                                 no_multi_gpu=args.no_multi_gpu, device=args.device, gpu_ids=args.gpu_ids)
        print(json.dumps(plan, indent=2))
        return
    base_output = Path(diagnostic_config.get("outputs_dir", str(ROOT / "outputs/diagnostics")))
    directory = Path(args.output_dir or base_output / benchmark / args.role).resolve()
    if args.analyze_only:
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        weights = json.loads((directory / "weights.json").read_text(encoding="utf-8"))
        records = read_jsonl(directory / "results.jsonl")
        if len(records) != run["expected_records"]:
            raise ValueError("Incomplete run")
        if artifact_hash(weights) != run["weights_sha256"] or weights["model_identity"] != run["model_identity"]:
            raise ValueError("Saved weights disagree with run provenance")
        manifest = read_jsonl(directory / "manifest.jsonl")
        if artifact_hash({"examples": manifest}) != run["panel_sha256"]:
            raise ValueError("Saved panel disagrees with provenance")
        expected = {(e["question_hash"], seed) for e in manifest for seed in run["seeds"]}
        if {(r["example"]["question_hash"], r["seed"]) for r in records} != expected:
            raise ValueError("Record identities do not match the planned panel")
        write_report(directory, records, weights, run["seeds"])
        print(f"Verified report: {directory / 'diagnosis.md'}")
        return
    if not args.collect_only and not args.weights:
        parser.error("--weights is required; collect fit/tune pools and run scripts/learn_selection_weights.py first")
    config = copy.deepcopy(config)
    config["pipeline"].update(num_answers=4, num_reasons=4, subset_size=6)
    config["model"]["use_vllm"] = False
    config["model"]["allow_device_fallback"] = False
    config["model"]["allow_quantization_fallback"] = False
    rows = read_jsonl(args.manifest) if args.manifest else load_pool(benchmark, args.role, args.dataset_revision)
    for row in rows:
        if row.get("benchmark", benchmark) != benchmark or row.get("role") != args.role:
            raise ValueError("Manifest mixes benchmarks or omits/mismatches role")
        row["benchmark"] = benchmark
        row.setdefault("topic", "unlabelled")
        row.setdefault("topic_origin", "provided_annotation" if row["topic"] != "unlabelled" else "unavailable")
        grade(str(row["gold"]), row)
    excluded = {question_hash(r["question"]) for path in args.exclude_manifest for r in read_jsonl(path)}
    examples = select_examples(rows, None if args.full else args.questions, args.sample_seed, excluded)
    if not examples:
        raise ValueError("Empty evaluation/calibration panel")
    identity = model_identity(config, args.adapter)
    weights = load_weights(args.weights, identity, examples) if not args.collect_only else None
    source_files = [Path(__file__), ROOT / "evaluation/strict_answers.py", ROOT / "evaluation/selection_protocol.py", ROOT / "pipeline/verifier.py", ROOT / "pipeline/sampling.py", ROOT / "pipeline/inference.py"]
    source_files += [ROOT / "evaluation/diagnostic_data.py"]
    source_files += [ROOT / "evaluation/parallel_execution.py"]
    code_sha = artifact_hash({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    if weights and weights.get("implementation_sha256") and weights["implementation_sha256"] != code_sha:
        raise ValueError("Generation/scoring implementation changed since calibration; recollect and recalibrate")
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        git_head = None
    versions = {}
    for package in ("torch", "transformers", "datasets", "peft", "sentence-transformers", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    run = dict(plan, model_identity=identity, panel_sha256=artifact_hash({"examples": examples}),
               code_sha256=code_sha, git_head=git_head, versions=versions, python=platform.python_version(),
               expected_records=len(examples) * len(args.seeds), actual_questions=len(examples),
               effective_config_sha256=artifact_hash(config),
               weights_sha256=artifact_hash(weights) if weights else None, sample_seed=args.sample_seed,
               confidence_note="Geometric mean under the temperature/top-p sampling distribution; not proof of factual truth.")
    directory.mkdir(parents=True, exist_ok=True)
    run_path, results_path = directory / "run.json", directory / "results.jsonl"
    if run_path.exists() or results_path.exists():
        if not args.resume:
            raise ValueError("Output exists; use --resume or a fresh --output-dir")
        if json.loads(run_path.read_text(encoding="utf-8")) != run:
            raise ValueError("Resume provenance differs: configuration/model/weights/panel/code/seeds changed")
        if yaml.safe_load((directory / "effective_config.yaml").read_text(encoding="utf-8")) != config:
            raise ValueError("Saved effective config differs from resume configuration")
    else:
        run_path.write_text(json.dumps(run, indent=2), encoding="utf-8")
        (directory / "effective_config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
        (directory / "manifest.jsonl").write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in examples), encoding="utf-8")
        if weights:
            (directory / "weights.json").write_text(json.dumps(weights, indent=2), encoding="utf-8")
    expected = {(e["question_hash"], seed) for e in examples for seed in args.seeds}
    records = recover_records(directory, expected)
    done = {task_key(r) for r in records}
    pending = [(e, seed) for e in examples for seed in args.seeds if (e["question_hash"], seed) not in done]
    if pending:
        devices = execution_devices(args.multi_gpu, args.no_multi_gpu, args.gpu_ids, args.device, config)
        expected_revisions = records[0].get("resolved_revisions") if records else None
        if any(r.get("resolved_revisions") != expected_revisions for r in records):
            raise ValueError("Resumed records mix model revisions/precision")
        statuses = execute_tasks(pending, devices, directory, args.adapter, weights, identity, code_sha,
                                 expected_revisions, backend_factory=GenerationBackend)
        (directory / "execution.json").write_text(json.dumps(dict(cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            worker_status=statuses, tasks_this_invocation=len(pending)), indent=2), encoding="utf-8")
    records = merge_records(directory, examples, args.seeds)
    if weights:
        write_report(directory, records, weights, args.seeds)
        print(f"Verified report: {directory / 'diagnosis.md'}")
    else:
        print(f"Calibration pools: {results_path}")
    if any(r.get("error") for r in records):
        raise SystemExit("Run contains execution errors; inspect results.jsonl and rerun in a fresh directory after fixing them")
