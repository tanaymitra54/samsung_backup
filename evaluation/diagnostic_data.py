"""Metadata-preserving loaders and deterministic, disjoint calibration partitions."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path

BENCHMARKS = ("gsm8k", "mmlu", "bbh", "arc_challenge", "strategyqa")
BBH_TASKS = (
    "boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa",
    "dyck_languages", "formal_fallacies", "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects", "logical_deduction_three_objects",
    "movie_recommendation", "multistep_arithmetic_two", "navigate", "object_counting",
    "penguins_in_a_table", "reasoning_about_colored_objects", "ruin_names",
    "salient_translation_error_detection", "snarks", "sports_understanding", "temporal_sequences",
    "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects", "web_of_lies", "word_sorting",
)


def question_hash(question: str) -> str:
    # Ignore the prompt wrappers and option label spelling used by legacy runners.
    # Keep all option text so distinct MCQs are not collapsed by their stem alone.
    text = re.sub(r"^\s*Question:\s*", "", question, flags=re.I)
    text = re.sub(r"\s*\nAnswer:\s*$", "", text, flags=re.I)
    text = re.sub(r"\n\s*(?:[A-Z]|[0-9]+)[.)]\s*", "\n<option> ", text)
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def assigned_role(question: str, shared_eval_split: bool) -> str:
    bucket = int(question_hash(question)[:8], 16) % 100
    if shared_eval_split:
        return "fit" if bucket < 20 else "tune" if bucket < 40 else "eval"
    return "fit" if bucket < 80 else "tune"


def _row(benchmark, dataset, split, source_id, question, gold, kind, topic="unlabelled", labels=None):
    return dict(benchmark=benchmark, dataset=dataset, source_split=split,
                source_id=str(source_id), question=question, question_hash=question_hash(question),
                gold=str(gold), answer_type=kind, labels=labels or [], topic=topic,
                topic_origin="dataset" if topic != "unlabelled" else "unavailable")


def _mcq(question, labels, texts):
    return question + "\n" + "\n".join(f"{a}. {b}" for a, b in zip(labels, texts))


def load_pool(benchmark: str, role: str, revision: str | None = None) -> list[dict]:
    if role not in {"fit", "tune", "eval"}:
        raise ValueError(role)
    from datasets import load_dataset
    try:
        dataset_version = importlib.metadata.version("datasets")
    except importlib.metadata.PackageNotFoundError:
        dataset_version = "fixture"
    cache_base = Path(os.environ.get("PRISM_DATASETS_CACHE") or os.environ.get("HF_HOME") or Path(__file__).resolve().parents[1] / "cache/huggingface")
    # Arrow feature schemas from datasets 4.x are not readable by 3.x.
    # Preserve other users' caches and isolate prepared data by library version.
    kwargs = {"cache_dir": str(cache_base / f"prism-datasets-{dataset_version}")}
    if revision:
        kwargs["revision"] = revision
    shared = benchmark in {"bbh", "strategyqa"}
    split = "test" if role == "eval" else "train"
    rows = []
    if benchmark == "gsm8k":
        for i, x in enumerate(load_dataset("openai/gsm8k", "main", split=split, **kwargs)):
            rows.append(_row(benchmark, "openai/gsm8k", split, f"gsm8k:{split}:{i}", x["question"], x["answer"], "numeric"))
    elif benchmark == "mmlu":
        split = "test" if role == "eval" else "validation"
        for i, x in enumerate(load_dataset("cais/mmlu", "all", split=split, **kwargs)):
            labels = list("ABCD")
            rows.append(_row(benchmark, "cais/mmlu", split, f"mmlu:{split}:{i}",
                             _mcq(x["question"], labels, x["choices"]), labels[int(x["answer"])],
                             "choice", x["subject"], labels))
    elif benchmark == "arc_challenge":
        for i, x in enumerate(load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split, **kwargs)):
            # Canonicalize numeric ARC option labels to letters using their positions.
            old_labels = [str(a) for a in x["choices"]["label"]]
            labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(old_labels)])
            gold = labels[old_labels.index(str(x["answerKey"]))]
            rows.append(_row(benchmark, "allenai/ai2_arc", split, f"arc:{split}:{x.get('id', i)}",
                             _mcq(x["question"], labels, x["choices"]["text"]), gold, "choice", labels=labels))
    elif benchmark == "strategyqa":
        split = "train"  # Labeled public split; reserve by question hash, not an official test claim.
        # This repository mixes unlabeled test, labeled train, filtered train and
        # paragraph JSON schemas. Loading the repository generically is ambiguous.
        from huggingface_hub import hf_hub_download
        source_path = Path(hf_hub_download(repo_id="voidful/StrategyQA", repo_type="dataset",
                                          filename="strategyqa_train.json", revision=revision))
        source_rows = json.loads(source_path.read_text(encoding="utf-8"))
        for i, x in enumerate(source_rows):
            value = x["answer"]
            if isinstance(value, str):
                if value.lower() not in {"true", "false", "yes", "no"}:
                    raise ValueError(f"Unexpected StrategyQA label: {value}")
                value = value.lower() in {"true", "yes"}
            rows.append(_row(benchmark, "voidful/StrategyQA", split, f"strategyqa:{x.get('qid', i)}", x["question"], "yes" if value else "no", "boolean"))
            rows[-1]["dataset_resolved_revision"] = source_path.parent.name
            rows[-1]["reference_facts"] = x.get("facts", [])
            rows[-1]["reference_decomposition"] = x.get("decomposition", [])
    elif benchmark == "bbh":
        split = "test"
        for task in BBH_TASKS:
            # Failure is explicit: never silently drop an entire task.
            for i, x in enumerate(load_dataset("lukaemon/bbh", task, split=split, **kwargs)):
                gold = str(x["target"]).strip()
                if gold.startswith("(") and gold.endswith(")") and len(gold) == 3:
                    kind = "choice"
                elif gold.lower() in {"yes", "no", "true", "false"}:
                    kind = "boolean"
                elif task in {"multistep_arithmetic_two", "object_counting"}:
                    kind = "numeric"
                else:
                    kind = "text"
                import re
                labels = re.findall(r"\(([A-Z])\)", x["input"]) if kind == "choice" else []
                rows.append(_row(benchmark, "lukaemon/bbh", split, f"bbh:{task}:{i}", x["input"], gold, kind, task, sorted(set(labels))))
    else:
        raise ValueError(benchmark)
    if shared or role != "eval":
        rows = [r for r in rows if assigned_role(r["question"], shared) == role]
    removed = 0
    if role == "eval" and not shared:
        # Official split names do not guarantee distinct questions (MMLU
        # contains real validation/test duplicates). Exclude all calibration
        # source questions, not just those sampled for a particular weight fit.
        calibration_rows = load_pool(benchmark, "fit", revision) + load_pool(benchmark, "tune", revision)
        calibration_hashes = {r["question_hash"] for r in calibration_rows}
        before = len(rows)
        rows = [r for r in rows if r["question_hash"] not in calibration_hashes]
        removed = before - len(rows)
    for row in rows:
        row["role"] = role
        row["dataset_revision_requested"] = revision
        row["calibration_source_duplicates_removed"] = removed
        row["split_policy"] = "question_hash_partition_of_public_split" if shared else "official_test_or_partitioned_training_split"
    return rows


def select_examples(rows: list[dict], limit: int | None, seed: int, excluded: set[str] | None = None) -> list[dict]:
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        row = dict(row)
        digest = question_hash(row["question"])
        if digest in (excluded or set()) or digest in seen:
            continue
        seen.add(digest)
        row["question_hash"] = digest
        groups[row.get("topic", "unlabelled")].append(row)
    rng = random.Random(seed)
    topics = sorted(groups)
    rng.shuffle(topics)
    for pool in groups.values():
        rng.shuffle(pool)
    selected = []
    # Round-robin over true metadata strata, with exactly min(limit, pool size) rows.
    while any(groups.values()) and (limit is None or len(selected) < limit):
        for topic in topics:
            if groups[topic] and (limit is None or len(selected) < limit):
                selected.append(groups[topic].pop())
    return selected
