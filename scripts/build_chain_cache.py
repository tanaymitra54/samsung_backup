"""
==============================================================================
FILE: scripts/build_chain_cache.py
ROLE: Generate a scored chain pool for ANY dataset, not just GSM8K
==============================================================================

WHY THIS EXISTS
---------------
The GSM8K chain pool showed the PRM works (AUC 0.815, recoverable-set
separation 12/19 vs 3/19 for the old signal) but that no selection method can
DEMONSTRATE a win there:

    plain majority vote ..... 90.3%
    best aggregation rule ... 92.0%   (+1.7, inside the +/-3.4 noise band)
    ORACLE .................. 96.7%   (+6.4 = only ~2 SE)

Total headroom is 19 questions out of 300. Even flawless selection is barely
two standard errors from baseline, and resolving the observed +1.7 points at
80% power would need ~4,750 questions -- GSM8K's test split has 1,319. The
benchmark is saturated for this model: the ceiling sits too close to the floor
for any selection method to prove itself, however good it is.

Harder benchmarks have far more headroom, so a real improvement has room to
show up as a real number. This script builds the same cached-pool artefact for
those, letting prm_experiment.py and the QUBO tuner run unchanged against them.

WHAT IT PRODUCES
----------------
The same JSON shape tune_qubo_params.py's pre-cache emits, plus the questions
and golds inline:

    {"chains": [[chain, ...], ...],       # per question, each verifier-scored
     "questions": [...], "golds": [...],  # so consumers need not re-derive them
     "excluded_indices": [...], "dataset": "math500"}

Storing golds inline matters: the GSM8K cache did not, so every consumer had to
re-load the dataset and re-derive them in the same order, which silently breaks
the moment a dataset filters or reorders differently.

NUMERIC-ANSWER FILTER
---------------------
MATH/AIME golds are frequently symbolic ("\\frac{\\pi}{2}", "2\\sqrt{3}").
Comparing those correctly needs a CAS-grade equivalence checker; comparing them
naively silently marks correct answers wrong and poisons every downstream
number. Until such a checker is wired in, this keeps only plain-numeric-answer
problems, which is still a much harder and less saturated set than GSM8K. The
count kept vs dropped is always reported so the restriction is never invisible.

USAGE
-----
    python scripts/build_chain_cache.py --dataset math500 --n 300
    python scripts/build_chain_cache.py --dataset gsm8k --n 300 --split test
    python scripts/build_chain_cache.py --dataset aime --n 90
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import warnings
warnings.filterwarnings("ignore")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


_NUMERIC = re.compile(r"^-?\d+(?:\.\d+)?$")


def _clean(ans: str) -> str:
    """Strip LaTeX wrappers and separators that hide an otherwise plain number."""
    s = str(ans).strip()
    s = re.sub(r"^\\boxed\{(.*)\}$", r"\1", s.strip())
    s = s.replace("\\!", "").replace("\\,", "").replace("$", "").replace(",", "")
    s = s.replace("\\%", "").replace("%", "")
    return s.strip()


# ── Dataset adapters ─────────────────────────────────────────────────────────
#
# Each returns (questions, golds) with golds as floats. Adding a dataset means
# adding one function here; nothing downstream changes.

def load_gsm8k(n: int, split: str = "test"):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=split)
    qs, gs = [], []
    for item in ds:
        if "####" not in item["answer"]:
            continue
        tail = _clean(item["answer"].split("####")[-1])
        m = re.findall(r"-?\d+(?:\.\d+)?", tail)
        if not m:
            continue
        qs.append(item["question"])
        gs.append(float(m[-1]))
        if len(qs) >= n:
            break
    return qs, gs, 0


def load_math500(n: int, split: str = "test"):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split=split)
    qs, gs, dropped = [], [], 0
    for item in ds:
        raw = _clean(item.get("answer", ""))
        if not _NUMERIC.fullmatch(raw):
            dropped += 1
            continue
        qs.append(item["problem"])
        gs.append(float(raw))
        if len(qs) >= n:
            break
    return qs, gs, dropped


def load_aime(n: int, split: str = "train"):
    """AIME answers are integers 0-999 by construction, so none are dropped."""
    from datasets import load_dataset
    last_err = None
    for repo in ("HuggingFaceH4/aime_2024", "Maxwell-Jia/AIME_2024"):
        try:
            ds = load_dataset(repo, split=split)
            break
        except Exception as e:      # try the next mirror
            last_err = e
    else:
        raise RuntimeError(f"could not load an AIME dataset: {last_err}")

    qs, gs, dropped = [], [], 0
    for item in ds:
        problem = item.get("problem") or item.get("Problem") or ""
        raw = _clean(item.get("answer") or item.get("Answer") or "")
        if not _NUMERIC.fullmatch(raw):
            dropped += 1
            continue
        qs.append(problem)
        gs.append(float(raw))
        if len(qs) >= n:
            break
    return qs, gs, dropped


_LOADERS = {"gsm8k": load_gsm8k, "math500": load_math500, "aime": load_aime}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(_LOADERS),
                    help="Which benchmark to build a pool for.")
    ap.add_argument("--n", type=int, default=300, help="Questions to sample.")
    ap.add_argument("--split", default=None, help="Override the dataset's default split.")
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--qubo-params", default="config/best_qubo_params.yaml")
    ap.add_argument("--out", default=None,
                    help="Output path (default: results/cached_chains_<dataset>_<n>q.json)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--adapter-path", default=None,
                    help="LoRA adapter to merge before sampling, to build a pool "
                         "from a fine-tuned model rather than the base one.")
    ap.add_argument("--oracle-scoring", action="store_true",
                    help="ABLATION ONLY: let the verifier see gold while scoring. "
                         "Leaks the answer key into chain quality; the default "
                         "(gold-free) is what deployment actually sees.")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else Path(
        f"results/cached_chains_{args.dataset}_{args.n}q.json"
    )

    loader = _LOADERS[args.dataset]
    split_kw = {"split": args.split} if args.split else {}
    print(f"[data] loading {args.dataset} (n={args.n}) ...", flush=True)
    questions, golds, dropped = loader(args.n, **split_kw)
    print(f"[data] {len(questions)} questions with plain-numeric answers")
    if dropped:
        print(f"[data] {dropped} dropped for symbolic/non-numeric answers "
              "(see the NUMERIC-ANSWER FILTER note in this file's docstring)")
    if not questions:
        print("[error] no usable questions; nothing to build.", file=sys.stderr)
        sys.exit(1)

    # ── Pipeline components ──────────────────────────────────────────────────
    import yaml
    from pipeline.sampling import DiverseSampler
    from pipeline.verifier import ReasonVerifier
    from pipeline.inference import InferencePipeline

    cfg_path = str(_PROJECT_ROOT / args.config)
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("[init] loading model ...", flush=True)
    inf = InferencePipeline(config_path=cfg_path, use_vllm=False,
                            device=args.device, adapter_path=args.adapter_path)
    sampler = DiverseSampler(config_path=cfg_path, device=args.device,
                             shared_model=inf.model, shared_tokenizer=inf.tokenizer)
    verifier = ReasonVerifier(config_path=cfg_path, device=args.device)
    print("[init] ready.", flush=True)

    pool = cfg["pipeline"]["num_answers"] * 4
    print(f"[gen] sampling {pool} chains per question, scoring "
          f"{'WITH ORACLE GOLD' if args.oracle_scoring else 'gold-free'} ...", flush=True)

    all_chains = []
    t0 = time.time()
    for qi, (q, gold) in enumerate(zip(questions, golds)):
        chains = sampler.sample(q, task_type="math")
        scoring_gold = str(gold) if args.oracle_scoring else None
        verifier.score_batch(chains, task_type="math", gold=scoring_gold, question=q)
        all_chains.append(chains)

        if (qi + 1) % 10 == 0:
            el = time.time() - t0
            eta = (len(questions) - qi - 1) / ((qi + 1) / el)
            print(f"[gen] {qi+1}/{len(questions)}  ({el:.0f}s elapsed, ETA {eta:.0f}s)",
                  flush=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": args.dataset,
            "n": len(questions),
            "scoring": "oracle" if args.oracle_scoring else "gold-free",
            "adapter": args.adapter_path,
            "questions": questions,
            "golds": golds,
            "chains": all_chains,
            "excluded_indices": [],
        }, f)

    el = time.time() - t0
    print(f"\n[done] {len(questions)} questions x {pool} chains in {el/60:.1f} min")
    print(f"[done] -> {out_path}")
    print(f"\nNext: score it with a PRM and analyse headroom:")
    print(f"    python scripts/prm_experiment.py --chains {out_path} \\")
    print(f"        --prm-cache results/prm_{args.dataset}_{args.n}q.json")


if __name__ == "__main__":
    main()
