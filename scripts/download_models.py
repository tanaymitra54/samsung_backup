#!/usr/bin/env python3
"""
==============================================================================
FILE: scripts/download_models.py
ROLE: Offline Model Weights & Tokenizer Downloader Utility
==============================================================================

Pre-fetch every model the pipeline loads at runtime, so a compute node with no
outbound internet can run the whole thing offline.

THREE MODELS ARE NEEDED, not one. Earlier versions fetched only the LLM, which
fails on an air-gapped compute node the moment the verifier or the QUBO builder
initialises:

  1. The SLM itself            -- config model.name
  2. The NLI cross-encoder     -- config verifier.nli_model, loaded by
                                  ReasonVerifier for language-task scoring
  3. all-MiniLM-L6-v2          -- loaded by QUBOBuilder (chain embeddings for the
                                  similarity matrix) and by InferencePipeline
                                  (relevance re-ranking)

Usage
-----
    # Everything needed for a run, main model only (recommended)
    python scripts/download_models.py --main-only

    # A specific model
    python scripts/download_models.py --models Qwen/Qwen2.5-3B-Instruct

    # Every candidate in config.yaml as well
    python scripts/download_models.py

After this, set HF_HUB_OFFLINE=1 on the compute node to guarantee nothing
reaches for the network.
"""

import argparse
import os
import sys
import time

import yaml
from huggingface_hub import snapshot_download

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",   # architecture-mandated base SLM
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

# Loaded by SentenceTransformer inside QUBOBuilder and InferencePipeline.
EMBEDDER_ID = "sentence-transformers/all-MiniLM-L6-v2"


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _fmt_size(path: str) -> str:
    total = 0
    for root, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except OSError:
                pass
    return f"{total / 1e9:.2f} GB" if total >= 1e9 else f"{total / 1e6:.0f} MB"


def fetch(model_id: str, cache_dir: str, token: str | None) -> bool:
    print(f"\n{'=' * 68}")
    print(f"  Downloading  {model_id}")
    print(f"  Destination  {cache_dir}")
    print(f"{'=' * 68}", flush=True)
    t0 = time.time()
    try:
        path = snapshot_download(repo_id=model_id, cache_dir=cache_dir, token=token)
        print(f"  [ok] {model_id}  ({_fmt_size(path)} in {time.time() - t0:.0f}s)", flush=True)
        print(f"       {path}", flush=True)
        return True
    except Exception as exc:
        print(f"  [FAIL] {model_id}: {exc}", file=sys.stderr, flush=True)
        return False


def verify(model_id: str, cache_dir: str) -> bool:
    """Confirm the cached copy actually loads, rather than just existing on disk."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
        print(f"  [verify] {model_id:<45} loads OK ({cfg.model_type})", flush=True)
        return True
    except Exception as exc:
        print(f"  [verify] {model_id:<45} FAILED: {str(exc)[:60]}", file=sys.stderr, flush=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download every model the pipeline needs")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--models", nargs="*", default=None, help="Override model list")
    parser.add_argument("--main-only", action="store_true",
                        help="Fetch only config model.name, not the whole candidates list")
    parser.add_argument("--skip-support", action="store_true",
                        help="Skip the NLI verifier and MiniLM embedder (not recommended)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg.get("model", {})
    cache_dir = model_cfg.get("cache_dir", "./cache/models")

    if args.models:
        models = args.models
    elif args.main_only:
        models = [model_cfg.get("name")]
    else:
        models = model_cfg.get("candidates") or DEFAULT_MODELS

    # Support models the pipeline loads implicitly.
    support = []
    if not args.skip_support:
        nli = cfg.get("verifier", {}).get("nli_model")
        if nli:
            support.append(nli)
        support.append(EMBEDDER_ID)

    os.makedirs(cache_dir, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print("=" * 68)
    print("  MODEL DOWNLOAD")
    print("=" * 68)
    print(f"  cache_dir : {os.path.abspath(cache_dir)}")
    print(f"  HF token  : {'set' if token else 'not set (public models only)'}")
    print(f"  LLMs      : {', '.join(m for m in models if m)}")
    print(f"  support   : {', '.join(support) if support else 'skipped'}")

    failed = []
    for model_id in [m for m in models if m] + support:
        if not fetch(model_id, cache_dir, token):
            failed.append(model_id)

    # ── Verification pass ────────────────────────────────────────────────────
    print(f"\n{'=' * 68}")
    print("  VERIFYING CACHED COPIES")
    print(f"{'=' * 68}")
    for model_id in [m for m in models if m] + [s for s in support if s != EMBEDDER_ID]:
        if model_id not in failed:
            verify(model_id, cache_dir)

    if not args.skip_support:
        try:
            from sentence_transformers import SentenceTransformer
            SentenceTransformer("all-MiniLM-L6-v2")
            print(f"  [verify] {EMBEDDER_ID:<45} loads OK", flush=True)
        except Exception as exc:
            print(f"  [verify] {EMBEDDER_ID:<45} FAILED: {str(exc)[:60]}",
                  file=sys.stderr, flush=True)
            failed.append(EMBEDDER_ID)

    print(f"\n{'=' * 68}")
    if failed:
        print(f"  FAILED: {', '.join(failed)}", file=sys.stderr)
        print("=" * 68)
        sys.exit(1)

    print("  All models downloaded and verified.")
    print("  On an offline compute node, export HF_HUB_OFFLINE=1 before running.")
    print("=" * 68)


if __name__ == "__main__":
    main()
