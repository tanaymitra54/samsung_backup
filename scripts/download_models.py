#!/usr/bin/env python3
"""Prefetch the configured LLM, verifier and embedder into one model cache."""

import argparse
import os
import sys

import yaml
from huggingface_hub import snapshot_download

def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Download model weights to local cache")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--models", nargs="*", default=None, help="Override model list")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_dir = cfg.get("model", {}).get("cache_dir", "./cache/models")
    models = args.models or [cfg["model"]["name"], cfg["verifier"]["nli_model"], "sentence-transformers/all-MiniLM-L6-v2"]

    os.makedirs(cache_dir, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    failed = []
    for model_id in models:
        print(f"\n{'=' * 60}\nDownloading {model_id} -> {cache_dir}\n{'=' * 60}")
        try:
            path = snapshot_download(
                repo_id=model_id,
                cache_dir=cache_dir,
                token=token,
            )
            print(f"Done: {path}")
        except Exception as exc:
            print(f"FAILED: {model_id}: {exc}", file=sys.stderr)
            failed.append(model_id)

    if failed:
        print(f"\nFailed models: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)

    print("\nAll models downloaded successfully.")


if __name__ == "__main__":
    main()
