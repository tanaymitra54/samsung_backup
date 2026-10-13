"""Fail early on incompatible packages, CUDA allocation, or diagnostic config."""
import argparse
import importlib
import importlib.metadata
import inspect
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config/config.yaml"))
    parser.add_argument("--cpu", action="store_true", help="Only check APIs/packages; not a CUDA qualification")
    parser.add_argument("--min-gpus", type=int, default=1)
    parser.add_argument("--output", default="verification/cluster_preflight.json")
    args = parser.parse_args()
    import torch
    import yaml
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
    report = dict(python=platform.python_version(), torch=torch.__version__, cuda_runtime=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(), scope="CPU API check" if args.cpu else "CUDA allocation and API check", packages={}, gpus=[])
    required = {"beta", "max_length", "model_adapter_name", "ref_adapter_name"}
    if not required <= set(inspect.signature(DPOConfig).parameters) or "processing_class" not in inspect.signature(DPOTrainer).parameters:
        raise RuntimeError("DPO API incompatible with the trainer")
    if "max_length" not in inspect.signature(SFTConfig).parameters or "processing_class" not in inspect.signature(SFTTrainer).parameters:
        raise RuntimeError("SFT API incompatible with the trainer")
    for package, module in (("transformers", "transformers"), ("trl", "trl"), ("peft", "peft"),
                            ("datasets", "datasets"), ("accelerate", "accelerate"), ("sentence-transformers", "sentence_transformers")):
        importlib.import_module(module)
        report["packages"][package] = importlib.metadata.version(package)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    pipeline = config["pipeline"]
    if (pipeline["num_answers"], pipeline["num_reasons"], pipeline["subset_size"]) != (4, 4, 6):
        raise ValueError("Diagnostic config must specify 4 templates x 4 chains and K=6")
    if pipeline["sampling_max_new_tokens"] <= 0 or pipeline["max_new_tokens"] <= 0:
        raise ValueError("Generation budgets must be positive")
    if not args.cpu:
        if not torch.cuda.is_available() or torch.cuda.device_count() < args.min_gpus:
            raise RuntimeError(f"Need at least {args.min_gpus} allocated CUDA GPUs")
        import bitsandbytes
        report["packages"]["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index):
                value = torch.randn(64, 64, device=f"cuda:{index}", dtype=torch.bfloat16)
                if not torch.isfinite(value @ value.T).all().item():
                    raise RuntimeError("Non-finite CUDA BF16 operation")
                # Exercise the actual NF4 kernel used by QLoRA, including backward.
                layer = bitsandbytes.nn.Linear4bit(64, 32, bias=False, compute_dtype=torch.bfloat16,
                                                   compress_statistics=True, quant_type="nf4").to(f"cuda:{index}")
                inputs = torch.randn(2, 64, device=f"cuda:{index}", dtype=torch.bfloat16, requires_grad=True)
                outputs = layer(inputs)
                outputs.float().square().mean().backward()
                if not torch.isfinite(outputs).all().item() or not torch.isfinite(inputs.grad).all().item():
                    raise RuntimeError("Non-finite bitsandbytes NF4 forward/backward")
                free, total = torch.cuda.mem_get_info(index)
                report["gpus"].append(dict(logical_index=index, name=torch.cuda.get_device_name(index),
                                          free_gib=free / 2**30, total_gib=total / 2**30,
                                          nf4_forward_backward=True,
                                          bf16_supported=torch.cuda.is_bf16_supported()))
                del layer, inputs, outputs, value
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
