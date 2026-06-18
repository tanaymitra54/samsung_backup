#!/usr/bin/env python3
"""
Setup Verification Script for QUBO Pipeline Benchmarking
Verifies all components are configured correctly before running benchmarks
"""

import os
import sys
import yaml
import torch
from pathlib import Path

def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

def check_config():
    print_section("1. Configuration Check")
    
    config_path = "config/config.yaml"
    if not Path(config_path).exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Model check
    model_name = config["model"]["name"]
    print(f"✅ Model: {model_name}")
    
    if "Qwen3.5-4B" not in model_name:
        print(f"⚠️  Warning: Expected Qwen3.5-4B, got {model_name}")
    
    # Device check
    device = config["evaluation"]["device"]
    print(f"✅ Target Device: {device}")
    
    # Pipeline params
    print(f"✅ Samples per question: {config['pipeline']['num_answers']} × {config['pipeline']['num_reasons']} = {config['pipeline']['num_answers'] * config['pipeline']['num_reasons']}")
    print(f"✅ Temperature range: {config['pipeline']['temperature_range']}")
    print(f"✅ Max new tokens: {config['pipeline']['max_new_tokens']}")
    
    # QUBO params
    print(f"✅ QUBO max vars: {config['qubo']['max_vars']}")
    print(f"✅ Penalty weight: {config['qubo']['penalty_weight']}")
    
    # Solver params
    print(f"✅ Solver GPU enabled: {config['solver']['gpu']['enabled']}")
    print(f"✅ Parallel reads: {config['solver']['gpu']['num_parallel_reads']}")
    
    # Benchmarks
    benchmarks = config["evaluation"]["benchmarks"]
    print(f"✅ Benchmarks configured: {', '.join(benchmarks)}")
    print(f"✅ Subset size: {config['evaluation']['subset_size']}")
    print(f"✅ Batch size: {config['evaluation']['batch_size']}")
    
    return True

def check_gpu():
    print_section("2. GPU Availability Check")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"✅ CUDA available: {torch.version.cuda}")
    print(f"✅ Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        mem_total = props.total_memory / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_free = mem_total - mem_reserved
        
        print(f"\n  GPU {i}: {props.name}")
        print(f"    Total memory: {mem_total:.2f} GB")
        print(f"    Free memory: {mem_free:.2f} GB")
        print(f"    Currently allocated: {mem_allocated:.2f} GB")
        print(f"    Currently reserved: {mem_reserved:.2f} GB")
        
        if i == 1:
            if mem_free > 10:
                print(f"    ✅ GPU 1 has sufficient free memory for Qwen3.5-4B")
            else:
                print(f"    ⚠️  GPU 1 may have insufficient memory (< 10GB free)")
    
    return True

def check_dependencies():
    print_section("3. Dependencies Check")
    
    required = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("datasets", "HuggingFace Datasets"),
        ("sentence_transformers", "Sentence Transformers"),
        ("sklearn", "scikit-learn"),
        ("yaml", "PyYAML"),
        ("numpy", "NumPy"),
        ("tqdm", "tqdm"),
    ]
    
    all_ok = True
    for module, name in required:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} not installed!")
            all_ok = False
    
    # Optional dependencies
    optional = [
        ("vllm", "vLLM (optional for faster inference)"),
        ("wandb", "Weights & Biases (optional for tracking)"),
        ("bitsandbytes", "BitsAndBytes (optional for 4-bit quantization)"),
    ]
    
    print("\nOptional dependencies:")
    for module, name in optional:
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} - not installed")
    
    return all_ok

def check_model_cache():
    print_section("4. Model Cache Check")
    
    cache_dir = Path("cache/models")
    if not cache_dir.exists():
        print(f"⚠️  Cache directory does not exist: {cache_dir}")
        print("   Models will be downloaded on first run")
        return True
    
    # Check for Qwen models
    qwen_dirs = list(cache_dir.glob("*Qwen*"))
    if qwen_dirs:
        print(f"✅ Found {len(qwen_dirs)} Qwen model(s) in cache:")
        for d in qwen_dirs:
            print(f"   - {d.name}")
    else:
        print("⚠️  No Qwen models in cache - will download on first run")
    
    return True

def check_output_dir():
    print_section("5. Output Directory Check")
    
    output_dir = Path("outputs")
    if not output_dir.exists():
        print(f"⚠️  Output directory does not exist, creating: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Output directory: {output_dir.absolute()}")
    
    # List existing results
    existing = list(output_dir.glob("*.json"))
    if existing:
        print(f"✅ Found {len(existing)} existing result file(s)")
        # Show most recent
        if existing:
            latest = max(existing, key=lambda p: p.stat().st_mtime)
            print(f"   Most recent: {latest.name}")
    else:
        print("   (No existing results)")
    
    return True

def check_pipeline_modules():
    print_section("6. Pipeline Modules Check")
    
    sys.path.insert(0, str(Path.cwd()))
    
    modules = [
        "pipeline.sampling",
        "pipeline.verifier",
        "pipeline.qubo_builder",
        "pipeline.solver",
        "pipeline.inference",
        "evaluation",
    ]
    
    all_ok = True
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except Exception as e:
            print(f"❌ {module} - {e}")
            all_ok = False
    
    return all_ok

def estimate_runtime():
    print_section("7. Runtime Estimation")
    
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    
    subset_size = config["evaluation"]["subset_size"]
    num_samples = config["pipeline"]["num_answers"] * config["pipeline"]["num_reasons"]
    benchmarks = config["evaluation"]["benchmarks"]
    
    # Rough estimates (seconds per question)
    time_per_question = {
        "sampling": num_samples * 0.5,  # 0.5s per sample
        "verification": 0.3,
        "qubo_building": 0.2,
        "solving": 0.1,
        "final_inference": 0.5,
    }
    
    total_time_per_q = sum(time_per_question.values())
    
    print(f"Estimated time per question: ~{total_time_per_q:.1f} seconds")
    print(f"\nBreakdown:")
    for stage, time in time_per_question.items():
        print(f"  - {stage}: {time:.1f}s")
    
    # Total estimate
    num_benchmarks = len([b for b in benchmarks if b != "strategyqa"])  # Exclude unavailable
    total_questions = subset_size * num_benchmarks
    total_minutes = (total_time_per_q * total_questions) / 60
    
    print(f"\nTotal for {num_benchmarks} benchmark(s) × {subset_size} questions:")
    print(f"  Estimated time: ~{total_minutes:.0f} minutes ({total_minutes/60:.1f} hours)")
    print(f"  (Actual time may vary based on model speed and GPU load)")

def recommend_commands():
    print_section("8. Recommended Commands")
    
    print("\n# Quick test (10 samples, 2 benchmarks):")
    print("python scripts/run_all_benchmarks.py \\")
    print("  --device cuda:1 \\")
    print("  --subset-size 10 \\")
    print("  --benchmarks gsm8k mmlu \\")
    print("  --seed 42")
    
    print("\n# Medium test (50 samples, all benchmarks except strategyqa):")
    print("python scripts/run_all_benchmarks.py \\")
    print("  --device cuda:1 \\")
    print("  --subset-size 50 \\")
    print("  --benchmarks gsm8k mmlu arc_challenge bbh \\")
    print("  --seed 42")
    
    print("\n# Full benchmark (200 samples as configured):")
    print("python scripts/run_all_benchmarks.py \\")
    print("  --device cuda:1 \\")
    print("  --subset-size 200 \\")
    print("  --benchmarks gsm8k mmlu arc_challenge bbh \\")
    print("  --seed 42 \\")
    print("  --output-dir outputs/qwen35_4b_full")
    
    print("\n# Monitor GPU usage during run:")
    print("watch -n 1 nvidia-smi")

def main():
    print("\n" + "="*70)
    print("  QUBO Pipeline Benchmarking - Setup Verification")
    print("="*70)
    
    checks = [
        ("Configuration", check_config),
        ("GPU", check_gpu),
        ("Dependencies", check_dependencies),
        ("Model Cache", check_model_cache),
        ("Output Directory", check_output_dir),
        ("Pipeline Modules", check_pipeline_modules),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n❌ Error in {name} check: {e}")
            results[name] = False
    
    # Additional info sections (don't affect pass/fail)
    estimate_runtime()
    recommend_commands()
    
    # Summary
    print_section("Summary")
    
    all_passed = all(results.values())
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {name}")
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Ready to run benchmarks!")
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues before running benchmarks")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
