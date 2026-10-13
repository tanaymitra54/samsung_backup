# Verification scope

Checks were performed on 13 October 2026. These artifacts describe executable behavior and data integrity, not pretrained benchmark performance.

| Artifact / check | Scope |
|---|---|
| `pytorch_runtime_report.json` | 58 passing tests, zero skipped, using actual PyTorch 2.5.1 CPU on Windows/Python 3.11 with the pinned core stack |
| `pytorch_runtime_junit.xml` | Raw test results for protocol/recovery/statistics plus tiny-Qwen generation, real SFT/DPO updates, reference freeze, adapter reload and two spawned CPU workers |
| `cpu_preflight.json` | Actual package imports and DPO/SFT API/config compatibility; CUDA qualification explicitly absent |
| `dataset_loader_check.json` | Real five-dataset loading, all gold-answer parsing, metadata/counts and zero fit/tune/eval question overlap after exclusions |
| Shell / Python / docs / Compose | Bash syntax, Python compilation, local document links, Git whitespace and Compose v2.29.2 configuration validation passed |

The runtime checks exposed NumPy/PyArrow compatibility and generic-tokenizer input-field issues; these were corrected. Dataset checking also exposed incompatible prepared caches from different `datasets` versions; the loader now isolates those caches without deleting other environments' data.

No full Qwen benchmark run, production learned coefficient artifact, A6000 CUDA/NF4 kernel run, NCCL distributed DPO run, Docker image build or post-training accuracy gain is claimed. Follow [the cluster guide](../docs/cluster/README.md) to perform the real GPU preflight, generation/sharding smoke and one-step QLoRA/DPO smoke before full jobs. The GPU preflight exercises BF16 and actual NF4 forward/backward on each allocated device.

The small real-model tests use randomly initialized models. Their generated answers and losses are functional-test evidence only; they do not identify domains where pretrained QUBO improves over CoT.
