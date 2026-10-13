# Budget-40 panel

Legacy fixed panels for the architecture QLoRA loop. These are functional checks, not the new CoT-16 / Greedy-top-six diagnostic evaluation. Use [the cluster guide](../../docs/cluster/README.md) for the maintained workflow.

| Manifest | Size | Role |
|---|---:|---|
| `eval_manifest.jsonl` | 40 | Test / held-out graded questions (8 × 5 benchmarks) |
| `train_manifest.jsonl` | 40 | Reserved QLoRA prompts; includes eight public BBH test and eight MMLU validation questions |

Seed: 42.

HF sources:
- GSM8K: `openai/gsm8k` (eval=test, train=train)
- ARC-Challenge: `allenai/ai2_arc` / `ARC-Challenge` (eval=test, train=train)
- MMLU: `cais/mmlu` STEM subjects (eval=test, train=validation or test-disjoint)
- BBH: `lukaemon/bbh` (test only; eval and train are disjoint IDs)
- StrategyQA: `voidful/StrategyQA` (train only; eval and train are disjoint IDs)

Regenerate with:

```bash
python scripts/curate_budget40.py
```
