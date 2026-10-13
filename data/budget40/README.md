# Budget-40 panel

Fixed panels for the matched-budget outcomes table and the architecture QLoRA loop.

| Manifest | Size | Role |
|---|---:|---|
| `eval_manifest.jsonl` | 40 | Test / held-out graded questions (8 × 5 benchmarks) |
| `train_manifest.jsonl` | 40 | Train-split prompts for QLoRA (disjoint source IDs) |

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
