# Running verified dataset diagnostics

Use the five independent entrypoints from the repository root:

| Dataset | Command | Topic metadata |
|---|---|---|
| GSM8K | `python scripts/evaluate_gsm8k.py` | Requires reviewed annotations for topic analysis |
| MMLU | `python scripts/evaluate_mmlu.py` | All subjects, preserved per question |
| BBH | `python scripts/evaluate_bbh.py` | All 27 tasks, preserved per question |
| ARC-Challenge | `python scripts/evaluate_arc.py` | Requires reviewed annotations for topic analysis |
| StrategyQA | `python scripts/evaluate_strategyqa.py` | Requires reviewed annotations for topic analysis |

Main-config evaluation defaults: 200 distinct questions, five seeds (42–46), 16 shared traces per question/seed, QUBO/Greedy K=6. Fit/tune default to 200 questions and one seed. Output budgets are 512 tokens per chain/synthesis. `--plan` and `--help` work without PyTorch. These runners do not train the language model or update weights using evaluation answers. See [the cluster guide](cluster/README.md) for setup, multi-GPU, Slurm and retraining. Legacy method definitions remain separate.

## 1. Install and check the environment

Use the repository GPU environment and requirements. For a read-only dataset check, run:

```sh
python scripts/check_diagnostic_datasets.py
python -m pytest -q tests/test_protocol.py tests/test_diagnostic_evaluation.py
python scripts/evaluate_gsm8k.py --plan
```

PyTorch, transformers, datasets, PEFT and sentence-transformers are needed for inference; DPO additionally needs supported modern TRL. `verification/dataset_loader_check.json`, when present, records actual dataset-only QA and never model performance.

## 2. Collect disjoint calibration pools

Example for GSM8K:

```sh
python scripts/evaluate_gsm8k.py --role fit --collect-only --questions 200 --seeds 42 --device cuda:0
python scripts/evaluate_gsm8k.py --role tune --collect-only --questions 200 --seeds 42 --device cuda:0
python scripts/learn_selection_weights.py --fit outputs/diagnostics/gsm8k/fit/results.jsonl --tune outputs/diagnostics/gsm8k/tune/results.jsonl --output config/gsm8k_learned_weights.json
```

Repeat with each dataset entrypoint to produce per-dataset artifacts. Alternatively pass multiple fit files and multiple tune files to learn one shared artifact spanning those benchmarks. Fit/tune must have at least 20 distinct questions each and both correct and incorrect fit traces. No low-scoring questions are dropped. If all fit traces are correct or all are wrong, gather more data or use a more capable model; the learner refuses an unidentifiable fit.

GSM8K/ARC use training data for calibration and official test for evaluation. MMLU uses validation for calibration and test for evaluation. Official test pools exclude any questions duplicated in the calibration source split; real MMLU validation/test duplicates were found during dataset QA. BBH partitions its public test split 20% fit / 20% tune / 60% reserved eval; StrategyQA explicitly loads its labeled training JSON and partitions it the same way. These are stable question-hash partitions. They do not turn the public splits into official unseen test sets.

For previously trained adapters, pass `--adapter PATH` during both calibration and evaluation and pass `--exclude-manifest PATH` for every known training set. The exclusion files must expose `question`; both plain training manifests and generated diagnostic preference files qualify. Prompt wrappers and option label differences are normalized for overlap detection. Model/adapter provenance is checked, but no script can reconstruct unknown historical training data. Do not claim unseen evaluation for an adapter with unknown training history.

## 3. Run evaluation

```sh
python scripts/evaluate_gsm8k.py --weights config/gsm8k_learned_weights.json --device cuda:0
python scripts/evaluate_gsm8k.py --weights config/gsm8k_learned_weights.json --device cuda:0 --resume
python scripts/evaluate_gsm8k.py --analyze-only
```

Use the corresponding artifact with each other dataset. `--output-dir` chooses a fresh run directory. `--resume` checks panel, seeds, model/adapter, code and weight provenance; changing these requires a fresh directory. Errors remain in the denominator and disable positive evidence labels. Failed runs exit nonzero. Token exhaustion is recorded; inputs are never silently truncated, and OOM never silently reduces generation budgets. Maximum output budgets remain configurable in YAML (512 by default); raise them **before calibration** if chains frequently hit the limit.

`--questions 200` supports an overall exploratory comparison but usually leaves too few MMLU subject/BBH task questions for an area-specific conclusion. Increase panel size or provide a focused, prespecified manifest with at least 30 distinct questions per topic and preferably more. `--full` evaluates the entire available reserved panel and can be expensive. Evidence thresholds do not guarantee adequate statistical power. Model runs are required before any better/worse claim is justified.

The methods are:

- `cot16`: majority answer from 16 CoT chains; first-trace tie break, no score or gold tie break.
- `qubo6`: exact minimum of the full 16-variable QUBO at K=6, followed by synthesis.
- `greedy6`: six largest learned scores from the same pool, followed by identical synthesis formatting.
- `random6`: seeded random six from the pool, followed by the same synthesis.
- `cot16_synthesis`: all 16 traces followed by one synthesis generation.
- `weighted_vote16`, `qubo6_vote`, `greedy6_vote`, `random6_vote`: controls with no synthesis generation.

The new QUBO intentionally bypasses legacy answer-group pruning to satisfy exactly-six selection from 16. Its objective is negative learned correctness score plus a learned within-answer similarity penalty; it is not a measurement of quantum hardware or annealing speed. The exhaustive solver checks all 8,008 six-of-sixteen subsets. At fixed K, a uniform diversity bonus cannot change the ranking and is omitted.

## 4. Inspect actual evidence

Each run stores `run.json`, `manifest.jsonl`, `effective_config.yaml`, `weights.json` and `results.jsonl`. Records include raw generated text, per-signal scores, temperatures, prompts, token counts, cosine similarities, selected indices, QUBO objective/energy, votes, synthesis outputs and errors. `--analyze-only` verifies the saved panel/provenance, replays selection and regrades outputs without loading the ML stack.

`diagnosis.md` / `diagnosis.json` give per-topic paired comparisons, confidence intervals, corrected p-values, per-seed wins/losses, and per-question correctness/failure categories. Small groups and inconsistent results are explicitly inconclusive. Topic analyses are exploratory and require confirmation on a fresh reserved panel after selecting improvement targets. `human_review.csv` includes raw outputs and the selected traces plus blank reviewer fields; fill those using independent reasoning/evidence checks. A correct final answer, NLI entailment of an answer from its own trace, or high model confidence cannot establish absence of hallucinations.

For reviewed topics, supply a JSONL manifest with fields such as:

```json
{"benchmark":"gsm8k","role":"eval","source_id":"reviewed:1","question":"...","gold":"...","answer_type":"numeric","topic":"reviewed topic name","topic_origin":"human_review"}
```

Choice questions also require `labels`; answer types are `numeric`, `choice`, `boolean`, `text`. The parser requires a final-answer line (or a standalone valid answer) and does not harvest incidental letters/numbers from reasoning. Keep `role` accurate and exclude training/calibration overlaps. No automated topic labels are fabricated.

## 5. Retrain on training examples from observed failure areas

Reproduce the identified difficulty on reserved **fit** examples; never export evaluation failures directly into training and then report accuracy on the same test panel.

```sh
python scripts/build_diagnostic_preferences.py --input outputs/diagnostics/gsm8k/fit/results.jsonl --output data/diagnostic_preference_pairs.jsonl
python training/run_dpo.py --pairs data/diagnostic_preference_pairs.jsonl --validate-only
python training/run_dpo.py --pairs data/diagnostic_preference_pairs.jsonl --sft-adapter checkpoints/YOUR_SFT_ADAPTER --max-steps 1 --output-dir checkpoints/dpo_smoke
```

Inspect the real smoke run for finite loss, reference/policy behavior and an adapter that reloads before launching full training (omit `--max-steps 1`, use a fresh output directory). Starting without `--sft-adapter` learns a fresh adapter against the base reference. After training, evaluate the emitted adapter with `--adapter`, recollect fit/tune pools for that adapter, refit coefficients, and compare on a fresh confirmation panel. The training provenance lists question hashes for exclusion.

The previous `scripts/generate_preference_pairs.py` format has no reserved-role provenance and is not accepted by the strict trainer. Regenerate using the diagnostic exporter; do not merely relabel legacy test-derived pairs as training data.

Use `--topics NAME ...` with the preference exporter to focus retraining on independently annotated fit examples from the diagnosed areas. Brier scores, invalid-answer rates, high-score wrong answers and output-budget exhaustion are reported as observable model/score health measures; they are not hallucination verdicts.
