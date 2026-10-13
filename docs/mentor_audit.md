# Mentor audit — 13 October 2026

Audited ecodesamsung `main` at `50913fa`, after a fresh successful pull. Findings below describe that commit before this change. This is a source-code and manifest audit, not a new model-accuracy experiment. Existing historical CSVs do not establish the requested current 16-chain comparison.

## 1. Was DPO properly implemented?

**No: the supplied training path was incomplete and unreliable.**

- `scripts/generate_preference_pairs.py` writes `data/preference_pairs.jsonl` with `prompt/chosen/rejected` fields. `training/run_dpo.py` instead reads two unrelated JSON arrays under `training_data/positive_traces.json` and `negative_traces.json`. These components are not connected.
- The trainer passes plain `TrainingArguments` plus `beta/max_length/max_prompt_length` directly to `DPOTrainer`. Modern TRL expects DPO-specific settings in `DPOConfig`; the existing dependency `trl>=0.8.0` does not constrain the incompatible API. Changing only `tokenizer` to `processing_class` is insufficient. See the [official TRL API](https://huggingface.co/docs/trl/v0.19.0/en/dpo_trainer).
- It always loads the base model and creates a new adapter. It has no input for continuing the saved SFT adapter despite its “run after SFT” description.
- Preference checks used substring matching for non-math answers and 1% numeric tolerance. A wrong MCQ label appearing in prose could pass; `0` was also mishandled by a truthiness fallback. These are not reliable correct/incorrect labels.
- The architecture runner invokes SFT/QLoRA; it never invokes DPO automatically.

The new trainer consumes validated JSONL fit pairs, uses `DPOConfig`, supports `--sft-adapter`, and loads a frozen copy of that SFT adapter as its reference. It uses the model chat template. The new pair exporter rechecks actual completions and rejects evaluation/tune questions and unparsable negatives. **Data validation is tested; actual GPU optimization and post-DPO gains still require a real training/evaluation run.** Final-answer correctness does not prove the chosen reasoning is factually sound.

## 2. Were selection and scoring weights learned dynamically?

**No.** `ReasonVerifier` reads fixed SC/NLI/confidence/arithmetic coefficients from YAML (math 0.4/0.3/0.2/0.1; commonsense 0.5/0.3/0.2/0). Trace signal values change per question; the coefficients do not. `QUBOBuilder` likewise loads fixed coefficients. `HyperparameterQUBO` explicitly identifies itself as an unused stub.

The old tuning script searches a manually specified GSM8K train grid but never learns verifier coefficients. It excludes questions where all self-scores are below 0.2 from the denominator. Its first training questions can overlap SFT training questions. At exact K, `diversity_bonus` adds the same K times a constant to every solution and cannot change the selection; `cardinality_penalty` is bypassed. The builder does not use `answer_agree_weight` in its pair costs. Much of this grid therefore cannot identify the advertised effects.

The new diagnostic runner requires a learned artifact: four nonnegative logistic scoring coefficients plus an intercept fitted on reserved fit questions, and a redundancy penalty selected on disjoint tune questions. The penalty is chosen using **selected-answer vote accuracy**, a documented proxy, not claimed optimal end-to-end synthesis accuracy. No artifact or learned values are fabricated. After retraining, recollect and recalibrate for the new adapter. **Weights remain frozen on evaluation questions**; adapting to test labels would invalidate the comparison. Legacy runners still use their existing YAML weights.

## 3. Which benchmarks and how many questions did existing runners use?

| Entry point | Evaluation / training | Actual scope at default settings |
|---|---|---|
| `scripts/run_all_benchmarks.py`, main config | Evaluation | GSM8K test first 200; ARC-Challenge test first 200; StrategyQA labeled train first 200; MMLU only five STEM subjects, 40 each; BBH 27 tasks, seven each = 189 if every task loads |
| Same runner, fast config | Evaluation | GSM8K/ARC/StrategyQA 50 each; MMLU five subjects, ten each = 50; BBH one per task = 27 if every task loads |
| `scripts/run_architecture_pipeline.py` | Base eval → SFT/QLoRA → eval | Existing manifests contain exactly 40 evaluation and 40 training questions, eight per benchmark. Default CLI is one training round. DPO is not part of the loop |
| `training/sft.py` | Iterative SFT/QLoRA | Main config: GSM8K 200 train and 200 test per round, three rounds. CLI flags can override these. Its non-GSM8K `_load_split` delegates to evaluation loaders and does not honor the requested training split correctly |
| `scripts/generate_training_data.py` | SFT trace generation | Source caps: GSM8K train 2,000; MMLU validation 800; ARC train 800; StrategyQA train 800; LogiQA train 600 = at most 5,000 before filtering. `TOTAL_TARGET=6000` does not guarantee 6,000 rows. Final split is 90/10; OpenOrca is disabled |
| `scripts/tune_qubo_params.py` | Parameter tuning | Default 300 GSM8K train questions, with low-self-score questions excluded; not a clean evaluation |
| `scripts/prepare_training_data.py` | Legacy SFT preparation | Targets 20,000 using OpenMathInstruct-2 plus traces from evaluation CSVs. Its own header warns that this leaks evaluation items |

Manifest counts were read directly. The existing evaluation manifest has 32 `test` rows and eight StrategyQA `train` rows. The training manifest has 24 `train`, eight BBH `test`, and eight MMLU `validation` rows. BBH/StrategyQA need reserved disjoint partitions because the public labeled data does not offer the same official train/test setup as GSM8K.

Counts above describe the code/configuration, not a claim that a job finished with that many rows. BBH loader exceptions were silently swallowed. MMLU subsetting also behaves unexpectedly when the requested total is below five.

## 4. Were the algorithms QUBO-16/K6, CoT-16 and Greedy-16?

**No.** Sampling iterates four fixed prompt templates and `num_answers=4`, giving 16 traces. `num_reasons` is stored but not used by this loop. QUBO first chooses an answer group and solves only within that group; K becomes `min(6, group size)` and filtering can reduce it further. Final prompts contain reasoning text without the corresponding extracted answers.

The named `cot` and `greedy` baselines each generate **one** deterministic answer. Separate `sc16` and `best16` controls exist, but `best16` is one highest-score answer, not top-six synthesis. With no more than 20 variables the configured solver exhaustively minimizes the objective; this is not evidence of an annealing or quantum speed advantage.

The new diagnostic scripts implement the confirmed definition: 16 shared CoT traces, QUBO exactly six from all 16, Greedy highest-scored six, and CoT majority vote over the same 16. CoT all-16 synthesis and random-six synthesis are additional controls. QUBO/Greedy/random and CoT synthesis each add one deterministic generation; actual prompt lengths/token costs are recorded. Six-trace votes and confidence-weighted voting help distinguish selection effects from the extra synthesis step.

## What the new diagnostics can establish

They regrade raw outputs with benchmark-specific conservative parsing, retain all selected questions in the denominator, save every trace and selection, and compare methods on identical question/seed pairs. Repeated seeds are clustered by question. Reports include paired bootstrap intervals, question-level sign-flip tests, per-seed exact McNemar tests, and Holm correction across report comparisons. “Repeatable advantage” requires at least 30 distinct questions, at least three seeds, a consistent direction across seeds, an interval excluding zero, corrected p ≤ 0.05, and zero execution errors. This is evidence on the observed panel, not proof of a general advantage.

## Completed verification

- **58 tests passed with real PyTorch/TRL installed**, covering grading, exact-six selection, independent energy checks, gold-independent selection, leakage rejection, calibration learning, paired statistics, DPO pair validation, CLI resume/offline analysis, actual tiny-model SFT/DPO updates, frozen SFT reference weights, adapter reload and two spawned CPU workers. Synthetic/tiny-model outputs are never treated as benchmark performance. Evidence: [`verification/pytorch_runtime_report.json`](../verification/pytorch_runtime_report.json) and the accompanying JUnit XML.
- Real dataset loading and all gold labels passed QA for all five datasets. After question de-duplication and calibration-source exclusions, every fit/tune/eval intersection was zero. The persisted evidence is [`verification/dataset_loader_check.json`](../verification/dataset_loader_check.json).
- Available distinct fit / tune / eval pool sizes: GSM8K 5,947 / 1,526 / 1,319; MMLU 1,212 / 317 / 13,909; BBH 1,298 / 1,306 / 3,903; ARC-Challenge 882 / 236 / 1,172; StrategyQA 416 / 476 / 1,397. These are **data pool counts, not completed model evaluations**. MMLU includes 57 subjects and BBH 27 tasks.
- Real-data QA initially detected 28 unique MMLU validation/test overlaps. The loader now excludes calibration-source duplicates and also de-duplicates within each selected pool.
- The pinned CPU API preflight, Python compilation, shell syntax, documentation links, Docker Compose configuration and Git whitespace checks passed. Actual runtime checks exposed and fixed tokenizer input-field handling and NumPy/PyArrow compatibility; prepared dataset caches are now isolated by library version to avoid old Arrow schema reuse. No pretrained-model benchmark accuracy, learned production-weight artifact, A6000 CUDA/NF4/NCCL training or post-training gain has been verified here. The maintained multi-GPU/Slurm guide is [`docs/cluster/README.md`](cluster/README.md).

MMLU subject and BBH task are real dataset metadata. GSM8K/ARC/StrategyQA topics remain **unlabelled** unless independently annotated in a supplied manifest. Keyword guesses are not presented as verified fields. Failure categories identify missing correct candidates, missed correct answers during selection, and wrong final answers despite a correct selected candidate. “Hallucination” is not inferred from accuracy or NLI scores; the report exports an unfilled human review sheet for checking reasoning claims and evidence.
