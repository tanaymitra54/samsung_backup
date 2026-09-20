# Audit of Debanjan Konar's Review vs. What We Actually Built

## Changes made (2026-09-20) — protocol reset, no local fine-tune

SFT / DPO / full H100 eval were **not** run on this machine. The pipeline and
scripts are now aligned with the email. Re-generate data and train on the H100.

Verified here: `python3 tests/test_protocol.py` → **15 passed**.

### Current standing vs the 7 points

| # | Reviewer | Status after this change | What changed |
|---|---|---|---|
| 1 | Exact K, tune only on val | **Code fixed.** Exact K in the solver. Soft λ_c off. Tune script uses GSM8K **train** hold-out | `pipeline/qubo_math.py`, `pipeline/solver.py`, `config/config.yaml` |
| 2 | Matched budget + exhaustive at N=16 | **Code fixed.** Exhaustive for n≤20. Eval reports SC-16 and Best-16 from the **same** 16 traces as QUBO | `pipeline/solver.py`, `pipeline/matched_baselines.py`, `scripts/run_all_benchmarks.py` |
| 3 | n=100/81 is noise | **Reporting fixed.** Wilson 95% CI and ECE helper. Old 81/100 numbers are still exploratory until H100 re-run | `pipeline/stats_utils.py` |
| 4 | No test leak in SFT/tune/thresholds | **Code fixed.** Tune = GSM8K train. MMLU datagen = `validation`. Eval scorer no longer sees gold | `scripts/tune_qubo_params.py`, `scripts/generate_training_data.py`, `pipeline/orchestrator.py` |
| 5 | Do not penalise shared answers; do not mix conflicts | **Code fixed.** `answer_agree_weight=0`. Cosine only inside an answer group. Final prompt keeps one group | `pipeline/answer_groups.py`, `pipeline/qubo_builder.py`, `pipeline/inference.py` |
| 6 | Nested val; target = accuracy + calibration | **Partly fixed.** Tune target stays final-answer match, on train hold-out. ECE helper added. Verifier weights stay **hand-set** (stated, not “learned”) | `scripts/tune_qubo_params.py`, `pipeline/hyperparam_qubo.py` |
| 7 | Verified correct vs incorrect DPO labels | **Code fixed.** `select_verified_pair` requires gold-correct and gold-incorrect. `training/run_dpo.py` no longer appends gold to both sides | `pipeline/preference_labels.py`, `scripts/generate_preference_pairs.py` |

### Extra bugs from the original audit that were also fixed

| Issue | Fix |
|---|---|
| Empty QUBO silently became indices `[0..5]` | Top-score fallback + warning (`pipeline/selection.py`) |
| `xᵀQx` doubled off-diagonal cost | Energy is diagonal + upper triangle once (`qubo_energy`) |
| Gold passed into the scorer at eval | `score_with_gold=False` by default |
| Missing `prepare_final_prompt` / `compose_final_prompt` | Added in `pipeline/inference.py` |
| `hyperparam_qubo.py` looked like a learner | Marked unused stub |

### What you must run on the H100 (not here)

1. `python scripts/tune_qubo_params.py` — train-split hold-out only.
2. `python scripts/generate_training_data.py` — then `generate_preference_pairs.py`.
3. SFT, then DPO, with the new pairs.
4. `python scripts/run_all_benchmarks.py` — read QUBO vs **SC-16 / Best-16**, plus CIs. Do not treat old 81/100 JSON as proof.

Follow-up: **file** `config/config_fast.yaml` now matches exact-K / no answer-agree. HUBO energy and **file** `scripts/generate_comparison.py` use **function** `qubo_energy`. **file** `scripts/prepare_training_data.py` is marked legacy (test CSVs in `outputs/`).

Follow-up after the explore pass: `config/config_fast.yaml` now matches
exact-K / no answer-agree. HUBO energy and `scripts/generate_comparison.py`
use `qubo_energy` (no doubled pairs). `scripts/prepare_training_data.py` is
marked legacy — it still reads test CSVs from `outputs/`.

---

## Historical audit (branch `tanay`, HEAD `855ea22`, before the protocol reset)

**Scope:** every one of the 7 points in the email, verified against the code, configs, saved results and generated data in this repo (branch `tanay`, HEAD `855ea22`).
**Method:** read the pipeline source, configs, tuning scripts, training scripts, eval JSONs and generated datasets; re-measured where the claim was checkable from saved artifacts.
**Bottom line:** 4 of 7 points are correct and unaddressed, 3 are partially correct (the substance holds, but the specific premise is either stale or already partly handled in code). Nothing in the email is flatly wrong. Point 1 is the one that hurts most — we can *measure* that the QUBO is not picking the target subset size.

---

## 0. Verdict at a glance (pre-fix)

| # | Reviewer's point (short) | Verdict | Where we stand |
|---|---|---|---|
| 1 | QUBO has no cardinality term → no target subset size | **Valid — partially addressed, ineffective in practice** | Soft penalty exists in code (`_apply_cardinality_constraint`), but measurement shows we select 1 trace or fall back to "first 6", never a genuine size-6 subset |
| 2 | Comparison confounded: QUBO uses 16 chains, CoT uses 1 | **Correct — unaddressed** | Baselines are 1 generation each; QUBO is 16 + verifier + final call. No budget-matched control exists. No exhaustive ground truth either |
| 3 | 100 / 81 questions is too small; MMLU + BBH may be noise | **Correct — unaddressed** | Confirmed sizes. Same config reruns swing ±10-25 points. No CIs, no repeats, no seed variance report |
| 4 | GSM8K used to generate training chains *and* evaluated | **Partially correct** | The *shipped* SFT data uses GSM8K **train** — no leak there. But `tune_qubo_params.py` tunes on GSM8K **test**, and `load_mmlu` reads MMLU **test**. The point stands for tuning |
| 5 | Don't penalise all traces sharing a final answer | **Correct — we do exactly the opposite, deliberately** | `answer_agree_weight` adds a penalty when two chains share an answer. Conflicting-answer traces are also freely mixed in the final prompt |
| 6 | Learn fusion weights + QUBO params with nested validation/CV | **Correct — largely unaddressed** | Grid search exists but on 10 test questions, no CV/nesting, no calibration metric. Verifier fusion weights are hand-set constants. `hyperparam_qubo.py` is a non-functional stub |
| 7 | Preference pairs: same prompt, one verified-correct vs one incorrect; don't use self-score alone | **Partially correct** | Same-prompt ✓. Independently verified only for math (≈56% of data). For ARC/StrategyQA/LogiQA the label is self-score only. Rejected is "lowest score in pool", not verified-wrong |

---

## 1. Claim-by-claim: reviewer's point vs. what we have

### Point 1 — "The stated QUBO does not enforce a target subset size… Add an exact K constraint and tune only on a validation set"

| Reviewer's claim (left) | What we have actually done (right) | Verdict |
|---|---|---|
| "The Energy function, E(x) contains no cardinality term" | **True against the document he read.** `PIPELINE_SUMMARY_FOR_SHARE.html` §1 states our QUBO as `Q_ii = −correctness + diversity_bonus(0.5)` and `Q_ij = cosine × penalty_weight(2.0)` — no cardinality term. `README.md` §7.1 defines `E(x) = xᵀQx` with the same two terms only. **But the code has moved on:** `pipeline/qubo_builder.py::_apply_cardinality_constraint` implements the soft penalty `λ_c·(Σx_i − k)²`, with diagonal `+λ_c(1−2k)` and off-diagonal `+2λ_c`. Added in commit `449a657` (2026-07-01). `k = pipeline.subset_size = 6`. `λ_c = 0.1` (`config/config.yaml`), `0.05` (`config/best_qubo_params.yaml`). | Claim is **stale vs code**, but see next row — the code fix does not work |
| "It will select every extra trace whose score exceeds its redundancy penalty; it does not inherently choose roughly the target number" | **Measurement confirms him, not us.** `outputs/loop_2026_08_19_205216/round_1/traces.jsonl` — 50 questions, 16 candidates each, run **after** the cardinality fix: `{selected_indices == [0,1,2,3,4,5] → 16 records}` and single-index selections (`[14]`, `[0]`, `[3]`, `[2]`, `[5]`, …) `→ 34 records`. **Genuine size-6 optimal selections: 0/50.** The 16 "6-length" cases are not the solver choosing 6 — `[0,1,2,3,4,5]` is exactly the empty-selection fallback in `pipeline/orchestrator.py::select_best_reasoning` (`if not selected: selected = list(range(min(subset_size, n_samples)))`). So in practice the subset size is *never* enforced. | **Valid** |
| "Add either an exact K constraint" | Not done. We only added a **soft** penalty, `λ_c = 0.05–0.1`, two orders of magnitude below the redundancy terms (`penalty_weight × cosine` ≈ 0.6–1.0). `best_qubo_params.yaml` itself says "λ_c needs empirical tuning per dataset". It was never validated. No exact-K / penalty-sweep-to-equality implementation exists. | **Not done** |
| "tune the hyperparameters only on a validation set" | Violated — see Point 4/6. Tuning ran on GSM8K **test**. | **Not done** |

**Note on the degenerate outcome.** With `db = 0.1, λ_c = 0.05, k = 6`, diagonals become `−q + 0.1 − 0.55 = −q − 0.45`, so every candidate is attractive individually while any pair costs 0.6–1.1. Selecting exactly one is therefore the energy minimum, and selecting zero is the second-best — which is why we see 1s and empty-fallbacks and nothing in between. The soft penalty cannot overcome the pairwise penalties at this λ_c.

---

### Point 2 — "The comparison is presently confounded… Compare at identical generation budgets. With 16 binary variables, exhaustive evaluation of all 65,536 subsets is trivial"

| Reviewer's claim (left) | What we have actually done (right) | Verdict |
|---|---|---|
| "QUBO uses 16/20 sampled chains while CoT appears to use one" | **Confirmed.** `scripts/run_all_benchmarks.py::baseline_greedy` and `baseline_cot` each make exactly one `inference.generate_answer()` call. `run_qubo_pipeline` → `sampler.sample()` which is `len(perturbations)=4 × num_answers=4 = 16` generations, plus verifier scoring, plus one final generation. `config.pipeline.num_answers: 4` with the comment "# 4 perturbations × 4 = 16 chains total". `PIPELINE_SUMMARY_FOR_SHARE.html`: "Candidate generation: 4 prompt perturbations × 4 samples = 16 reasoning chains / question". So greedy = 1×512 tokens, CoT = 1×512 tokens, QUBO = 16×256 + 1×512. Not budget-matched in compute, tokens, or number of forward passes. | **Correct** |
| "Much of the gain may be from test time sampling, not QUBO" | Untested and plausible. No self-consistency-over-16 or best-of-16 control exists in the evaluation path. `scripts/generate_comparison.py` does compare 7 selection methods, but (a) it runs on **2 questions** (`FAST_MODE_QUESTION_LIMIT = 2`) and (b) it never generates a final answer — it only writes energy / average redundancy / average relevance to CSV. So there is no evidence in the repo that isolates QUBO's contribution. | **Correct — unaddressed** |
| "16 binary variables… exhaustive evaluation of all 65,536 subsets is trivial. Use it as the ground truth optimizer" | **The variable count is right.** `qubo_builder` sets `n_clusters = min(max_vars, 50)` but `_cluster_reasons` uses `n = min(len(embeddings), n_clusters)`; with 16 samples that yields 16 singleton clusters and 16 representatives, so `Q` is 16×16 → 2¹⁶ = 65,536 subsets. Brute force is microseconds. **We do not do it.** Grep for brute-force/exhaustive optimisation in `pipeline/` returns nothing; `solver.py` only offers SA (CPU/GPU), parallel tempering, counterdiabatic and OpenJij SQA. So we use a stochastic approximate solver on an instance small enough to solve exactly — no optimality gap is ever reported. | **Correct — not done** |

**Extra note:** the "no guarantee of optimal solution" risk in `README.md` §14 is real but *invisible* at this scale — we could have had the exact ground truth for free, which would also have made the SA-vs-optimal gap a legitimate result to report.

---

### Point 3 — "The reported 100 questions or 81 questions results are too small for validation. Several apparent differences, particularly MMLU and BBH may be sampling noise"

| Reviewer's claim (left) | What we have actually done (right) | Verdict |
|---|---|---|
| "100 questions or 81 questions" | **Confirmed exactly.** `results/eval/all_benchmarks_20260727_*.json`: GSM8K `num_samples: 100`, MMLU `num_samples: 100`, BBH `num_samples: 54` then `81`. The 81 comes from `evaluation/__init__.py::load_bbh`: `per_config = subset_size // 27` BBH tasks, so `subset_size=300 → 11`, and the share snapshot used `n=27`. `config.evaluation.subset_size: 300` is never actually reached in the saved runs. | **Correct** |
| "may be sampling noise" | **Confirmed, empirically.** Three saved runs of the same config give wildly different numbers: GSM8K greedy **0.15 / 0.22 / 0.40**, GSM8K QUBO **0.70 / 0.80 / 0.88**; MMLU QUBO **0.533 / 0.480 / 0.360 / 0.490**; BBH QUBO **0.444 / 0.519 / 0.630**. At n=100 the 95% binomial half-width is ±~10pp, so several of these swings sit outside pure sampling error, and the SFT-vs-base MMLU QUBO delta (+0.480 → +0.360, i.e. *worse*) flips sign vs the DPO run (+0.490). There are no confidence intervals, no repeated seeds, no per-benchmark error bars anywhere in `results/` or `outputs/`. Seeding is nominal only (`set_seed(42)`) because sampling draws `random.uniform` temperatures. | **Correct — unaddressed** |
| — | Two aggravating details the review didn't mention: `load_mmlu` evaluates only **5 STEM subjects × 20 = 100** questions, so "MMLU" here is not MMLU; and BBH at `subset_size=300` gives **11 questions per task**. | Additional |

---

### Point 4 — "GSM8K is used to generate training chains and is also evaluated… never use test questions during SFT or DPO, scoring-weight tuning, prompt design, or threshold selection"

| Reviewer's claim (left) | What we have actually done (right) | Verdict |
|---|---|---|
| "The document says GSM8K is used to generate training chains and is also evaluated" | **True of the document.** `PIPELINE_SUMMARY_FOR_SHARE.html` §2 states the SFT set is `nvidia/OpenMathInstruct-2 (600K) → stratified 20K` **plus** "54 correct CoT traces from our benchmark CSVs (`outputs/`)". Those CSVs are produced by `run_all_benchmarks.py` over GSM8K/MMLU/BBH **test** splits. `scripts/prepare_training_data.py` (the old path) also extracts `pred_cot` traces from `outputs/*/` — i.e. test-split questions — into `training_data/combined_dataset`. So the *documented* recipe does leak test items into SFT. | **Correct against the doc** |
| "Train only on the training split… GSM8K has distinct 7.5K train and 1K test sets" | **The shipped data actually complies for GSM8K.** `data/finetune_train.jsonl` (1774 examples) and `finetune_val.jsonl` (200) were produced by `scripts/generate_training_data.py`, which loads `gsm8k / main / split="train"` (`load_gsm8k`). Source mix: `gsm8k 992, arc 333, strategyqa 311, logiqa 138` (train); val: `gsm8k 111, arc 38, strategyqa 35, logiqa 16`. No GSM8K test questions in the shipped SFT/DPO data. ARC also uses `split="train"`. So the specific GSM8K train/test leak he suspects is **not present in the artifact**. | **Incorrect for the artifact** |
| "never use test questions… during scoring-weight tuning, or threshold selection" | **Violated in two places.** (a) `scripts/tune_qubo_params.py::load_gsm8k` → `load_dataset("gsm8k", "main", split="test")` and the winning combo is selected by accuracy on those same test questions — hyperparameters chosen on the eval set. (b) `scripts/generate_training_data.py::load_mmlu` → `load_dataset("cais/mmlu", "all", split="test")`, and `evaluation/__init__.py::load_mmlu` also evaluates MMLU `split="test"` → latent train/test overlap for MMLU, currently inert only because `--n-mmlu` defaults to **0** (`generation_stats.json`: `"mmlu": {"generated": 0}`). (c) Two thresholds are hand-picked with no validation split: `QUALITY_THRESHOLD = 0.70` in `fast_finetune_pipeline.py` and `if top_score < 0.5` in `generate_training_data.py`. | **Correct** |
| "tune on a separate validation split" | `finetune_val.jsonl` exists but is a **90/10 random split of the same pool** (`stratified_split`, `TRAIN_FRAC = 0.90`) and is only used as an SFT eval dataset and for the epoch accuracy metric — never for QUBO or verifier tuning. QUBO tuning has no validation set at all. | **Not done** |

---

### Point 5 — "Do not penalize all traces that share a final answer… penalize duplicate rationales within an answer group. Mixing traces supporting conflicting answers in the final answer prompt is likely to harm accuracy"

| Reviewer's claim (left) | What we have actually done (right) | Verdict |
|---|---|---|
| "Do not penalize all traces that share a final answer" | **We do exactly this, on purpose.** `pipeline/qubo_builder.py`: off-diagonal is `Q[i][j] = (α·cosine + β·answer_agree) × penalty_weight` with `agree = 1.0 if (a_i and a_j and a_i == a_j) else 0.0`, where `a` is the normalised extracted final answer. `config/config.yaml`: `answer_sim_weight: 0.6`, `answer_agree_weight: 0.4`; `config/best_qubo_params.yaml`: `0.7 / 0.3`. The code comment calls it "Fix #2 — Answer-aware off-diagonal… Two chains with the same extracted answer are penalised for co-selection even if their text embeddings appear diverse." So β·penalty_weight (0.3–0.4 × 1.0 ≈ 0.3–0.4 of a total off-diagonal ≤ ~1.1) is spent actively suppressing answer agreement. | **Correct — implemented the opposite** |
| "Agreement is often useful evidence" | Internally inconsistent: the verifier already rewards agreement via `consensus_weight: 0.10` (self-consistency, Wang et al.) on the **diagonal**, while the off-diagonal penalises it. The same signal is simultaneously rewarded and punished. | Additional |
| "penalize duplicate rationales within an answer group" | `α·cosine_similarity` (MiniLM embeddings, `answer_sim_weight`) does penalise near-duplicate rationales, so half the advice is covered — but it is not restricted to *within* an answer group, so it also penalises complementary rationale that happens to be lexically similar. | **Partially covered** |
| "Mixing traces supporting conflicting answers in the final answer prompt is likely to harm accuracy" | **Confirmed — nothing prevents it.** `pipeline/inference.py::build_final_prompt` joins the QUBO-selected reasons into one numbered list with no answer-group filtering, no majority vote, no consistency check. `selected_reasons` can freely mix traces that conclude different answers (they were only *penalised* for agreeing, never grouped). The only filter is a relevance re-rank by cosine to the question. | **Correct — unaddressed** |

---

### Point 6 — "Learn the score fusion weights and QUBO parameters with nested validation or cross-validation. The target must be final-answer accuracy and calibration"

| Reviewer's claim (left) | What we have actually done (right) | Verdict |
|---|---|---|
| "Learn the score fusion weights" | **Not done.** Verifier fusion weights are hard-coded constants in `config/config.yaml → verifier.scoring`: math `0.63 / 0.27 / 0.10`, language `0.60 / 0.25 / 0.15`, plus `structure_mu: 1.5`, `structure_tau: 1.0`. They are *justified in comments*, never fitted. The share doc still advertises the older `gold_match×0.6 + arithmetic_consistency×0.4`, so the documented numbers don't even match the config. | **Correct — not done** |
| "and QUBO parameters with nested validation or cross-validation" | **Grid search yes, validation no.** `scripts/tune_qubo_params.py` sweeps `penalty_weight × diversity_bonus × cardinality_penalty × answer_agree_weight` — nominally 81 combos; `results/qubo_hyperparam_search.json` contains **52** entries (partial), evaluated over **`effective_n` = 9–10 GSM8K test questions** with `excluded: 1`. `best_qubo_params.yaml` admits: "52/81 combos evaluated, pw=1.5 not yet tested… All pw=1.0 combinations scored identically… First pw=1.0 combo used as canonical best." Accuracy only ever took the values 0.5 / 0.6 / 0.7 / 0.7778 across all 52 combos — 4 distinct outcomes from 52 configurations. No cross-validation, no nesting, no held-out selection. | **Correct** |
| "The target must be final-answer accuracy and calibration" | **Half satisfied.** Target *is* final-answer accuracy: `tune_qubo_params.run_grid_search` extracts the last number from the generated answer and matches gold via `numeric_match`. **Calibration is measured nowhere** — no ECE, no Brier, no reliability curve anywhere in the repo; the verifier's `correctness_score` is treated as a ranking signal only. | **Partially done** |

**Related dead code.** `pipeline/hyperparam_qubo.py` (`HyperparameterQUBO`) looks like it was meant to be the "learn the parameters" component — it is exported from `pipeline/__init__.py` but **never constructed anywhere**. It is also non-functional: `build_hyperparam_qubo` returns a `Q` whose only content is a uniform `10.0` inside each one-hot block with no objective term, and `decode_solution` silently falls back to index 0 when a block is all-zero, so the one-hot constraint is never enforced. Claims about "QUBO hyperparameter abstraction layers" in `CODEBASE_DOCUMENTATION.md` describe a stub.

---

### Point 7 — "Keep a clean training mixture… generate preference pairs from the same prompt where one solution is independently verified correct and another is incorrect. Avoid using self-score alone as the DPO preference label"

| Reviewer's claim (left) | What we have actually done (right) | Verdict |
|---|---|---|
| "clean training mixture for math and general reasoning" | **Reasonably satisfied.** `data/finetune_train.jsonl` mixes `gsm8k 992 / arc 333 / strategyqa 311 / logiqa 138`; OpenOrca is deliberately disabled (`--n-openorca 0`, "irrelevant NLP annotation tasks hurt reasoning SFT") and MMLU contributed 0. So the mixture is intentional and documented. **But** `PIPELINE_SUMMARY_FOR_SHARE.html` still describes a ~20K OpenMathInstruct-2 mixture that was never the artifact actually used — the doc and the data disagree. | **Partially** |
| "pairs from the same prompt" | **Done.** `scripts/generate_preference_pairs.py` iterates one JSONL record (= one prompt) and takes `chosen` = the record's assistant message, `rejected` = `min(pool, key=correctness_score)` from that same record's `full_chains_pool`. Same prompt ✓. Output: `data/preference_pairs.jsonl`, 1187 pairs, `--min-margin 0.3`. | **Done** |
| "one solution independently verified correct and another incorrect" | **Verified-correct holds only for math.** In `generate_training_data.py::run_qubo_pipeline` step 7b, the gold check (`if task_type == "math"` → `_gold_match(pred_num, gold_num)`, else `filtered`) only runs for math. So for GSM8K the chosen chain is independently gold-verified. For `arc` (333), `strategyqa` (311) and `logiqa` (138) — **782 of 1774 training examples = 44%** — nothing is checked against gold; the chosen label is the verifier's own `correctness_score` (NLI + lexical coverage + structural depth). **The rejected side is never verified incorrect anywhere** — it is simply the lowest-scoring member of the same pool, i.e. another sample from the same model at a different temperature/template. | **Partially correct** |
| "Avoid using self-score alone as the DPO preference label" | **We use self-score alone as the margin.** `generate_preference_pairs.py` reads `chosen_score = meta.correctness_score` and `rejected_score = lowest_chain.correctness_score` and keeps the pair iff `chosen_score − rejected_score ≥ 0.3`. The only independent verification in the loop is the math gold filter that shaped `chosen` upstream. For 44% of the data the label is purely self-score. | **Correct** |

**Latent bug worth knowing about.** `training/run_dpo.py::format_dpo_pair` appends the *same* gold answer to both members:

```python
chosen   = f"{chosen_trace}\n\nAnswer: {gold_answer}<|im_end|>"
rejected = f"{rejected_trace}\n\nAnswer: {gold_answer}<|im_end|>"
```

so DPO would learn to prefer one rationale while both end in the correct answer — not a right-vs-wrong preference. That file also reads `training_data/positive_traces.json` / `negative_traces.json`, neither of which exists in the repo, so it is not the path that ran. The path that actually ran is `scripts/run_dpo.py` + `data/preference_pairs.jsonl`, which correctly uses each chain's own answer.

---

## 2. Where the review's premise doesn't match the current code

Useful to know before replying, so we neither over-concede nor look like we ignored him.

| Reviewer said | Actually true | Why it matters |
|---|---|---|
| "E(x) contains no cardinality term" | Code has `_apply_cardinality_constraint` (soft, λ_c = 0.05–0.1) since `449a657` (2026-07-01); the **documents** he read omit it | The document is stale, not the code. Fix the doc, but don't claim the requirement is met — the measurement in Point 1 shows it isn't |
| "GSM8K is used to generate training chains and is also evaluated" | Shipped SFT data uses GSM8K `split="train"`; only the older `prepare_training_data.py` / share-doc recipe pulls test-split CoT traces | We can answer this concretely rather than conceding a leak we don't have |
| Verifier weights | Share doc says `0.6/0.4`; `config.yaml` says `0.63/0.27/0.10` | The doc describes a pipeline that no longer exists |
| "81 questions" | 81 = BBH at `subset_size=300`; GSM8K/MMLU were 100 | Both numbers map to real saved runs |

---

## 3. Scorecard

**Already satisfied (no action needed)**

| Requirement | Evidence |
|---|---|
| Same-prompt preference pairs | `generate_preference_pairs.py` groups by record |
| Train-split GSM8K in SFT data | `generate_training_data.py::load_gsm8k` → `split="train"`; ARC `split="train"` |
| Train/val split for SFT | `stratified_split`, `finetune_train.jsonl` 1774 / `finetune_val.jsonl` 200 |
| Tuning target is final-answer accuracy | `run_grid_search` → `numeric_match(pred, gold)` |
| Preference margin filter | `--min-margin 0.3`, 1187 pairs |
| Deliberate, documented mixture choice | OpenOrca/MMLU dropped on purpose |

**Not satisfied (all confirmed above)**

| Requirement | Where it breaks |
|---|---|
| Exact-K / enforced subset size | `qubo_builder._apply_cardinality_constraint` (soft λ_c), measured 0/50 genuine size-6 |
| Budget-matched baselines | `run_all_benchmarks.py` greedy/CoT = 1 generation each |
| Exhaustive ground-truth optimiser | Not implemented; SA used on a 2¹⁶ instance |
| Sample sizes / statistics | n = 100 GSM8K/MMLU, 54–81 BBH; no CIs or repeats |
| No test set in tuning | `tune_qubo_params.load_gsm8k` → `split="test"` |
| No test set in SFT (MMLU) | `generate_training_data.load_mmlu` → `split="test"` (currently inert, n=0) |
| Don't penalise answer agreement | `answer_agree_weight` 0.3–0.4 |
| Don't mix conflicting answers in the final prompt | `inference.build_final_prompt` has no answer-group filter |
| Learn verifier fusion weights | Hard-coded in `config.yaml` |
| Nested CV / cross-validation | None |
| Calibration metric | None anywhere |
| Independently verified *incorrect* rejected chain | Rejected = lowest self-score in pool |
| Independent verification for non-math | 44% of training examples are self-score-labelled |

---

## 4. Fix list, ordered by what unblocks the most

1. **Enforce K properly.** Replace the soft penalty with a penalty sweep or exact-K binary search so the optimum has `Σx_i = k`, then verify on the saved 50-question trace log that subset sizes actually equal 6. Until then, any claim of "QUBO selects the best subset" is unsupported — the solver currently returns 1 trace or an empty selection that `orchestrator.select_best_reasoning` silently replaces with candidates 0–5.
2. **Add the exhaustiveness baseline.** For ≤20 variables, brute-force all subsets and report (a) the optimal energy and (b) SA's optimality gap. Free, exact, and it directly answers point 2's second half. Only argue for annealing beyond that scale.
3. **Budget-match everything.** Add a 16-sample self-consistency baseline and a best-of-16 baseline with the same token budget as the QUBO arm, then re-run. If QUBO's advantage survives that, the paper-worthy claim is real; if not, we found the confound early.
4. **Move all tuning onto a held-out split.** Split GSM8K *train* into tune/val; never touch `split="test"` in `tune_qubo_params.py`. Switch `load_mmlu` to `split="dev"`/`validation` before it is ever enabled.
5. **Drop the answer-agreement penalty.** Set `answer_agree_weight: 0.0`, keep `α·cosine` for duplicate-rationale suppression, and instead group selected traces by their extracted answer before composing the final prompt (majority group, or best-scoring group only).
6. **Learn the fusion weights, or stop calling them learned.** Either fit `0.63/0.27/0.10` etc. on the tuning split, or delete `pipeline/hyperparam_qubo.py` and state plainly that weights are hand-set priors.
7. **Report uncertainty.** n=100 → always print a binomial CI; run ≥3 seeds; increase `subset_size` for MMLU beyond 5 STEM subjects and for BBH beyond 11 questions/task.
8. **Fix the preference labels.** Keep the math gold check, extend an independent check to ARC/StrategyQA/LogiQA (they all have gold answers available), and label rejected as *verified wrong*, not *lowest self-score*.
9. **Resync the documents.** `PIPELINE_SUMMARY_FOR_SHARE.html`, `README.md` §7.1 and `CODEBASE_DOCUMENTATION.md` must show the cardinality term, the real verifier weights, and the dataset that was actually used (1974 QUBO-curated examples, not 20K OpenMathInstruct).

---

## 5. Additional issues found while auditing (not in the review)

| Issue | Evidence | Impact |
|---|---|---|
| Silent fallback masks solver failure | `orchestrator.select_best_reasoning`: `if not selected: selected = list(range(min(subset_size, n_samples)))`. 16/50 saved traces hit this | QUBO output is arbitrary (first 6 by index) in ~32% of cases and nothing logs it |
| Off-diagonal penalty is doubled by `xᵀQx` | `E = xᵀQx` counts `Q_ij` twice, but `qubo_builder` writes the intended pairwise cost directly into `Q[i][j]`. README's worked example ("penalty for sim 0.9 is 1.8") is therefore 3.6 in reality; the cardinality `+2λ_c` lands as `4λ_c` | Documented energy arithmetic is off by 2×, and `penalty_weight`/`λ_c` meanings are skewed |
| `pipeline/hyperparam_qubo.py` is dead and broken | Never constructed (only re-exported in `pipeline/__init__.py`); `build_hyperparam_qubo` emits uniform `10.0` one-hot blocks with no objective; `decode_solution` ignores an all-zero block | Misleadingly named module suggests parameter learning that does not exist |
| `fast_finetune_pipeline.py::stage_eval` references undefined `base_data` / `ft_data` after calling `_print_comparison` | `scripts/fast_finetune_pipeline.py` — `all_bms = sorted(set(k for k in base_data …))` with no definition | `NameError` if that path is reached; the comparison table never prints |
| MMLU eval is 5 STEM subjects only | `evaluation/__init__.py::load_mmlu` subjects list; `per_subject = subset_size // 5` | "MMLU 100 questions" is a narrow STEM slice, not the standard benchmark |
| BBH eval is 3 questions/task at `subset_size=81` | `per_config = subset_size // len(bbh_configs)` | Per-task numbers are meaningless; only the pool is reported |
| `compute_accuracy` in `evaluation/__init__.py` does bidirectional substring matching | `elif truth.strip().lower() in pred.strip().lower(): correct += 1` | Would inflate open-ended accuracy if used; `run_all_benchmarks.py` uses its own stricter `is_correct` so saved results are unaffected |
| Candidate pool is 12, not 16, in the shipped data | `data/finetune_train.jsonl`: `full_chains_pool` size 12 for 1553/1774 records, 0 for 221 | The "16 chains" figure in the share doc doesn't match the data-generation run |

---

## 6. Evidence index

| Artifact | What it proves |
|---|---|
| `pipeline/qubo_builder.py` | `_apply_cardinality_constraint`, `answer_agree_weight`, `top_k_per_cluster`, energy definition |
| `config/config.yaml`, `config/best_qubo_params.yaml` | λ_c 0.1 / 0.05, k = 6, β 0.4 / 0.3, verifier weights, `num_answers: 4` |
| `results/qubo_hyperparam_search.json` | 52 combos, `effective_n` 9–10, 4 distinct accuracies |
| `results/eval/all_benchmarks_20260727_*.json` | n = 100 / 100 / 54 / 81; run-to-run swings |
| `results/eval/{base,sft,dpo}_{gsm8k,mmlu,bbh}_results.jsonl` | Base 0.800 → SFT 0.840 → DPO 0.880 (GSM8K QUBO) vs MMLU QUBO 0.480 → 0.360 → 0.490 |
| `outputs/loop_2026_08_19_205216/round_1/traces.jsonl` | 34/50 single-trace, 16/50 fallback `[0..5]`, 0/50 genuine size-6 |
| `scripts/tune_qubo_params.py` | `split="test"` in `load_gsm8k`; accuracy-on-test selection |
| `scripts/generate_training_data.py` | GSM8K `split="train"`, MMLU `split="test"`, math-only gold filter, `top_score < 0.5` threshold |
| `scripts/generate_preference_pairs.py` | chosen/rejected both from `correctness_score`; margin 0.3 |
| `training/run_dpo.py` | gold answer appended to *both* chosen and rejected; reads non-existent pos/neg files |
| `scripts/run_all_benchmarks.py` | 1-generation baselines vs 16-chain QUBO |
| `pipeline/solver.py` | SA / PT / counterdiabatic / OpenJij only — no exact solver |
| `pipeline/inference.py` | `build_final_prompt` with no answer-group filtering |
| `evaluation/__init__.py` | Test splits, 5-subject MMLU, `per_config` BBH |
| `data/generation_stats.json` | mmlu generated 0; gsm8k/arc/strategyqa/logiqa yields; avg scores |
| `pipeline/hyperparam_qubo.py` | Non-functional stub, never constructed |
| `PIPELINE_SUMMARY_FOR_SHARE.html` | The document the review was written against (no cardinality term, `0.6/0.4` weights, 20K OpenMathInstruct plan) |
