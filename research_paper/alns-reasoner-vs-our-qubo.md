# ALNS-Reasoner vs. our QUBO selection: novelty and differences

Positioning note comparing the ALNS-Reasoner paper against this project's
reasoning-trace selection pipeline.

## The paper under comparison

**"Destroy-repair of reasoning chains via adaptive large neighborhood search in
small language models"**
Kaleem Ullah Qasim, Jiashu Zhang, Muhammad Waqas Aslam, Muhammad Kafeel Shaheen.
*Information Sciences* (Elsevier), DOI `10.1016/j.ins.2026.123959` (online 29 Jul 2026).

Retrievable, verbatim: the paper introduces **ALNS-Reasoner**, which *"treats a
reasoning chain as an editable combinatorial structure and improves it via
destroy-repair cycles drawn from Adaptive Large Neighborhood Search (ALNS)"*,
and *"structured destroy-repair changes the question from how many chains to
sample to where to spend the next call; reframing is the largest accuracy
lever."*

> Retrieval caveat: the full text is paywalled (ScienceDirect captcha, no
> open-access PDF). The objective function, operator definitions, population
> model, benchmarks, and results were **not** retrieved. The operator-level
> points below are inferred from the abstract plus the standard ALNS template
> (Ropke & Pisinger, 2006). Update this file once the full PDF is available.

## Our pipeline (for contrast)

Generate a pool of complete reasoning traces -> score each with a fused
verifier -> build a **QUBO** over whole traces (binary keep/discard) -> select a
size-K subset via exact enumeration / simulated annealing -> compose the final
prompt. Core novelty:

- Reasoning-trace selection as explicit combinatorial optimization.
- Verifier score fused into the QUBO diagonal; semantic-similarity redundancy
  penalty off-diagonal.
- SLM-adapted variable reduction (semantic clustering, target <= 200 vars).
- QUBO-guided SFT feedback loop.

## Comparison

| Dimension | ALNS-Reasoner | Ours (QUBO) |
|---|---|---|
| Paradigm | Metaheuristic: destroy-repair neighborhood search | Quadratic unconstrained binary optimization (quantum-inspired) |
| Unit optimized | One chain as an editable structure (rewrite steps) | Subset of whole pre-generated traces (portfolio selection) |
| Where the LLM runs | Inside the search loop: the repair operator calls the model to regenerate destroyed parts, adaptively allocating calls | Only before (sample pool) and after (final answer); the selection loop is LLM-free |
| Search mechanism | Adaptive operator selection + acceptance; local moves | Global energy `E(x) = sum_i Q_ii x_i + sum_{i<j} Q_ij x_i x_j`; exact-K enumeration (n <= 20) / SA (n > 20) |
| Objective terms | Chain quality + acceptance (no explicit inter-trace quadratic redundancy, per abstract) | Verifier fusion on diagonal (SC / NLI / confidence / arithmetic) + pairwise cosine redundancy off-diagonal + diversity bonus + hard cardinality K |
| Diversity source | Generated on the fly by destroy-repair neighborhoods | Diverse sampling (templates x temperature), then semantic clustering |
| Output | An improved chain | A composed prompt from selected traces (+ SFT data) |
| Compute profile | Sequential; budget spent on LLM calls during search | One-shot combinatorial selection over a frozen pool; parallelizable solver |

## Novelty we hold over ALNS-Reasoner (per available info)

1. **Global selection vs. local generation.** We solve subset selection with
   explicit inter-trace redundancy economics; ALNS-Reasoner edits a single
   chain.
2. **Correctness-aware verifier term** fused into the objective
   (arithmetic / NLI / consistency signals).
3. **Exact-K cardinality constraint** on the selected subset.
4. **QUBO-guided SFT feedback loop** converting inference-time gains into
   model gains.
5. **No LLM calls inside the optimizer**, a different cost profile from
   destroy-repair search.

## Where ALNS-Reasoner differs / is stronger (the honest gap)

- **Their search creates new content** (repair regenerates destroyed segments);
  **ours is pool-limited** - optimal selection cannot fix a bad initial pool.
- Their central claim directly competes with our framing: *"where to spend the
  next call" beats "how many chains to sample."* Any related-work section must
  argue against this and, per the same concern already logged in
  `NOTES/LEAD_FEEDBACK_AUDIT.md`, use **budget-matched baselines** so that
  sampling compute is not mistaken for selection gain.

## Naming caution

"ALNS" (Adaptive Large Neighborhood Search) is **unrelated** to our solver
name. Our exact-K path is enumeration, not ALNS; simulated annealing is only
the n > 20 fallback. Do not let the shared word "search" blur the two methods.

## Follow-up to finish this comparison

1. Obtain the full PDF (institutional access via the DOI) or an author preprint.
2. Extract: objective / acceptance function, destroy and repair operator
   definitions, single-chain vs. population, benchmarks, budget-matched
   baselines, any inter-chain interaction.
3. Rewrite the operator-level rows above with exact facts.
