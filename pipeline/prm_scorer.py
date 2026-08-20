"""
==============================================================================
FILE: pipeline/prm_scorer.py
ROLE: Process Reward Model (PRM) chain scorer -- step-level reasoning quality
==============================================================================

WHY THIS EXISTS
---------------
Measured on the cached 300-question GSM8K pool, the existing gold-free quality
signal (0.35 x arithmetic-consistency + 0.65 x cross-chain consensus) cannot
separate a correct chain from a popular-but-wrong one:

    plain majority vote over all 20 chains ....... 90.3%
    quality-weighted vote over all 20 chains ..... 90.3%
    QUBO-selected subset vote (tuned, 81 combos).. 90.3%
    ORACLE (any of the 20 chains is correct) ..... 96.7%

All three selection strategies land on exactly the same 271/300, because the
dominant term (consensus) IS majority voting -- so anything built on it
reproduces the majority answer by construction. The 6.3-point gap to oracle
lives entirely in 19 questions where the majority is wrong but a correct chain
exists. On those 19, the current signal ranks the correct chains ABOVE the
wrong-majority chains in only 3 cases (16%) -- worse than a coin flip, because
on exactly those questions "correct" and "popular" are anti-correlated and the
signal is 65% popularity.

A Process Reward Model breaks that circularity. It scores each reasoning STEP
on its own merits, with no reference to what any other chain concluded, so a
correct-but-lonely chain can outrank a fluent-but-wrong crowd. This is the
best-supported result in the test-time-compute literature:

  * Lightman et al. 2023, "Let's Verify Step by Step" -- step-level (process)
    supervision decisively beats outcome-level supervision for selection.
  * Wang et al. 2024, "Math-Shepherd" -- automatic PRM label construction.
  * Snell et al. 2024, "Scaling LLM Test-Time Compute Optimally..." --
    verifier-guided selection beats self-consistency at matched sample budget.

AGGREGATION
-----------
A PRM emits one correctness probability per step. Turning that vector into one
chain score is a real modelling choice, so it is configurable:

  "min"   -- the weakest step gates the chain. Standard for PRM-as-verifier:
             one broken step invalidates a derivation regardless of how clean
             the rest looks. Default.
  "prod"  -- product of step probabilities. Similar intent to min but decays
             with chain length, so it penalises long chains structurally.
  "mean"  -- average step quality. More forgiving; use when chains are noisy
             and a single mislabelled step should not veto an otherwise sound
             derivation.
  "last"  -- probability at the final step only, i.e. an outcome-style reward.
             Included mainly as an ablation against the process-level modes.

MODELS
------
Tested against the Qwen2.5-Math-PRM family, which marks step boundaries with a
special "<extra_0>" token and exposes a 2-class head at those positions.
`Qwen/Qwen2.5-Math-PRM-7B` is the recommended choice on an 80GB card;
`Skywork-o1-Open-PRM-Qwen-2.5-1.5B` fits a small GPU for code validation.
"""

from __future__ import annotations

import re
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Step-boundary token used by the Qwen2.5-Math-PRM family.
_QWEN_STEP_SEP = "<extra_0>"

_PRM_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


class PRMScorer:
    """Scores a reasoning chain by grading each of its steps independently."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Math-PRM-7B",
        device: Optional[str] = None,
        aggregation: str = "min",
        cache_dir: Optional[str] = None,
        max_length: int = 4096,
    ):
        if aggregation not in ("min", "prod", "mean", "last"):
            raise ValueError(
                f"aggregation must be one of min/prod/mean/last, got {aggregation!r}"
            )
        self.aggregation = aggregation
        self.max_length = max_length
        self.model_name = model_name

        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        use_cuda = self.device.type == "cuda"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )

        # Qwen2.5-Math-PRM ships its own modeling code (trust_remote_code), and
        # that code reads `config.pad_token_id` directly. Recent transformers
        # moved the generation-related token ids off PretrainedConfig, so the
        # attribute no longer exists and model construction dies with
        # "'Qwen2RMConfig' object has no attribute 'pad_token_id'".
        #
        # Loading the config explicitly and setting the attribute ourselves fixes
        # it without pinning transformers or patching the vendored model file:
        # once it is in the config's __dict__, the remote code finds it.
        config = AutoConfig.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True
        )
        if getattr(config, "pad_token_id", None) is None:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.tokenizer.eos_token_id
            config.pad_token_id = pad_id

        load_kwargs = {
            "cache_dir": cache_dir,
            "config": config,
            "torch_dtype": torch.bfloat16 if use_cuda else torch.float32,
            "trust_remote_code": True,
        }
        if use_cuda:
            # Place shards straight onto the GPU instead of materialising the
            # whole 7B on CPU first and then copying it across.
            load_kwargs["device_map"] = {"": self.device.index or 0}
            load_kwargs["low_cpu_mem_usage"] = True

        self.model = AutoModel.from_pretrained(model_name, **load_kwargs)
        if not use_cuda:
            self.model = self.model.to(self.device)
        self.model.eval()

        sep_ids = self.tokenizer.encode(_QWEN_STEP_SEP, add_special_tokens=False)
        if len(sep_ids) != 1:
            raise ValueError(
                f"{model_name} does not tokenise {_QWEN_STEP_SEP!r} as a single token "
                f"(got {len(sep_ids)}). This scorer targets the Qwen2.5-Math-PRM "
                "step-separator format; a different PRM family needs its own adapter."
            )
        self.step_sep_id = sep_ids[0]

    # ── Step segmentation ────────────────────────────────────────────────────

    @staticmethod
    def split_steps(reason: str) -> list[str]:
        """Split a reasoning trace into steps.

        Chains in this pipeline are free-form text, not pre-segmented, so steps
        are recovered heuristically: explicit numbered/bulleted markers when the
        model produced them, otherwise non-empty lines, otherwise sentences.
        Segmentation quality matters -- a PRM grades whatever spans it is given
        -- so this prefers the model's own structure wherever it exists.
        """
        text = (reason or "").strip()
        if not text:
            return []

        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

        # Prefer explicit step markers ("1.", "Step 2:", "- ", "* ") when the
        # model emitted them: those are the author's own step boundaries.
        marked = [
            ln for ln in lines
            if re.match(r"^(?:step\s*\d+\s*[:.)]|\d+\s*[.)]|[-*•])\s+", ln, re.I)
        ]
        if len(marked) >= 2:
            return marked

        if len(lines) >= 2:
            return lines

        # Single blob: fall back to sentence boundaries.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return sentences if sentences else [text]

    # ── Scoring ──────────────────────────────────────────────────────────────

    def _step_probs(self, question: str, steps: list[str]) -> list[float]:
        """Per-step correctness probabilities from the PRM head."""
        if not steps:
            return []

        # The PRM reads the steps as an assistant turn with each step terminated
        # by the separator token; it emits a judgement at each separator.
        assistant = _QWEN_STEP_SEP.join(steps) + _QWEN_STEP_SEP
        messages = [
            {"role": "system", "content": _PRM_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        with torch.inference_mode():
            outputs = self.model(input_ids=enc["input_ids"])

        # Qwen2.5-Math-PRM returns a 2-class logit per position; class 1 is
        # "this step is correct". Its vendored modeling code returns a bare
        # tuple/tensor rather than a standard ModelOutput with `.logits`, and
        # ModelOutput itself supports [0] indexing, so probe in that order
        # instead of assuming either shape.
        if torch.is_tensor(outputs):
            logits = outputs
        elif hasattr(outputs, "logits") and outputs.logits is not None:
            logits = outputs.logits
        else:
            logits = outputs[0]

        if logits.shape[-1] != 2:
            raise ValueError(
                f"{self.model_name} produced a final dimension of {logits.shape[-1]}, "
                "expected 2 (incorrect/correct). This scorer assumes the "
                "Qwen2.5-Math-PRM 2-class step head."
            )
        probs = F.softmax(logits.float(), dim=-1)[..., 1]  # (1, seq)

        mask = enc["input_ids"] == self.step_sep_id
        step_probs = probs[mask].detach().cpu().tolist()

        # Truncation can drop trailing separators; that is fine, we grade what
        # survived rather than silently padding with optimistic scores.
        return step_probs

    def _aggregate(self, step_probs: list[float]) -> float:
        if not step_probs:
            # No gradeable steps -- neutral, not zero. Returning 0 here would
            # let a parsing failure masquerade as a confidently-bad chain.
            return 0.5
        if self.aggregation == "min":
            return float(min(step_probs))
        if self.aggregation == "mean":
            return float(sum(step_probs) / len(step_probs))
        if self.aggregation == "last":
            return float(step_probs[-1])
        prod = 1.0
        for p in step_probs:
            prod *= p
        return float(prod)

    def score_chain(self, question: str, reason: str) -> tuple[float, list[float]]:
        """Return (aggregate_score, per_step_probabilities) for one chain."""
        steps = self.split_steps(reason)
        step_probs = self._step_probs(question, steps)
        return self._aggregate(step_probs), step_probs

    def score_chains(
        self, question: str, chains: list[dict], write_key: str = "prm_score"
    ) -> list[dict]:
        """Score every chain for one question, writing results in place.

        Writes `prm_score` (aggregate) and `prm_step_scores` (the raw vector).
        The per-step vector is kept because it is the input to the planned
        complementarity objective -- two chains failing on the SAME steps are
        redundant, whereas chains with disjoint weak steps genuinely cover for
        each other. That distinction is invisible to any scalar score.
        """
        for chain in chains:
            score, step_probs = self.score_chain(question, chain.get("reason", ""))
            chain[write_key] = score
            chain["prm_step_scores"] = step_probs
        return chains
