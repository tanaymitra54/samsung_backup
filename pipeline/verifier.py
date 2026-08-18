"""
pipeline/verifier.py  —  Quality Scorer for Reasoning Traces
=============================================================

PURPOSE
-------
This module assigns a scalar QUALITY SCORE in [0, 1] to each generated
reasoning trace. The score answers the question: "How good is this reason?"

The score feeds directly into the QUBO diagonal (see qubo_builder.py):

    Q[i][i] = -quality_score(reason_i) + diversity_bonus

Because the QUBO solver MINIMISES energy, a higher quality_score makes
Q[i][i] more negative → lowers the system energy when reason_i is selected
→ the solver is incentivised to select high-quality reasons.

TWO SCORING PATHS
-----------------
Scoring logic differs by task type because "quality" means different things:

  1. MATH tasks  (GSM8K, MATH-500, AIME, ...)
     ─────────────────────────────────────────
     A math reason is good if it:
       (a) Reaches the CORRECT final numerical answer   [primary signal]
       (b) Has internally consistent arithmetic steps   [secondary signal]

     Formula:
         score = 0.70 × answer_match  +  0.30 × arithmetic_consistency

  2. LANGUAGE / COMMONSENSE tasks  (BBH, StrategyQA, ARC, MMLU, ...)
     ─────────────────────────────────────────────────────────────────
     These require richer scoring (professor emphasis). A language reason
     is good if it:
       (a) Logically ENTAILS the answer (NLI signal)    [dominant, 60%]
       (b) Addresses the question's key CONCEPTS         [coverage, 25%]
       (c) Shows multi-step REASONING STRUCTURE          [structure, 15%]

     Formula:
         score = 0.60 × P(entailment)
               + 0.25 × lexical_coverage(reason, question)
               + 0.15 × structural_completeness(reason)

All weights are configurable in config.yaml under verifier.scoring.
"""

import ast
import re
import yaml
import numpy as np
import torch
from typing import Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline.device_utils import resolve_device


# ---------------------------------------------------------------------------
# MINIMAL ENGLISH STOPWORD LIST
#
# WHY: When computing lexical coverage (Signal 2 for language tasks), we want
# to measure overlap of MEANINGFUL words — nouns, verbs, adjectives. Function
# words like "the", "is", "in" appear in virtually every sentence and would
# inflate the overlap score without adding signal. We filter them out.
#
# We use a hardcoded set instead of NLTK/spaCy to avoid adding heavy
# dependencies to the project.
# ---------------------------------------------------------------------------
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "each", "and", "but", "or", "nor", "so", "yet", "both",
    "either", "neither", "not", "very", "just", "this", "that", "these",
    "those", "it", "its", "he", "she", "they", "we", "you", "i", "me",
    "him", "her", "us", "them", "what", "which", "who", "how", "why",
    "when", "where", "if", "then", "than", "more", "most", "such", "same",
    "also", "only", "like", "get", "got", "let", "make", "one", "two",
    "three", "all", "any", "no", "so", "up", "out", "now", "our", "your",
}


class ReasonVerifier:
    """
    Scores reasoning traces on a [0, 1] quality scale.

    Every formula and coefficient below is explained inline so the scoring
    logic can be defended clearly in a mentor review.
    """

    def __init__(self, config_path: str = "config/config.yaml", device: str | None = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        verifier_cfg = self.config["verifier"]
        self.math_mode = verifier_cfg["math_mode"]
        self.nli_threshold = verifier_cfg["nli_threshold"]

        # -----------------------------------------------------------------
        # Load scoring weights from config.yaml (verifier.scoring section).
        # Defaults are the values justified in the design document.
        # Your mentor can change these without touching any Python file.
        # -----------------------------------------------------------------
        scoring = verifier_cfg.get("scoring", {})
        self.math_answer_w    = scoring.get("math_answer_weight",     0.63)
        self.math_consist_w   = scoring.get("math_consistency_weight", 0.27)
        self.consensus_w      = scoring.get("consensus_weight",        0.10)
        self.lang_nli_w       = scoring.get("lang_nli_weight",        0.60)
        self.lang_cov_w       = scoring.get("lang_coverage_weight",   0.25)
        self.lang_struct_w    = scoring.get("lang_structure_weight",  0.15)
        # Sigmoid parameters for structural completeness (language only)
        self.struct_mu        = scoring.get("structure_mu",  1.5)
        self.struct_tau       = scoring.get("structure_tau", 1.0)
        # Gold-free weights, used by score_batch() when no answer key is supplied
        # (i.e. at evaluation / deployment). See config.yaml for the rationale.
        self.gold_free_consensus_w   = scoring.get("gold_free_consensus_weight",   0.65)
        self.gold_free_consistency_w = scoring.get("gold_free_consistency_weight", 0.35)

        preferred_device = device or self.config.get("evaluation", {}).get("device")
        self.device = resolve_device(preferred_device)

        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            verifier_cfg["nli_model"],
        ).to(self.device)
        self.nli_tokenizer = AutoTokenizer.from_pretrained(verifier_cfg["nli_model"])
        self.nli_model.eval()

        # Determine the label index for "entailment" from the model's own config.
        # For cross-encoder/nli-deberta-v3-base the order is:
        #   {contradiction: 0, entailment: 1, neutral: 2}
        # We read it from the model config so this still works if the NLI model
        # is swapped for another one with a different label ordering.
        label2id = getattr(self.nli_model.config, "label2id", {})
        self._entail_idx = label2id.get(
            "entailment", label2id.get("ENTAILMENT", 1)
        )

    # =========================================================================
    # ─── SHARED UTILITIES ────────────────────────────────────────────────────
    # =========================================================================

    # -------------------------------------------------------------------------
    # Fix #4: AST-based arithmetic evaluator
    # -------------------------------------------------------------------------

    def _ast_eval_node(self, node) -> Optional[float]:
        """
        Recursively evaluate an AST node without using eval().

        SUPPORTED NODE TYPES:
          ast.Constant  — numeric literals (Python 3.8+; replaces deprecated ast.Num)
          ast.UnaryOp   — unary minus (-x)
          ast.BinOp     — binary arithmetic (+, -, *, /)

        Any other node type (e.g. ast.Call, ast.Name) returns None immediately,
        so expressions with function calls or variable references are silently
        skipped — they cannot be verified without execution context.

        WHY NOT eval():
        Even a restricted eval() namespace can be escaped via crafted LLM output.
        Pure AST-tree recursion is fully sandboxed: only the node types in this
        function's match arms can ever produce a value.
        """
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            operand = self._ast_eval_node(node.operand)
            return -operand if operand is not None else None
        if isinstance(node, ast.BinOp):
            left  = self._ast_eval_node(node.left)
            right = self._ast_eval_node(node.right)
            if left is None or right is None:
                return None
            op = node.op
            if isinstance(op, ast.Add):  return left + right
            if isinstance(op, ast.Sub):  return left - right
            if isinstance(op, ast.Mult): return left * right
            if isinstance(op, ast.Div):
                if abs(right) < 1e-12:
                    return None   # division by zero — skip
                return left / right
        return None

    def _evaluate_expressions_ast(self, text: str) -> list[tuple]:
        """
        Extract arithmetic expressions of the form "LHS = RHS" from text and
        verify each one by evaluating LHS via safe AST recursion.

        EXTRACTION STRATEGY:
          Regex finds all "<expr> = <number>" patterns where <expr> may contain
          digits, parentheses, and arithmetic operators (including Unicode × and ÷).
          The stated RHS is the number immediately after the "=" sign.

        SANITISATION:
          Before parsing, operator synonyms are normalised:
            × → *    (Unicode multiplication)
            ÷ → /    (Unicode division)
            \u00d7 → *  (common encoding)
          A bare "x" between two numbers (e.g. "3 x 4") is converted to "*".

        VERIFICATION:
          is_correct = |computed_LHS − stated_RHS| < tolerance
          where tolerance uses the same hybrid formula as _gold_match:
            max(0.01, 0.01 × |stated_RHS|)

        RETURNS:
          List of (lhs_str, stated_result, computed_result, is_correct) tuples.
          Expressions that cannot be parsed or evaluated are silently skipped.
        """
        # Broad pattern: captures anything that looks like "arithmetic_expr = number"
        # The LHS may contain digits, spaces, operators, and parentheses.
        pattern = re.compile(
            r'([\d\s().+\-*/×÷x]+)'  # LHS expression
            r'\s*=\s*'
            r'(-?\d+(?:[.,]\d+)?)',     # stated RHS (number)
        )

        results = []
        for m in pattern.finditer(text):
            lhs_raw   = m.group(1).strip()
            rhs_raw   = m.group(2).replace(',', '')  # strip comma thousands-separator

            # Sanitise Unicode operators and "x" multiplication
            lhs_clean = lhs_raw
            lhs_clean = lhs_clean.replace('×', '*').replace('÷', '/').replace('\u00d7', '*')
            # Convert bare "x" used as a multiplication sign (e.g. "3 x 4") to "*".
            # Strategy: replace whitespace-surrounded "x" between digit/paren tokens.
            # We use a two-step approach to avoid variable-width lookbehind (not
            # supported in Python's re module):
            #   Step A: match "digit(s)/closing-paren, optional spaces, x, optional spaces, digit/opening-paren"
            #   Step B: reconstruct with "*" in the middle
            lhs_clean = re.sub(r'([\d)]) x ([\d(])', r'\1 * \2', lhs_clean)
            lhs_clean = re.sub(r'([\d)])x([\d(])',   r'\1*\2',   lhs_clean)
            lhs_clean = lhs_clean.strip()

            try:
                stated_result = float(rhs_raw)
            except ValueError:
                continue

            if not lhs_clean:
                continue

            # Parse and safely evaluate the LHS
            try:
                tree = ast.parse(lhs_clean, mode='eval')
                computed = self._ast_eval_node(tree.body)
            except (SyntaxError, ValueError, RecursionError):
                continue

            if computed is None:
                continue

            # Hybrid tolerance: same formula as _gold_match
            tol = max(0.01, 0.01 * abs(stated_result))
            is_correct = abs(computed - stated_result) < tol
            results.append((lhs_clean, stated_result, computed, is_correct))

        return results

    def _extract_arithmetic(self, text: str) -> list[tuple]:
        """
        Find all explicit arithmetic operations written as  A ○ B = R
        and return them as (operand_a, operator, operand_b, result) tuples.

        PATTERNS HANDLED:
            3 × 4 = 12    3 * 4 = 12    3 x 4 = 12
            12 + 5 = 17   12 - 5 = 7
            10 / 2 = 5    10 ÷ 2 = 5

        WHY we require explicit "= result":
        The step-chaining consistency check (see below) needs to know which
        number is the *result* of each step so it can trace whether that result
        feeds into a later step. Without the explicit result we cannot determine
        chaining, only the operation itself.

        NOTE: This method is retained for backward-compatibility with score_breakdown()
        and external callers. The primary scoring path now uses _evaluate_expressions_ast.
        """
        ops = []
        # Each pattern captures: (left_operand, right_operand, result)
        patterns = [
            # Multiplication: × x *
            (r"(-?\d+\.?\d*)\s*[×x\*]\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)", "*"),
            # Division: ÷ /
            (r"(-?\d+\.?\d*)\s*[÷/]\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)", "/"),
            # Addition
            (r"(-?\d+\.?\d*)\s*\+\s*(-?\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)", "+"),
            # Subtraction (negative sign already handled in number pattern)
            (r"(-?\d+\.?\d*)\s*\-\s*(\d+\.?\d*)\s*=\s*(-?\d+\.?\d*)", "-"),
        ]
        for pattern, op_sym in patterns:
            for a, b, r in re.findall(pattern, text):
                ops.append((float(a), op_sym, float(b), float(r)))
        return ops

    def _extract_last_number(self, text: str) -> Optional[float]:
        """
        Extract the final numeric value from a text string.

        We take the LAST number because reasoning traces typically conclude
        with "the answer is X" — the final number is the answer.
        Handles commas (e.g., "1,234") and negative values.
        Returns None if no number is found.
        """
        cleaned = text.strip().replace(",", "")
        matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
        if not matches:
            return None
        try:
            v = float(matches[-1])
            # Return as int when the value is whole (cleaner display, same numeric value)
            return int(v) if v.is_integer() else v
        except ValueError:
            return None

    def _gold_match(self, pred: Optional[float], gold: Optional[float]) -> float:
        """
        Return 1.0 if the predicted number matches the gold answer, else 0.0.

        HYBRID TOLERANCE FORMULA:
            threshold = max(absolute_floor, relative_fraction × |gold|)
            match = 1.0  if  |pred − gold| < threshold
                    0.0  otherwise

        WHY HYBRID (not pure absolute or pure relative):

          Pure ABSOLUTE tolerance (e.g., |pred - gold| < 0.01):
            Problem: gold=1000, pred=999.5 → difference=0.5 > 0.01 → FAILS.
            A 0.05% error on a large integer is rounding, not a wrong answer.

          Pure RELATIVE tolerance (e.g., |pred - gold| / |gold| < 0.01):
            Problem: gold=0, pred=0.005 → division by zero.
            Problem: gold=0.001, pred=0.0019 → 90% relative error but plausibly rounding.

          HYBRID: threshold = max(0.01, 0.01 × |gold|)
            gold=0      → max(0.01, 0) = 0.01  (pure absolute — safe near zero)
            gold=500    → max(0.01, 5) = 5.00  (1% of 500 — sensible for large values)
            gold=0.005  → max(0.01, 0.00005) = 0.01  (absolute floor kicks in)
            gold=1000   → max(0.01, 10) = 10   (tolerates rounding on large numbers)

        WHY 1% relative fraction:
        GSM8K answers are integers. Any rounding that produces a value within
        1% of the true integer is almost certainly a correct computation that
        was displayed with a decimal. This is a principled and defensible choice.
        """
        if pred is None or gold is None:
            return 0.0
        abs_floor = 0.01          # never stricter than ±0.01
        rel_fraction = 0.01       # 1% relative tolerance
        threshold = max(abs_floor, rel_fraction * abs(gold))
        return 1.0 if abs(pred - gold) < threshold else 0.0

    # =========================================================================
    # ─── MATH SCORING HELPERS ────────────────────────────────────────────────
    # =========================================================================

    def _step_chaining_consistency(self, operations: list[tuple]) -> float:
        """
        [LEGACY] Measure how well arithmetic steps chain into one another.

        This method is retained for backward-compatibility (used in score_breakdown).
        The primary scoring path in verify_math() now uses _evaluate_expressions_ast,
        which directly verifies whether each stated arithmetic result is correct.

        FORMULA:
            consistency = (# steps whose result is used as an operand in a later step)
                          / (total steps − 1)

        EDGE CASE — fewer than 2 operations:
        Returns 0.5 (neutral) to avoid rewarding or penalising sparse traces.
        """
        n = len(operations)
        if n <= 1:
            return 0.5

        chained = 0
        for i in range(n - 1):
            result_i = operations[i][3]
            later_operands = set()
            for j in range(i + 1, n):
                later_operands.add(operations[j][0])
                later_operands.add(operations[j][2])
            threshold = max(0.01, 0.01 * abs(result_i))
            if any(abs(result_i - op) < threshold for op in later_operands):
                chained += 1

        return chained / (n - 1)

    # =========================================================================
    # ─── LANGUAGE SCORING HELPERS ────────────────────────────────────────────
    # =========================================================================

    def _lexical_coverage(self, reason: str, question: str) -> float:
        """
        Fraction of the question's key concepts that appear in the reason.

        FORMULA:
            coverage = |content_words(reason) ∩ content_words(question)|
                       / max(1, |content_words(question)|)

        where content_words(text) = lowercase alphabetic tokens that are:
            (a) NOT in the stopword list   (filters function words)
            (b) NOT purely numeric         (filters digit-only tokens)
            (c) At least 3 characters long (filters noise like 'ok', 'vs')

        WHY THIS SIGNAL:
        A reason that does not mention any of the key entities or verbs from
        the question is likely off-topic or hallucinated. Lexical coverage is
        a fast, model-free sanity check that the reason at least ADDRESSES
        the question before we trust the NLI model's entailment score.

        WHY EXCLUDE NUMERIC TOKENS:
        For math-heavy questions the "content words" are numbers. Overlap of
        raw numbers between a question and its reason is almost guaranteed
        (you're solving for the same numbers) but tells us nothing about
        reasoning quality. We restrict to alphabetic tokens to focus on the
        semantic content.

        NOTE: This function is called ONLY from verify_commonsense(). It is
        never used for math tasks. The defensive numeric filtering is still
        included because some commonsense benchmarks (ARC, MMLU) include
        numerical options.

        EXAMPLE:
            question = "Why do birds migrate south in winter?"
            reason   = "Birds fly south to escape cold temperatures in winter."
            content_q = {birds, migrate, south, winter}  (4 words)
            content_r = {birds, fly, south, escape, cold, temperatures, winter}
            overlap   = {birds, south, winter}   → coverage = 3/4 = 0.75

        EDGE CASE:
        If the question has no content words (e.g., it's a pure number string),
        we return 0.5 (neutral) to avoid dividing by zero and avoid unfairly
        penalising the reason.
        """
        def content_words(text: str) -> set:
            # re.findall(r"[a-z]+") extracts only alphabetic tokens →
            # automatically filters punctuation, digits, and mixed tokens.
            tokens = re.findall(r"[a-z]+", text.lower())
            return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}

        q_words = content_words(question)
        if not q_words:
            return 0.5   # neutral — question has no meaningful content words

        r_words = content_words(reason)
        overlap = q_words & r_words
        return len(overlap) / len(q_words)

    def _structural_completeness(self, reason: str) -> float:
        """
        Score how structurally developed (multi-step) the reasoning trace is.

        FORMULA — sigmoid on sentence count:

            structure(r) = σ( (s − μ) / τ )
            where σ(x) = 1 / (1 + exp(−x))
                  s  = number of non-trivial sentences in the reason
                  μ  = structure_mu  (default 1.5) — sigmoid inflection point
                  τ  = structure_tau (default 1.0) — softness / temperature

        EXAMPLE SCORES (with defaults μ=1.5, τ=1.0):
            1 sentence  → σ((1 − 1.5) / 1.0) = σ(−0.5) ≈ 0.38  (under-developed)
            2 sentences → σ((2 − 1.5) / 1.0) = σ( 0.5) ≈ 0.62  (adequate)
            3 sentences → σ((3 − 1.5) / 1.0) = σ( 1.5) ≈ 0.82  (good)
            5 sentences → σ((5 − 1.5) / 1.0) = σ( 3.5) ≈ 0.97  (thorough)

        WHY SIGMOID (not linear count):
          - Linear scoring would give unbounded rewards for very long reasons,
            incentivising verbose padding over concise correct reasoning.
          - The sigmoid is naturally bounded to (0, 1) and saturates —
            going from 3 to 10 sentences barely improves the score. This
            rewards the PRESENCE of multi-step reasoning, not raw length.
          - The inflection at μ=1.5 means a single-sentence reason (≈0.38)
            is penalised but not catastrophically; two or more sentences
            cross the midpoint (0.5) and receive a positive signal.

        WHY COMMONSENSE ONLY (not math):
          For math tasks a single well-formed equation like "3×4=12, 12+5=17"
          is perfectly valid and extremely terse. Penalising brevity in math
          would unfairly hurt concise-but-correct arithmetic reasoning.
          The structural signal is deliberately excluded from verify_math().

        μ and τ are tunable in config.yaml under verifier.scoring.
        """
        # Split on sentence-ending punctuation followed by whitespace or string end.
        # Filter out very short fragments (len ≤ 5) to ignore trailing punctuation artefacts.
        sentences = re.split(r"(?<=[.!?])\s+", reason.strip())
        sentences = [s for s in sentences if len(s.strip()) > 5]
        s = max(1, len(sentences))

        # Sigmoid: σ((s − μ) / τ)
        z = (s - self.struct_mu) / self.struct_tau
        return float(1.0 / (1.0 + np.exp(-z)))

    # =========================================================================
    # ─── MATH SCORING ────────────────────────────────────────────────────────
    # =========================================================================

    def verify_math(
        self,
        reason: str,
        expected_answer: Optional[str] = None,
        predicted_answer: Optional[str] = None,
    ) -> float:
        """
        Score a math reasoning trace in [0, 1].

        ┌─────────────────────────────────────────────────────────────────┐
        │  FORMULA (when gold answer is available):                       │
        │                                                                 │
        │    score = α × answer_match  +  β × consistency               │
        │    where α = math_answer_weight  (default 0.63)                 │
        │          β = math_consistency_weight (default 0.27)            │
        │                                                                 │
        │  Note: consensus_weight (γ = 0.10) is applied separately in    │
        │  score_batch(), blended AFTER individual scores are computed.   │
        │                                                                 │
        │  FORMULA (fallback — no gold answer):                           │
        │                                                                 │
        │    score = consistency                                          │
        └─────────────────────────────────────────────────────────────────┘

        SIGNAL DEFINITIONS:

          answer_match  ∈ {0.0, 1.0}
            Binary: 1 if the predicted number equals the gold answer within
            the hybrid tolerance (see _gold_match). 0 otherwise.

          arithmetic_consistency  ∈ [0.0, 1.0]
            Fraction of arithmetic expressions "LHS = RHS" in the trace where
            evaluating LHS (via safe AST recursion) matches the stated RHS.
            This replaces the old step-chaining heuristic, which measured whether
            intermediate results were numerically similar to each other — a proxy
            that had no theoretical connection to arithmetic correctness.

        WHY 0.63 / 0.27 WEIGHTING (renormalised from 0.70 / 0.30):

          The remaining 0.10 is allocated to the cross-chain consensus signal
          (applied in score_batch). The within-trace ratio is preserved: answer
          correctness still dominates (0.63 vs 0.27), and consistency provides
          a secondary process-quality signal.

        FALLBACK (no gold answer):
          During sampling we haven't run evaluation yet. We return consistency
          alone so the QUBO can still prefer structurally sound reasons over
          chaotic ones, even without an oracle.
        """
        # ── Step 1: Arithmetic consistency (AST-verified expression correctness) ──────
        # Extract all "LHS = RHS" statements; verify each via safe AST evaluation.
        # consistency = fraction of expressions where computed LHS matches stated RHS.
        expressions = self._evaluate_expressions_ast(reason)
        if expressions:
            consistency = sum(1 for _, _, _, ok in expressions if ok) / len(expressions)
        else:
            # No verifiable arithmetic found — neutral fallback
            consistency = 0.5

        # ── Step 2: Answer match (outcome quality) ─────────────────────────────
        if expected_answer is not None:
            gold_num = self._extract_last_number(expected_answer)

            # Try to find the predicted answer in the reason text first.
            # If not present, fall back to the separately parsed answer field.
            pred_num = self._extract_last_number(reason)
            if pred_num is None and predicted_answer:
                pred_num = self._extract_last_number(predicted_answer)

            match = self._gold_match(pred_num, gold_num)

            # Weighted combination: correctness dominates (α), consistency supports (β).
            # The γ consensus weight is applied in score_batch() across the full sample pool.
            return self.math_answer_w * match + self.math_consist_w * consistency

        # No gold available — consistency is the only signal we can compute.
        return consistency

    # =========================================================================
    # ─── LANGUAGE SCORING ────────────────────────────────────────────────────
    # =========================================================================

    def verify_commonsense(
        self, reason: str, answer: str, question: str = ""
    ) -> float:
        """
        Score a language/commonsense reasoning trace in [0, 1].

        ┌─────────────────────────────────────────────────────────────────┐
        │  THREE-SIGNAL COMPOSITE FORMULA:                                │
        │                                                                 │
        │    score = α_nli × P(entailment)                               │
        │          + α_cov × lexical_coverage(reason, question)          │
        │          + α_str × structural_completeness(reason)             │
        │                                                                 │
        │  Defaults: α_nli=0.60, α_cov=0.25, α_str=0.15                 │
        │  (configurable in config.yaml → verifier.scoring)              │
        └─────────────────────────────────────────────────────────────────┘

        ── SIGNAL 1: NLI Entailment  (weight 0.60 — DOMINANT) ───────────

          Model: cross-encoder/nli-deberta-v3-base (seq-classification, 86M params)
          Input: (reason as premise, answer as hypothesis)
          Output: P(entailment) — probability that the reason LOGICALLY SUPPORTS
                  the answer.

          WHY NLI AS THE PRIMARY SIGNAL:
          An NLI model trained on large-scale natural language inference corpora
          has learned the logical distinction between:
            - entailment:     reason guarantees the answer is true
            - neutral:        reason is consistent with but does not prove the answer
            - contradiction:  reason contradicts the answer

          We want entailment, not just "not contradiction". A reason like
          "The sky is blue" is neutral to the answer "Paris is in France" —
          it doesn't contradict it, but it provides no logical support.
          The NLI model correctly scores this near 0 for entailment.

          WHY 60% WEIGHT:
          It is the strongest signal but not infallible. NLI models can be
          fooled by surface-level lexical overlap (keyword matching without
          genuine reasoning). The coverage and structure signals act as
          complementary guards against this failure mode.

        ── SIGNAL 2: Lexical Coverage  (weight 0.25) ─────────────────────

          Fraction of the question's KEY CONTENT WORDS that appear in the reason.
          Content words = alphabetic, non-stopword, ≥3-character tokens.
          Numeric tokens are excluded (see _lexical_coverage for full rationale).

          WHY 25% WEIGHT:
          A necessary (but not sufficient) condition for quality: a reason
          that ignores the question's key concepts is likely hallucinating.
          Coverage doesn't tell us if the reasoning is correct, but it screens
          out off-topic responses before they get a high entailment score.

          EXAMPLE: "Why do birds migrate south in winter?"
            Reason: "The sky is often cloudy in December."  (NLI might be fooled
                     by the winter-related sentence, but coverage=0/4 → low score)
            Reason: "Birds migrate south to escape cold winter temperatures."
                     (coverage ≈ 3/4 = 0.75 → signals on-topic content)

        ── SIGNAL 3: Structural Completeness  (weight 0.15) ─────────────

          Sigmoid of sentence count (see _structural_completeness for formula).
          A single-sentence commonsense reason scores ≈0.38; a three-sentence
          reason scores ≈0.82.

          WHY NEEDED FOR COMMONSENSE (but not math):
          Commonsense reasoning questions (BBH, StrategyQA) typically require
          causal chains: "X happens BECAUSE Y leads to Z". A single sentence
          almost certainly skips these intermediate steps. Brevity is NOT a
          virtue for commonsense reasoning in the same way it can be for math.

          WHY ONLY 15%:
          Sentence count is a noisy proxy. A well-articulated single sentence
          can be perfectly valid. We include this as a SMALL NUDGE, not a
          dominant factor.

        WORKED EXAMPLE (StrategyQA — "Do penguins live closer to the North or South Pole?"):
          answer = "South Pole"

          Reason A: "Penguins are native to Antarctica, which is located at
                     the South Pole, far from the Arctic."
            NLI ≈ 0.87, Coverage = {penguins,located,south,pole,arctic} ∩ {penguins,live,north,south,pole} = 3/5 = 0.60
            Structure (2 clauses ≈ 2 sentences) → ≈0.62
            score = 0.60×0.87 + 0.25×0.60 + 0.15×0.62 ≈ 0.52 + 0.15 + 0.09 = 0.76 ✓

          Reason B: "Penguins like cold weather."
            NLI ≈ 0.35, Coverage = {penguins,cold,weather} → {penguins} = 1/5 = 0.20
            Structure (1 sentence) → ≈0.38
            score = 0.60×0.35 + 0.25×0.20 + 0.15×0.38 ≈ 0.21 + 0.05 + 0.06 = 0.32 ✗
        """
        # ── Signal 1: NLI Entailment ──────────────────────────────────────
        #
        # We encode the (reason, answer) pair into the cross-encoder NLI model.
        # PREMISE  = the reason  (what we have as supporting evidence)
        # HYPOTHESIS = the answer (what we're checking is entailed)
        #
        # The model outputs logits over [contradiction, entailment, neutral].
        # Softmax converts these to probabilities; we take P(entailment).
        inputs = self.nli_tokenizer(
            reason, answer, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        # _entail_idx was resolved from model.config.label2id at init time,
        # so this is robust to model substitution (not hardcoded to index 1).
        nli_score = probs[0][self._entail_idx].item()

        # ── Signal 2: Lexical Coverage ────────────────────────────────────
        # Only computable when we have the original question string.
        # If no question is provided (e.g., called from legacy callers that
        # don't pass it), we default to 0.5 (neutral — no bonus, no penalty).
        coverage = self._lexical_coverage(reason, question) if question else 0.5

        # ── Signal 3: Structural Completeness ─────────────────────────────
        structure = self._structural_completeness(reason)

        # ── Composite score ───────────────────────────────────────────────
        score = (
            self.lang_nli_w   * nli_score
            + self.lang_cov_w * coverage
            + self.lang_struct_w * structure
        )
        # Clip to [0, 1] for safety (floating-point rounding should never
        # push outside bounds, but we be defensive).
        return float(np.clip(score, 0.0, 1.0))

    # =========================================================================
    # ─── DISPATCHER & BATCH API ───────────────────────────────────────────────
    # =========================================================================

    # -------------------------------------------------------------------------
    # Fix #1: Self-consistency scoring (Wang et al., 2023)
    # -------------------------------------------------------------------------

    def _normalise_answer_for_consensus(self, answer: str, task_type: str) -> str:
        """
        Normalise an extracted answer string for cross-chain comparison.

        MATH TASKS:
          Return the raw string; numeric comparison is handled by _gold_match
          rather than string equality, so we keep the original for the caller.
          (This method is only used for string-based consensus in language tasks;
          math consensus uses _gold_match directly.)

        LANGUAGE / COMMONSENSE TASKS:
          MCQ answers like "Option A", "the answer is B", "(C)" all refer to
          the same choice. We extract just the bare single letter.

          Strategy:
            1. Search for a standalone A–E letter using \b([A-Ea-e])\b.
            2. If found, return it lower-cased ("a", "b", … "e").
            3. Otherwise, fall back to lower-cased, whitespace-stripped string.

          This covers:
            "A"            → "a"
            "Option A"     → "a"
            "the answer is A" → "a"
            "(B)"          → "b"
            "Paris"        → "paris"  (open-ended, no letter extracted)
        """
        if task_type == "math":
            return answer.strip()
        # Language tasks: try to extract a bare MCQ letter first
        m = re.search(r'\b([A-Ea-e])\b', answer)
        if m:
            return m.group(1).lower()
        return answer.strip().lower()

    def compute_self_consistency(
        self, samples: list[dict], task_type: str = "math"
    ) -> list[dict]:
        """
        Compute cross-chain majority-vote agreement and write `consensus_score`
        into each sample dict.

        FORMULA (Wang et al., 2023):
            consensus(i) = Σ_{j≠i} 1[ŷ_j = ŷ_i] / (N − 1)

        consensus(i) is the fraction of OTHER chains that extracted the same
        answer as chain i. If 15 out of 20 chains agree, consensus(i) = 14/19.

        COMPARISON STRATEGY:
          Math tasks:     numeric equality via _gold_match (hybrid tolerance).
          Language tasks: string equality after _normalise_answer_for_consensus,
                          which extracts bare MCQ letter labels and lower-cases.

        EDGE CASES:
          N = 1:  No cross-chain comparison possible — consensus_score = 0.5 (neutral).
          answer is empty string: that chain contributes 0 to others' consensus
            (an empty answer never matches a non-empty one, and two empty answers
            don't get credit for "agreeing" — they simply have no signal).
        """
        n = len(samples)
        if n <= 1:
            for s in samples:
                s["consensus_score"] = 0.5
            return samples

        # Pre-extract normalised answers for all chains
        # Fall back to the reason text when the answer field is empty. Chains are
        # sampled with a token cap and frequently get truncated before they emit
        # the "Answer:" line, leaving answer="". The math branch below already
        # falls back to `reason`; without the same fallback here every language
        # chain compared as empty, consensus came out 0 for the whole pool, and
        # gold-free scoring lost its single strongest signal.
        norm_answers = [
            self._normalise_answer_for_consensus(
                s.get("answer", "") or s.get("reason", ""), task_type
            )
            for s in samples
        ]
        # For math: also extract numeric value for tolerance-based comparison
        num_answers: list[Optional[float]] = None
        if task_type == "math":
            num_answers = [self._extract_last_number(s.get("answer", "") or s.get("reason", ""))
                           for s in samples]

        for i, sample in enumerate(samples):
            if n == 1:
                sample["consensus_score"] = 0.5
                continue

            agree_count = 0
            for j in range(n):
                if j == i:
                    continue
                if task_type == "math":
                    # Use numeric tolerance — more robust than string comparison for floats
                    ai = num_answers[i]
                    aj = num_answers[j]
                    if ai is not None and aj is not None:
                        agree_count += int(self._gold_match(ai, aj) > 0.5)
                    # If either answer is None (couldn't extract a number), no agreement
                else:
                    # Language: string comparison after normalisation
                    a_i = norm_answers[i]
                    a_j = norm_answers[j]
                    # Both must be non-empty for a valid comparison
                    if a_i and a_j and a_i == a_j:
                        agree_count += 1

            sample["consensus_score"] = agree_count / (n - 1)

        return samples

    def verify(
        self,
        reason: str,
        task_type: str = "math",
        answer: Optional[str] = None,
        gold: Optional[str] = None,
        question: str = "",
    ) -> float:
        """
        Route to the appropriate scorer based on task type and return a [0, 1] score.

        task_type = "math"  → verify_math  (answer_match + step-chaining consistency)
        task_type = other   → verify_commonsense  (NLI + coverage + structure)

        Parameters
        ----------
        reason      : the generated reasoning trace to score
        task_type   : "math" or any commonsense benchmark identifier
        answer      : the model's predicted final answer (for math gold matching)
        gold        : the ground-truth correct answer string
        question    : the original question (used for lexical coverage in commonsense)
        """
        if task_type == "math":
            return self.verify_math(
                reason,
                expected_answer=gold,
                predicted_answer=answer,
            )
        else:
            return self.verify_commonsense(reason, answer or "", question=question)

    def score_breakdown(
        self,
        reason: str,
        task_type: str = "math",
        answer: Optional[str] = None,
        gold: Optional[str] = None,
        question: str = "",
    ) -> dict:
        """
        Return the individual signal values that compose the final quality score.

        Unlike verify() which returns a single float, this method returns a dict
        exposing every intermediate signal so you can explain exactly WHY a
        specific reason scored the way it did.

        Useful for:
          • Debugging — "why did this good-looking reason score 0.2?"
          • Mentor demos — show which signal drove a low score
          • Ablation — compare variants with different signal weights

        EXAMPLE OUTPUT (math task):
            {
              "task_type": "math",
              "answer_match": 1.0,
              "arithmetic_consistency": 0.667,
              "num_operations": 3,
              "final_score": 0.9
            }

        EXAMPLE OUTPUT (commonsense task):
            {
              "task_type": "commonsense",
              "nli_entailment": 0.82,
              "lexical_coverage": 0.6,
              "structural_completeness": 0.62,
              "sentence_count": 2,
              "final_score": 0.751
            }
        """
        breakdown: dict = {"task_type": task_type}

        if task_type == "math":
            # Compute each math signal independently
            operations = self._extract_arithmetic(reason)
            consistency = self._step_chaining_consistency(operations)
            breakdown["arithmetic_consistency"] = round(consistency, 4)
            breakdown["num_operations"] = len(operations)

            if gold is not None:
                gold_num = self._extract_last_number(gold)
                pred_num = self._extract_last_number(reason)
                if pred_num is None and answer:
                    pred_num = self._extract_last_number(answer)
                match = self._gold_match(pred_num, gold_num)
                breakdown["answer_match"] = match
                final = self.math_answer_w * match + self.math_consist_w * consistency
            else:
                final = consistency

            breakdown["final_score"] = round(float(final), 4)

        else:
            # Compute each language signal independently (avoids double NLI inference
            # compared to calling verify_commonsense() and then re-running separately)
            inputs = self.nli_tokenizer(
                reason, answer or "", return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                outputs = self.nli_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            nli_score = probs[0][self._entail_idx].item()

            coverage  = self._lexical_coverage(reason, question) if question else 0.5
            structure = self._structural_completeness(reason)

            # Count sentences for the breakdown (same logic as _structural_completeness)
            sentences = re.split(r"(?<=[.!?])\s+", reason.strip())
            sentences = [s for s in sentences if len(s.strip()) > 5]

            breakdown["nli_entailment"]         = round(nli_score, 4)
            breakdown["lexical_coverage"]        = round(coverage,  4)
            breakdown["structural_completeness"] = round(structure,  4)
            breakdown["sentence_count"]          = len(sentences)

            final = (
                self.lang_nli_w   * nli_score
                + self.lang_cov_w * coverage
                + self.lang_struct_w * structure
            )
            breakdown["final_score"] = round(float(np.clip(final, 0.0, 1.0)), 4)

        return breakdown

    def score_batch(
        self,
        samples: list[dict],
        task_type: str = "math",
        gold: Optional[str] = None,
        question: str = "",
    ) -> list[dict]:
        """
        Score a list of reasoning trace dicts and attach 'correctness_score' to each.

        The 'correctness_score' field is what qubo_builder.py reads:
            Q[i][i] = -sample['correctness_score'] + diversity_bonus

        SCORING PIPELINE:
          1. Call verify() for each sample to get the base per-trace score
             (answer_match + arithmetic_consistency for math;
              NLI + coverage + structure for language).
          2. Call compute_self_consistency() to compute cross-chain agreement
             and write consensus_score into each sample.
          3. Blend consensus into the final correctness_score:
               math:     score = α·match_score + γ·consensus
               language: score = base_score·(1−γ) + γ·consensus
             where γ = consensus_weight (default 0.10).

        Parameters
        ----------
        samples   : list of dicts, each with at least 'reason' and optionally 'answer'
        task_type : "math" or commonsense benchmark name
        gold      : ground-truth answer string (shared across the batch)
        question  : original question string (used for commonsense coverage signal)
        """
        # ── Step 1: Per-trace scoring ───────────────────────────────────────
        for sample in samples:
            base = self.verify(
                sample["reason"],
                task_type=task_type,
                answer=sample.get("answer"),
                gold=gold,
                question=question,
            )
            sample["_base_score"] = base  # store separately before consensus blend

        # ── Step 2: Cross-chain self-consistency ───────────────────────────
        # Computes consensus_score for each sample based on cross-chain agreement.
        self.compute_self_consistency(samples, task_type=task_type)

        # ── Step 3: Blend consensus into final correctness_score ──────────────
        #
        # Two regimes, distinguished by whether an answer key was supplied:
        #
        #  ORACLE (gold is not None) — curating TRAINING data. answer_match is
        #    already folded into base_score and legitimately dominates; consensus
        #    is a small corrective (γ = consensus_weight, default 0.10).
        #
        #  GOLD-FREE (gold is None) — EVALUATION or deployment. base_score holds
        #    process quality only (arithmetic consistency / NLI composite), which
        #    is a weak correctness signal on its own, so cross-chain consensus
        #    carries most of the weight. Passing gold here would leak the answer
        #    key into chain selection and inflate reported accuracy.
        gold_free = gold is None

        for sample in samples:
            base_score = sample.pop("_base_score")  # remove temporary key
            consensus  = sample["consensus_score"]

            if gold_free:
                blended = (
                    self.gold_free_consistency_w * base_score
                    + self.gold_free_consensus_w * consensus
                )
            else:
                gamma = self.consensus_w  # γ = consensus_weight from config
                # Blend: downweight the base score by (1-γ) and add γ·consensus.
                # This preserves the relative structure of base scores while
                # pulling high-consensus chains up regardless of individual score.
                blended = (1.0 - gamma) * base_score + gamma * consensus

            sample["correctness_score"] = float(np.clip(blended, 0.0, 1.0))

        return samples
