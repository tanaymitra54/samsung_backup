"""
scripts/test_verifier.py  —  Sanity-check the ReasonVerifier scoring signals

USAGE:
    python scripts/test_verifier.py

PURPOSE:
    Runs the verifier on hand-crafted examples (math AND commonsense) and
    prints a breakdown of every individual signal. Use this to:

    1. Confirm the scoring is behaving as expected after code changes.
    2. Demonstrate to your mentor EXACTLY which signal caused a score to
       be high or low (using the score_breakdown() method).
    3. Spot-check edge cases (no operations found, no question provided, etc.)
"""

import sys
import json
from pathlib import Path

# Make sure the project root is on the path when running from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.verifier import ReasonVerifier


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def show(label: str, breakdown: dict) -> None:
    """Pretty-print a score_breakdown() result."""
    print(f"\n  [{label}]")
    for k, v in breakdown.items():
        print(f"    {k:30s}: {v}")


def main():
    print("Loading ReasonVerifier (this loads the NLI model — may take a moment)...")
    verifier = ReasonVerifier()
    print("Loaded.\n")

    # =========================================================================
    # MATH SCORING TESTS
    # =========================================================================
    section("MATH SCORING — verify_math()")

    print("""
  FORMULA:  score = 0.70 × answer_match + 0.30 × arithmetic_consistency
  Where:
    answer_match         = 1.0 if predicted ≈ gold (hybrid tolerance), else 0.0
    arithmetic_consistency = fraction of steps whose result feeds into a later step
    """)

    # ── Test M1: Perfect chain, correct answer ────────────────────────────
    reason_m1 = (
        "First, I compute 3 × 4 = 12. "
        "Then I add 12 + 5 = 17. "
        "So the answer is 17."
    )
    bd = verifier.score_breakdown(reason_m1, task_type="math", gold="17")
    show("M1 — Perfect chain, correct answer (expect ~1.0)", bd)
    # Expected: answer_match=1.0, consistency≈1.0 (12 is used in the next step)

    # ── Test M2: Disconnected steps, correct answer ───────────────────────
    reason_m2 = (
        "We know 6 × 7 = 42. "
        "Also 9 + 3 = 12. "
        "The final answer is 17."
    )
    bd = verifier.score_breakdown(reason_m2, task_type="math", gold="17")
    show("M2 — Disconnected steps, correct answer (expect ~0.70)", bd)
    # Expected: answer_match=1.0, consistency=0.0 (neither 42 nor 12 is reused)
    # score = 0.70*1.0 + 0.30*0.0 = 0.70

    # ── Test M3: Correct chain, WRONG answer ─────────────────────────────
    reason_m3 = (
        "First, 3 × 4 = 12. "
        "Then 12 + 5 = 17. "
        "So the answer is 99."
    )
    bd = verifier.score_breakdown(reason_m3, task_type="math", gold="17")
    show("M3 — Correct chain, WRONG answer (expect ~0.30)", bd)
    # Expected: answer_match=0.0 (99 ≠ 17), consistency≈1.0
    # score = 0.70*0.0 + 0.30*1.0 = 0.30

    # ── Test M4: No gold answer — consistency only ────────────────────────
    reason_m4 = "We compute 5 × 6 = 30. Then 30 + 10 = 40."
    bd = verifier.score_breakdown(reason_m4, task_type="math")
    show("M4 — No gold answer, good chain (expect ~1.0 consistency)", bd)

    # ── Test M5: Large-number tolerance (hybrid threshold) ────────────────
    # gold=1000, pred=1001 — should match (within 1%)
    reason_m5 = "There are 1001 total items."
    bd = verifier.score_breakdown(reason_m5, task_type="math", gold="1000")
    show("M5 — Large number: pred=1001, gold=1000 (expect answer_match=1.0)", bd)

    # ── Test M6: Large-number failure (outside tolerance) ─────────────────
    # gold=1000, pred=985 — should NOT match (1.5% off)
    reason_m6 = "There are 985 total items."
    bd = verifier.score_breakdown(reason_m6, task_type="math", gold="1000")
    show("M6 — Large number: pred=985, gold=1000 (expect answer_match=0.0)", bd)

    # =========================================================================
    # COMMONSENSE / LANGUAGE SCORING TESTS
    # =========================================================================
    section("COMMONSENSE SCORING — verify_commonsense()")

    print("""
  FORMULA:  score = 0.60 × P(entailment)  +  0.25 × coverage  +  0.15 × structure
  Where:
    P(entailment) = NLI model probability that reason → answer
    coverage      = fraction of question content-words appearing in reason
    structure     = sigmoid of sentence count (1 sent ≈ 0.38, 3+ sents ≈ 0.82)
    """)

    question_q1 = "Do penguins live closer to the North Pole or the South Pole?"
    answer_q1   = "South Pole"

    # ── Test L1: Strong, multi-sentence, on-topic reason ─────────────────
    reason_l1 = (
        "Penguins are native to Antarctica, which is located at the South Pole. "
        "They are well adapted to cold Antarctic climates and are never found "
        "near the Arctic or North Pole."
    )
    bd = verifier.score_breakdown(
        reason_l1, task_type="commonsense", answer=answer_q1, question=question_q1
    )
    show("L1 — Strong multi-sentence reason (expect high ≥ 0.70)", bd)

    # ── Test L2: Weak single-sentence, vague reason ───────────────────────
    reason_l2 = "Penguins like cold weather."
    bd = verifier.score_breakdown(
        reason_l2, task_type="commonsense", answer=answer_q1, question=question_q1
    )
    show("L2 — Weak single-sentence reason (expect low ≤ 0.45)", bd)

    # ── Test L3: Off-topic hallucination ─────────────────────────────────
    reason_l3 = (
        "The polar bear is the largest carnivore on land. "
        "It lives in the Arctic near the North Pole. "
        "It has adapted to extreme cold temperatures."
    )
    bd = verifier.score_breakdown(
        reason_l3, task_type="commonsense", answer=answer_q1, question=question_q1
    )
    show("L3 — Off-topic hallucination (expect very low — wrong animal/pole)", bd)

    # ── Test L4: No question provided (coverage defaults to 0.5) ─────────
    question_q2 = "Why do leaves change color in autumn?"
    answer_q2   = "Reduced chlorophyll production"
    reason_l4 = (
        "As days get shorter in autumn, trees produce less chlorophyll. "
        "Without green chlorophyll, the yellow and orange pigments in leaves "
        "become visible, causing the color change."
    )
    bd = verifier.score_breakdown(
        reason_l4, task_type="commonsense", answer=answer_q2, question=question_q2
    )
    show("L4 — Good reason with question provided (expect high)", bd)

    bd_no_q = verifier.score_breakdown(
        reason_l4, task_type="commonsense", answer=answer_q2, question=""
    )
    show("L4b — Same reason, NO question (coverage=0.5 neutral, expect slightly lower)", bd_no_q)

    # =========================================================================
    # BATCH SCORING
    # =========================================================================
    section("BATCH SCORING — score_batch()")

    samples = [
        {"reason": reason_m1, "answer": "17"},
        {"reason": reason_m2, "answer": "17"},
        {"reason": reason_m3, "answer": "99"},
    ]
    scored = verifier.score_batch(samples, task_type="math", gold="17")
    print("\n  Batch results (math, gold='17'):")
    for i, s in enumerate(scored):
        print(f"    Sample {i}: correctness_score = {s['correctness_score']:.4f}")

    print("\n  QUBO diagonal entries for these samples (diversity_bonus=0.5):")
    bonus = 0.5
    for i, s in enumerate(scored):
        q_ii = -s["correctness_score"] + bonus
        print(f"    Q[{i}][{i}] = -{s['correctness_score']:.4f} + {bonus} = {q_ii:.4f}")
    print("  (More negative Q[i][i] = solver prefers to select reason i)")

    print("\n" + "=" * 70)
    print("  All tests complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
