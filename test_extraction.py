#!/usr/bin/env python3
"""Test the extraction logic"""
import sys
sys.path.insert(0, '.')
from evaluation.answer_utils import extract_predicted_answer

test_cases = [
    # (model_output, expected_answer, description)
    (" $18.\n\nHere is the step-by-step breakdown:", "18", "Answer at start with $"),
    ("18\n\nLet me explain...", "18", "Direct number"),
    (" 70000 dollars", "70000", "Number with word"),
    ("The answer is 540 meters", "540", "Answer in sentence"),
    ("... 3 eggs... 4 muffins... final: 18", "18", "Multiple numbers, want last"),
]

print("Testing extraction logic:")
print("="*70)

all_pass = True
for output, expected, desc in test_cases:
    extracted = extract_predicted_answer(output)
    status = "✓" if extracted == expected else "✗"
    if extracted != expected:
        all_pass = False
    print(f"{status} {desc}")
    print(f"  Input: {output[:50]}...")
    print(f"  Expected: {expected}, Got: {extracted}")
    if extracted != expected:
        print(f"  ❌ FAIL")
    print()

if all_pass:
    print("="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)
else:
    print("="*70)
    print("✗ SOME TESTS FAILED")
    print("="*70)
    sys.exit(1)
