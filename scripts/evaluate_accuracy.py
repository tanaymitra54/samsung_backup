import argparse
import os
import random
import sys
import time
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.sampling import DiverseSampler
from pipeline.verifier import ReasonVerifier
from pipeline.qubo_builder import QUBOBuilder
from pipeline.solver import SimulatedAnnealingSolver
from pipeline.inference import InferencePipeline


QUESTIONS_WITH_ANSWERS = [
    ("If x + y = 10 and x - y = 4, find the values of x and y.", ["7", "x=7,y=3", "x = 7, y = 3", "x=7, y=3"]),
    ("A train travels 120 km in 2 hours. If it speeds up by 20 km/h, what is the new speed?", ["80", "80 km/h", "80 kmh"]),
    ("A rectangle has perimeter 36 cm and its length is twice its width. Find its area.", ["72", "72 cm^2", "72 cm²"]),
    ("Solve for z: 3z - 7 = 2z + 5.", ["12", "z=12", "z = 12"]),
    ("A shop sells apples for $0.50 each and oranges for $0.75 each. If John buys 4 apples and 3 oranges, how much does he pay?", ["4.25", "$4.25", "4.25 dollars"]),
    ("All cats are mammals. Some mammals are predators. Can we conclude all cats are predators?", ["no", "No", "cannot", "Cannot"]),
]


def normalize_answer(text: str) -> str:
    return text.strip().lower().rstrip(".")


def is_correct(prediction: str, valid_answers: list[str]) -> bool:
    pred = normalize_answer(prediction)
    for ans in valid_answers:
        if pred == normalize_answer(ans):
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Run quick local accuracy harness")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(repo_root, "config", "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg.get("model", {}).get("name", "unknown-model")
    print(f"Model: {model_name}")

    print("Initializing pipeline components...")
    sampler = DiverseSampler()
    verifier = ReasonVerifier()
    qubo_builder = QUBOBuilder()
    solver = SimulatedAnnealingSolver()
    inference = InferencePipeline()

    correct = 0
    total = len(QUESTIONS_WITH_ANSWERS)
    rows = []

    for q_idx, (question, valid_answers) in enumerate(QUESTIONS_WITH_ANSWERS, 1):
        print(f"\n{'='*60}")
        print(f"Q{q_idx}: {question[:70]}...")

        t_start = time.time()
        prediction = ""
        verdict = False
        energy = 0.0
        error = ""

        try:
            print("  Sampling...")
            samples = sampler.sample(question)
            if not samples:
                raise RuntimeError("No samples generated")
            print(f"  {len(samples)} samples generated")

            print("  Verifying...")
            samples = verifier.score_batch(samples, task_type="math")

            print("  Building QUBO...")
            Q, qubo_var_indices = qubo_builder.build_qubo(samples)
            print(f"  QUBO: {Q.shape[0]}x{Q.shape[0]}")

            print("  Solving QUBO...")
            state, energy = solver.solve(Q)
            selected_indices = [qubo_var_indices[i] for i in range(len(state)) if state[i] == 1]
            if len(selected_indices) < 1:
                selected_indices = [0]
            print(f"  Selected {len(selected_indices)} reasons, energy={energy:.4f}")

            print("  Generating final answer...")
            prediction = inference.run(question, selected_indices, samples)
            print(f"  Prediction: {prediction}")
            verdict = is_correct(prediction, valid_answers)
        except Exception as e:
            error = str(e)
            print(f"  ERROR: {error}")

        elapsed = time.time() - t_start

        if verdict:
            correct += 1
            print(f"  CORRECT ({elapsed:.1f}s)")
        else:
            expected = valid_answers[0]
            print(f"  WRONG ({elapsed:.1f}s) — expected: {expected}")

        rows.append({
            "id": q_idx,
            "question": question,
            "prediction": prediction,
            "expected": valid_answers[0],
            "correct": int(verdict),
            "energy": round(energy, 4),
            "runtime_s": round(elapsed, 3),
            "error": error,
        })

    print(f"\n{'='*60}")
    print(f"Accuracy: {correct}/{total} = {100*correct/total:.1f}%")

    if args.output:
        import csv
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [
                "id", "question", "prediction", "expected", "correct", "energy", "runtime_s", "error"
            ])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
