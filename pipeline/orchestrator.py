"""End-to-end flow matching the pipeline diagram.

query + initial prompt
  -> SLM generates multiple paths
  -> QUBO mapping + Quantum solver
  -> Select the best reasoning
  -> generate final prompt
  -> QLoRA SFT (adapters on Qwen 3B)
  -> SLM Qwen 3B + LoRA adapters
  -> final answer
  -> Outputs directory
  -> re-run benchmark / re-train QLoRA
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from pipeline.inference import InferencePipeline, compose_final_prompt
from pipeline.qubo_builder import QUBOBuilder
from pipeline.sampling import DiverseSampler
from pipeline.solver import make_solver
from pipeline.verifier import ReasonVerifier


def select_best_reasoning(
    state: np.ndarray, qubo_var_indices: list[int], n_samples: int, subset_size: int
) -> list[int]:
    selected = [qubo_var_indices[i] for i in range(len(state)) if int(state[i]) == 1]
    if not selected:
        selected = list(range(min(subset_size, n_samples)))
    return selected


class PRISMPipeline:
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        device: str | None = None,
        load_models: bool = True,
    ):
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.device = device
        self.lora_adapter = (
            self.config.get("model", {}).get("lora_adapter") or ""
        )
        self.sampler = None
        self.verifier = None
        self.qubo_builder = None
        self.solver = None
        self.inference = None
        if load_models:
            self._load_components()

    def _load_components(self):
        self.inference = InferencePipeline(
            self.config_path, device=self.device, lora_adapter=self.lora_adapter or None
        )
        runtime_device = str(self.inference.device)
        self.sampler = DiverseSampler(
            self.config_path,
            device=runtime_device,
            shared_model=self.inference.model,
            shared_tokenizer=self.inference.tokenizer,
        )
        self.verifier = ReasonVerifier(self.config_path, device=runtime_device)
        self.qubo_builder = QUBOBuilder(
            self.config_path,
            device=runtime_device,
            shared_embedder=self.inference.embedder,
        )
        self.solver = make_solver(self.config_path, device=runtime_device)

    def reload_with_adapter(self, adapter_path: str):
        self.lora_adapter = adapter_path
        self.config["model"]["lora_adapter"] = adapter_path
        self._load_components()

    def run_query(
        self,
        question: str,
        initial_prompt: str | None = None,
        gold: str = "",
        task_type: str = "math",
        is_mcq: bool = False,
    ) -> dict:
        if self.sampler is None:
            self._load_components()

        prompt = initial_prompt or self.config["pipeline"].get("initial_prompt")

        # 1-2. query + initial prompt -> SLM generates multiple paths
        samples = self.sampler.sample(question, task_type=task_type)
        if not samples:
            return {
                "question": question,
                "initial_prompt": prompt,
                "paths": [],
                "selected_indices": [],
                "selected_traces": [],
                "final_prompt": "",
                "answer": "",
                "gold": gold,
            }

        # 3. QUBO mapping + Quantum solver (verifier scores feed the Q diagonal)
        n = len(samples)
        print(f"  scoring {n} traces...")
        samples = self.verifier.score_batch(samples, task_type=task_type, gold=gold or None)
        print("  building QUBO...")
        Q, qubo_var_indices = self.qubo_builder.build_qubo(samples)
        print("  solving QUBO...")
        state, energy = self.solver.solve(Q)
        print(f"  QUBO solved (energy={energy:.4f})")

        # 4. Select the best reasoning
        selected_indices = select_best_reasoning(
            state, qubo_var_indices, len(samples), self.inference.subset_size
        )

        # 5. generate final prompt
        final_prompt, ordered_reasons = self.inference.prepare_final_prompt(
            question, selected_indices, samples, is_mcq=is_mcq
        )

        # 6-8. Qwen 3B + LoRA adapters -> final answer
        print("  generating final answer...")
        answer = self.inference.generate_answer(final_prompt)

        return {
            "question": question,
            "initial_prompt": prompt,
            "paths": samples,
            "qubo_energy": energy,
            "selected_indices": selected_indices,
            "selected_traces": ordered_reasons,
            "final_prompt": final_prompt,
            "answer": answer,
            "gold": gold,
        }

    def save_output(self, result: dict, output_dir: str = "outputs") -> Path:
        os.makedirs(output_dir, exist_ok=True)
        traces_path = Path(output_dir) / "traces.jsonl"
        record = {
            "question": result["question"],
            "initial_prompt": result.get("initial_prompt"),
            "selected_traces": result.get("selected_traces", []),
            "final_prompt": result.get("final_prompt", ""),
            "answer": result.get("answer", ""),
            "gold": result.get("gold", ""),
            "selected_indices": result.get("selected_indices", []),
            "path_count": len(result.get("paths") or []),
        }
        with open(traces_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return traces_path

    def collect_sft_traces(
        self,
        questions: list[str],
        golds: list[str],
        output_dir: str = "outputs",
        task_type: str = "math",
        is_mcq: bool = False,
        keep_only_correct: bool = True,
    ) -> list[dict]:
        traces = []
        for qi, (question, gold) in enumerate(zip(questions, golds), 1):
            print(f"[{qi}/{len(questions)}] {question[:80]}")
            result = self.run_query(
                question, gold=gold, task_type=task_type, is_mcq=is_mcq
            )
            self.save_output(result, output_dir=output_dir)
            answer = (result.get("answer") or "").strip()
            gold_s = (gold or "").strip()
            matched = bool(gold_s) and (
                gold_s.lower() in answer.lower() or answer.lower() in gold_s.lower()
            )
            if keep_only_correct and gold_s and not matched:
                continue
            if not result.get("selected_traces"):
                continue
            traces.append(
                {
                    "question": question,
                    "selected_traces": result["selected_traces"],
                    "correct_answer": gold or answer,
                    "final_prompt": result.get("final_prompt", ""),
                    "answer": answer,
                }
            )
        return traces

    def qlora_sft(
        self,
        traces: list[dict],
        run_name: Optional[str] = None,
        dry_run: bool = False,
    ) -> str:
        from training.sft import QUBOSFTTrainer

        trainer = QUBOSFTTrainer(self.config_path)
        dataset = trainer.prepare_dataset_from_pipeline(traces)
        name = run_name or f"qlora-sft-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if dry_run:
            out = os.path.join(trainer.output_dir, name, "final_adapter")
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "dry_run.json"), "w") as f:
                json.dump({"examples": len(traces), "run_name": name}, f)
            return out
        return trainer.train(dataset, run_name=name)

    def retrain_from_outputs(
        self,
        output_dir: str = "outputs",
        dry_run: bool = False,
        run_name: Optional[str] = None,
    ) -> str:
        traces = load_traces_for_sft(output_dir)
        if not traces:
            raise RuntimeError(f"No SFT traces found under {output_dir}")
        return self.qlora_sft(traces, run_name=run_name, dry_run=dry_run)

    def run_iterative_loop(
        self,
        questions: list[str],
        golds: list[str],
        output_dir: str = "outputs",
        num_rounds: Optional[int] = None,
        task_type: str = "math",
        is_mcq: bool = False,
        dry_run: bool = False,
    ) -> str:
        rounds = num_rounds or self.config.get("training", {}).get("iterative_rounds", 3)
        adapter_path = self.config.get("model", {}).get("lora_adapter") or ""

        for round_idx in range(rounds):
            round_dir = os.path.join(output_dir, f"round_{round_idx + 1}")
            os.makedirs(round_dir, exist_ok=True)
            print(f"\n=== Round {round_idx + 1}/{rounds}: generate paths + QUBO + SFT ===")
            traces = self.collect_sft_traces(
                questions,
                golds,
                output_dir=round_dir,
                task_type=task_type,
                is_mcq=is_mcq,
            )
            if not traces:
                traces = load_traces_for_sft(round_dir)
            if not traces:
                print("No traces for QLoRA. Stopping.")
                break
            adapter_path = self.qlora_sft(
                traces,
                run_name=f"qlora-sft-round-{round_idx + 1}",
                dry_run=dry_run,
            )
            if round_idx < rounds - 1 and not dry_run:
                self.reload_with_adapter(adapter_path)

        return adapter_path


def load_traces_for_sft(output_dir: str) -> list[dict]:
    traces = []
    root = Path(output_dir)
    if not root.exists():
        return traces

    for path in sorted(root.rglob("traces.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                selected = rec.get("selected_traces") or []
                gold = rec.get("gold") or rec.get("correct_answer") or rec.get("answer")
                if not selected or not rec.get("question") or not gold:
                    continue
                traces.append(
                    {
                        "question": rec["question"],
                        "selected_traces": selected,
                        "correct_answer": gold,
                    }
                )
    return traces


def run_one_query(
    sampler: DiverseSampler,
    verifier: ReasonVerifier,
    qubo_builder: QUBOBuilder,
    solver,
    inference: InferencePipeline,
    question: str,
    task_type: str = "math",
    gold: str = "",
    is_mcq: bool = False,
    initial_prompt: str | None = None,
) -> dict:
    samples = sampler.sample(question, task_type=task_type)
    if not samples:
        return {"answer": "", "selected_indices": [], "selected_traces": [], "paths": []}
    n = len(samples)
    print(f"  scoring {n} traces...")
    samples = verifier.score_batch(samples, task_type=task_type, gold=gold or None)
    print("  building QUBO...")
    Q, qubo_var_indices = qubo_builder.build_qubo(samples)
    print("  solving QUBO...")
    state, energy = solver.solve(Q)
    print(f"  QUBO solved (energy={energy:.4f})")
    selected_indices = select_best_reasoning(
        state, qubo_var_indices, len(samples), inference.subset_size
    )
    final_prompt, ordered_reasons = inference.prepare_final_prompt(
        question, selected_indices, samples, is_mcq=is_mcq
    )
    print("  generating final answer...")
    answer = inference.generate_answer(final_prompt)
    return {
        "answer": answer,
        "paths": samples,
        "selected_indices": selected_indices,
        "selected_traces": ordered_reasons,
        "final_prompt": final_prompt,
        "qubo_energy": energy,
    }


def check_flow() -> None:
    """Runnable check: QUBO + quantum solver + select + final prompt. No SLM load."""
    Q = np.array(
        [
            [-1.0, 0.8, 0.1],
            [0.8, -0.2, 0.7],
            [0.1, 0.7, -0.9],
        ]
    )
    solver = make_solver()
    state, energy = solver.solve(Q)
    assert state.shape == (3,), state
    assert set(state.tolist()) <= {0, 1}, state
    selected = select_best_reasoning(state, [0, 1, 2], 3, subset_size=2)
    assert selected, selected
    prompt = compose_final_prompt(
        "What is 2+2?",
        ["Add two and two.", "2+2 equals 4."],
        subset_size=6,
    )
    assert "Question: What is 2+2?" in prompt, prompt
    assert "reasoning steps" in prompt

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "traces.jsonl"
        path.write_text(
            json.dumps(
                {
                    "question": "What is 2+2?",
                    "selected_traces": ["2+2=4"],
                    "gold": "4",
                }
            )
            + "\n"
        )
        loaded = load_traces_for_sft(tmp)
        assert len(loaded) == 1 and loaded[0]["correct_answer"] == "4", loaded

    print(f"flow check ok | solver={type(solver).__name__} state={state.tolist()} energy={energy:.4f}")


if __name__ == "__main__":
    check_flow()
