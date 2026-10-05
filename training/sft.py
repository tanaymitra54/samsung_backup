import argparse
import gc
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback
from trl import SFTConfig, SFTTrainer

from training.progress_ui import PipelineProgress

from evaluation import BenchmarkRunner
from evaluation.answer_utils import (
    extract_gsm8k_gold,
    extract_predicted_answer,
    is_correct_prediction,
)
from pipeline.inference import InferencePipeline
from pipeline.qubo_builder import QUBOBuilder
from pipeline.reasoning import run_reasoning_pipeline
from pipeline.sampling import DiverseSampler
from pipeline.solver import SimulatedAnnealingSolver
from pipeline.verifier import ReasonVerifier


class _QLoRAProgress(TrainerCallback):
    def __init__(self, progress: PipelineProgress):
        self.progress = progress

    def on_train_begin(self, args, state, control, **kwargs):
        total = state.max_steps or 1
        self.progress.start_stage("QLoRA training", total, "Fine-tuning the adapter on the selected prompts.")

    def on_step_end(self, args, state, control, **kwargs):
        total = state.max_steps or max(state.global_step, 1)
        self.progress.train_step(state.global_step, total)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        total = state.max_steps or max(state.global_step, 1)
        self.progress.train_step(state.global_step, total, float(logs["loss"]))


TASK_TYPE = {
    "gsm8k": "math",
    "bbh": "math",
    "strategyqa": "commonsense",
    "mmlu": "commonsense",
    "arc_challenge": "commonsense",
}


class QUBOSFTTrainer:
    """QLoRA loop around the QUBO reasoning pipeline.

    Each round samples traces, scores them without the gold answer, selects a
    subset with simulated annealing, and builds the same final prompt used at
    inference. QLoRA trains on that prompt. The saved adapter is the model for
    the next round and for the benchmark re-run.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config["model"]
        train_cfg = self.config["training"]
        eval_cfg = self.config.get("evaluation", {})

        self.model_name = model_cfg["name"]
        self.cache_dir = model_cfg.get("cache_dir")
        self.attn_implementation = model_cfg.get("attn_implementation")
        self.qlora = train_cfg.get("qlora", True)

        self.sft_epochs = train_cfg["sft_epochs"]
        self.learning_rate = train_cfg["learning_rate"]
        self.batch_size = train_cfg["batch_size"]
        self.lora_rank = train_cfg["lora_rank"]
        self.lora_alpha = train_cfg.get("lora_alpha", 32)
        self.lora_dropout = train_cfg.get("lora_dropout", 0.05)
        self.lora_target_modules = train_cfg.get(
            "lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.output_dir = train_cfg["output_dir"]
        self.max_seq_length = train_cfg.get("max_seq_length", 2048)
        self.warmup_steps = train_cfg.get("warmup_steps", 100)
        self.iterative_rounds = train_cfg.get("iterative_rounds", 3)
        self.benchmark = train_cfg.get("benchmark", "gsm8k")
        self.trace_split = train_cfg.get("trace_split", "train")
        self.trace_examples = train_cfg.get("trace_examples", eval_cfg.get("subset_size", 200))
        self.eval_examples = train_cfg.get("eval_examples", eval_cfg.get("subset_size", 200))
        self.outputs_dir = train_cfg.get("outputs_dir", "./outputs/qlora_rounds")

        if not torch.cuda.is_available():
            raise RuntimeError("This pipeline run requires an NVIDIA GPU, and CUDA is not available.")
        self.device = "cuda:0"
        self.gpu_name = torch.cuda.get_device_name(0)
        free_bytes, _ = torch.cuda.mem_get_info(0)
        self.free_gpu_mib = free_bytes / (1024 ** 2)
        self.inference_4bit = self.free_gpu_mib < 12000
        if self.inference_4bit:
            self.batch_size = 1
            self.max_seq_length = min(self.max_seq_length, 1024)
        self.progress = PipelineProgress()
        self.progress.set_gpu(
            f"{self.gpu_name} on {self.device} | {self.free_gpu_mib:.0f} MiB free"
        )

    def _build_bnb_config(self):
        if not self.qlora or not str(self.device).startswith("cuda"):
            return None
        model_cfg = self.config["model"]
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(
                torch, model_cfg.get("bnb_4bit_compute_dtype", "float16")
            ),
            bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        )

    def _build_model_and_tokenizer(self):
        bnb_config = self._build_bnb_config()
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "device_map": "auto" if str(self.device).startswith("cuda") else None,
            "torch_dtype": torch.float16 if str(self.device).startswith("cuda") else torch.float32,
        }
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        if self.attn_implementation and str(self.device).startswith("cuda"):
            model_kwargs["attn_implementation"] = self.attn_implementation

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return model, tokenizer

    def _target_answer(self, gold: str, benchmark: str) -> str:
        if benchmark == "gsm8k":
            return extract_gsm8k_gold(gold)
        return gold.strip()

    def prepare_dataset_from_pipeline(self, pipeline_results: list[dict]) -> Dataset:
        formatted = []
        for item in pipeline_results:
            prompt = item["final_prompt"].rstrip()
            answer = item["correct_answer"].strip()
            formatted.append({"text": f"{prompt} {answer}".strip()})
        return Dataset.from_list(formatted)

    def train(
        self,
        dataset: Dataset,
        run_name: Optional[str] = "qubo-sft-run",
        resume_adapter: Optional[str] = None,
    ) -> str:
        if not str(self.device).startswith("cuda"):
            raise RuntimeError("QLoRA training requires a CUDA-capable GPU.")

        model, tokenizer = self._build_model_and_tokenizer()
        if self.qlora:
            model = prepare_model_for_kbit_training(model)

        if resume_adapter:
            model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
            peft_config = None
        else:
            peft_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

        run_dir = os.path.join(self.output_dir, run_name or "default")
        os.makedirs(run_dir, exist_ok=True)
        adapter_dir = os.path.join(run_dir, "final_adapter")

        training_args = SFTConfig(
            output_dir=run_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=2,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            num_train_epochs=self.sft_epochs,
            fp16=True,
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
            report_to="wandb" if self.config.get("wandb_project") else "none",
            run_name=run_name,
            dataloader_num_workers=0,
            dataset_text_field="text",
            max_length=self.max_seq_length,
            packing=False,
            gradient_checkpointing=self.qlora,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
            callbacks=[_QLoRAProgress(self.progress)],
        )
        trainer.train()
        trainer.save_model(adapter_dir)
        del trainer, model, tokenizer
        self._release()
        return adapter_dir

    def _build_stack(self, adapter_path: Optional[str]):
        self.progress.note(f"Loading models onto {self.gpu_name} ({self.device})")
        inference = InferencePipeline(
            self.config_path,
            device=self.device,
            adapter_path=adapter_path,
            load_in_4bit=self.inference_4bit,
        )
        runtime_device = str(inference.device)
        sampler = DiverseSampler(
            self.config_path,
            device=runtime_device,
            shared_model=inference.model,
            shared_tokenizer=inference.tokenizer,
        )
        verifier = ReasonVerifier(self.config_path, device=runtime_device)
        if self.inference_4bit and verifier.device.type == "cuda":
            verifier.nli_model = verifier.nli_model.half()
        qubo_builder = QUBOBuilder(self.config_path, device=runtime_device)
        solver = SimulatedAnnealingSolver(self.config_path, device=runtime_device)
        return inference, sampler, verifier, qubo_builder, solver

    def _release(self, *objects):
        for obj in objects:
            for attr in ("model", "nli_model", "embedder"):
                if hasattr(obj, attr):
                    setattr(obj, attr, None)
        del objects
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_split(self, benchmark: str, split: str, limit: Optional[int]):
        if benchmark == "gsm8k":
            hf_split = "train" if split == "train" else "test"
            dataset = load_dataset("openai/gsm8k", "main", split=hf_split)
            if limit:
                dataset = dataset.select(range(min(limit, len(dataset))))
            questions = [item["question"] for item in dataset]
            golds = [extract_gsm8k_gold(item["answer"]) for item in dataset]
            return questions, golds

        runner = BenchmarkRunner(self.config_path)
        if limit:
            runner.subset_size = limit
            runner.full_eval = False
        questions, golds = runner.load_benchmark(benchmark)
        if limit:
            questions = questions[:limit]
            golds = golds[:limit]
        return questions, golds

    def _run_pipeline_split(
        self,
        questions: list[str],
        golds: list[str],
        benchmark: str,
        adapter_path: Optional[str],
        generate_answer: bool,
    ) -> list[dict]:
        task_type = TASK_TYPE.get(benchmark, "math")
        inference, sampler, verifier, qubo_builder, solver = self._build_stack(adapter_path)
        records = []
        self.progress.start_stage(
            "Benchmark answers" if generate_answer else "Select training traces",
            len(questions),
        )
        try:
            for index, (question, gold) in enumerate(zip(questions, golds), start=1):
                result = run_reasoning_pipeline(
                    sampler,
                    verifier,
                    qubo_builder,
                    solver,
                    inference,
                    question,
                    task_type=task_type,
                    generate_answer=generate_answer,
                )
                if not result or not result["selected_traces"]:
                    self.progress.question_done({
                        "index": index,
                        "result": "SKIPPED",
                        "gold": self._target_answer(gold, benchmark),
                        "prediction": "",
                        "question": " ".join(question.split())[:90],
                    })
                    continue
                prediction = result["predicted_answer"]
                target = self._target_answer(gold, benchmark)
                correct = False
                if generate_answer and prediction:
                    if benchmark == "gsm8k":
                        correct = is_correct_prediction(
                            extract_predicted_answer(prediction), target
                        )
                    else:
                        correct = target.lower() in prediction.lower()
                short_question = " ".join(question.split())[:90]
                result_label = "CORRECT" if correct else "WRONG" if generate_answer else "SELECTED"
                self.progress.question_done({
                    "index": index,
                    "result": result_label,
                    "gold": target,
                    "prediction": prediction if generate_answer else "",
                    "question": short_question,
                })
                records.append({
                    "id": index,
                    "question": question,
                    "selected_traces": result["selected_traces"],
                    "final_prompt": result["final_prompt"],
                    "predicted_answer": prediction,
                    "correct_answer": target,
                    "correct": int(correct),
                    "energy": result["energy"],
                })
        finally:
            self._release(inference, sampler, verifier, qubo_builder, solver)
        return records

    def _write_jsonl(self, path: str, rows: list[dict]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def run(
        self,
        num_rounds: Optional[int] = None,
        limit: Optional[int] = None,
        benchmark: Optional[str] = None,
    ) -> str:
        rounds = num_rounds or self.iterative_rounds
        benchmark = benchmark or self.benchmark
        trace_limit = limit or self.trace_examples
        eval_limit = limit or self.eval_examples
        adapter_path = None

        if self.inference_4bit:
            self.progress.note(
                f"{self.gpu_name} has {self.free_gpu_mib:.0f} MiB free, so the model runs in 4-bit on {self.device}."
            )
        for round_idx in range(rounds):
            round_no = round_idx + 1
            round_dir = os.path.join(self.outputs_dir, f"round_{round_no}")
            print(f"\n{'=' * 60}")
            print(f"Round {round_no}/{rounds}  adapter={adapter_path or 'base model'}")
            print(f"{'=' * 60}")

            print("  Sampling, scoring, and selecting training traces...")
            questions, golds = self._load_split(benchmark, self.trace_split, trace_limit)
            train_records = self._run_pipeline_split(
                questions, golds, benchmark, adapter_path, generate_answer=False
            )
            self._write_jsonl(os.path.join(round_dir, "train_traces.jsonl"), train_records)
            if not train_records:
                print("  No selected traces. Stopping.")
                break

            self.progress.note(f"QLoRA on {len(train_records)} final prompts using {self.gpu_name}.")
            dataset = self.prepare_dataset_from_pipeline(train_records)
            adapter_path = self.train(
                dataset,
                run_name=f"qubo-sft-round-{round_no}",
                resume_adapter=adapter_path,
            )

            print("  Re-running the pipeline with the QLoRA adapter...")
            eval_questions, eval_golds = self._load_split(benchmark, "test", eval_limit)
            eval_records = self._run_pipeline_split(
                eval_questions, eval_golds, benchmark, adapter_path, generate_answer=True
            )
            self._write_jsonl(os.path.join(round_dir, "benchmark.jsonl"), eval_records)
            accuracy = (
                sum(row["correct"] for row in eval_records) / len(eval_records)
                if eval_records else 0.0
            )
            summary = {
                "round": round_no,
                "benchmark": benchmark,
                "adapter": adapter_path,
                "train_examples": len(train_records),
                "eval_examples": len(eval_records),
                "accuracy": accuracy,
            }
            with open(os.path.join(round_dir, "summary.json"), "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)
            print(f"  Benchmark accuracy: {accuracy:.2%}")

        print(f"\nFinished. Final adapter: {adapter_path}")
        return adapter_path or ""


def main():
    parser = argparse.ArgumentParser(
        description="Run the QUBO pipeline, then QLoRA, and repeat with the adapter."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--benchmark", default=None)
    args = parser.parse_args()
    trainer = QUBOSFTTrainer(args.config)
    trainer.run(num_rounds=args.rounds, limit=args.limit, benchmark=args.benchmark)


if __name__ == "__main__":
    main()
