import os
import json
import yaml
import torch
from typing import Optional
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


class QUBOSFTTrainer:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config["model"]
        train_cfg = self.config["training"]

        self.model_name = model_cfg["name"]
        self.cache_dir = model_cfg.get("cache_dir")
        self.load_in_4bit = model_cfg.get("load_in_4bit", True)
        self.attn_implementation = model_cfg.get("attn_implementation")

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

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _build_bnb_config(self):
        if not self.load_in_4bit or self.device != "cuda":
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
            "device_map": "auto" if self.device == "cuda" else None,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
        }
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        if self.attn_implementation and self.device == "cuda":
            model_kwargs["attn_implementation"] = self.attn_implementation

        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, cache_dir=self.cache_dir
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        return model, tokenizer

    def _format_trace_as_chat(self, question: str, traces: list[str], answer: str) -> str:
        reasoning = "\n".join(f"{i+1}. {t}" for i, t in enumerate(traces))
        return (
            f"Question: {question}\n\n"
            f"Reasoning steps:\n{reasoning}\n\n"
            f"Answer: {answer}"
        )

    def prepare_dataset_from_pipeline(
        self, pipeline_results: list[dict]
    ) -> Dataset:
        formatted = []
        for item in pipeline_results:
            text = self._format_trace_as_chat(
                item["question"],
                item["selected_traces"],
                item["correct_answer"],
            )
            formatted.append({"text": text})
        return Dataset.from_list(formatted)

    def train(
        self,
        dataset: Dataset,
        run_name: Optional[str] = "qubo-sft-run",
        resume_from_checkpoint: Optional[str] = None,
    ):
        if self.device != "cuda":
            raise RuntimeError("SFT training requires a CUDA-capable GPU (H100 recommended).")

        model, tokenizer = self._build_model_and_tokenizer()

        if self.load_in_4bit:
            model = prepare_model_for_kbit_training(model)

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

        training_args = TrainingArguments(
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
            remove_unused_columns=False,
            report_to="wandb" if self.config.get("wandb_project") else None,
            run_name=run_name,
            dataloader_num_workers=2,
            ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            max_seq_length=self.max_seq_length,
            dataset_text_field="text",
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model(os.path.join(run_dir, "final_adapter"))

        return run_dir

    def iterative_train(
        self,
        pipeline_data_generator,
        num_rounds: Optional[int] = None,
    ):
        rounds = num_rounds or self.iterative_rounds
        current_model_name = self.model_name

        for round_idx in range(rounds):
            print(f"\n{'='*60}")
            print(f"Iterative SFT Round {round_idx + 1}/{rounds}")
            print(f"{'='*60}")

            print("  Generating QUBO-selected traces with current model...")
            traces = pipeline_data_generator(current_model_name)

            if not traces:
                print("  No traces generated. Stopping early.")
                break

            print(f"  Collected {len(traces)} training examples.")
            dataset = self.prepare_dataset_from_pipeline(traces)

            run_name = f"qubo-sft-round-{round_idx + 1}"
            adapter_path = self.train(dataset, run_name=run_name)

            if round_idx < rounds - 1:
                current_model_name = adapter_path

        print(f"\nIterative training complete. Final adapter: {current_model_name}")
