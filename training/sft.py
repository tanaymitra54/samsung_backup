import inspect
import os
from typing import Optional

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


class QUBOSFTTrainer:
    """QLoRA SFT for Qwen 3B on 2x H100 80GB (CUDA 12.2). Saves LoRA adapters."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config["model"]
        train_cfg = self.config["training"]

        self.model_name = model_cfg["name"]
        self.cache_dir = model_cfg.get("cache_dir")
        self.attn_implementation = model_cfg.get("attn_implementation")

        self.sft_epochs = int(train_cfg["sft_epochs"])
        self.learning_rate = float(train_cfg["learning_rate"])
        self.batch_size = int(train_cfg["batch_size"])
        self.grad_accum = int(train_cfg.get("gradient_accumulation_steps", 4))
        self.lora_rank = int(train_cfg["lora_rank"])
        self.lora_alpha = int(train_cfg.get("lora_alpha", 32))
        self.lora_dropout = float(train_cfg.get("lora_dropout", 0.05))
        self.lora_target_modules = train_cfg.get(
            "lora_target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.output_dir = train_cfg["output_dir"]
        self.max_seq_length = train_cfg.get("max_seq_length", 2048)
        self.warmup_steps = train_cfg.get("warmup_steps", 100)
        self.iterative_rounds = train_cfg.get("iterative_rounds", 3)
        self.merge_after_sft = train_cfg.get("merge_after_sft", False)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _build_bnb_config(self):
        if self.device != "cuda":
            return None
        model_cfg = self.config["model"]
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(
                torch, model_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
            ),
            bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
        )

    def _format_trace_as_chat(self, question: str, traces: list[str], answer: str) -> str:
        reasoning = "\n".join(f"{i+1}. {t}" for i, t in enumerate(traces))
        return (
            f"<|im_start|>user\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\nReasoning steps:\n{reasoning}\n\n"
            f"Answer: {answer}<|im_end|>"
        )

    def prepare_dataset_from_pipeline(self, pipeline_results: list[dict]) -> Dataset:
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
    ) -> str:
        if self.device != "cuda":
            raise RuntimeError(
                "QLoRA SFT needs CUDA. Hardware in config: 2x NVIDIA H100 80GB, CUDA 12.2."
            )

        bnb_config = self._build_bnb_config()
        bf16_supported = torch.cuda.is_bf16_supported()
        torch_dtype = torch.bfloat16 if bf16_supported else torch.float16

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch_dtype,
            attn_implementation=self.attn_implementation or "sdpa",
            cache_dir=self.cache_dir,
        )
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

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
        num_gpus = torch.cuda.device_count()

        training_args = TrainingArguments(
            output_dir=run_dir,
            per_device_train_batch_size=self.batch_size,
            gradient_accumulation_steps=self.grad_accum,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
            num_train_epochs=self.sft_epochs,
            fp16=not bf16_supported,
            bf16=bf16_supported,
            logging_steps=10,
            save_steps=200,
            save_total_limit=3,
            remove_unused_columns=False,
            report_to="wandb" if self.config.get("wandb_project") else "none",
            run_name=run_name,
            dataloader_num_workers=4,
            ddp_find_unused_parameters=False if num_gpus > 1 else None,
            gradient_checkpointing=True,
            optim="adamw_8bit",
            lr_scheduler_type="cosine",
        )

        params = inspect.signature(SFTTrainer.__init__).parameters
        sft_kwargs = {}
        if "processing_class" in params:
            sft_kwargs["processing_class"] = tokenizer
        elif "tokenizer" in params:
            sft_kwargs["tokenizer"] = tokenizer
        if "dataset_kwargs" in params:
            sft_kwargs["dataset_kwargs"] = {"text": "text", "max_seq_length": self.max_seq_length}
        elif "dataset_text_field" in params:
            sft_kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in params:
            sft_kwargs["max_seq_length"] = self.max_seq_length
        else:
            tokenizer.model_max_length = self.max_seq_length
        if "peft_config" in params:
            sft_kwargs["peft_config"] = peft_config

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            **sft_kwargs,
        )
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        adapter_path = os.path.join(run_dir, "final_adapter")
        trainer.save_model(adapter_path)
        tokenizer.save_pretrained(adapter_path)

        if self.merge_after_sft:
            del model, trainer
            torch.cuda.empty_cache()
            base = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                cache_dir=self.cache_dir,
            )
            merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()
            merged_path = os.path.join(run_dir, "merged_model")
            merged.save_pretrained(merged_path)
            tokenizer.save_pretrained(merged_path)

        return adapter_path

    def iterative_train(
        self,
        questions: list[str],
        golds: list[str],
        output_dir: str = "outputs",
        num_rounds: Optional[int] = None,
        dry_run: bool = False,
    ) -> str:
        from pipeline.orchestrator import PRISMPipeline

        pipe = PRISMPipeline(self.config_path)
        return pipe.run_iterative_loop(
            questions,
            golds,
            output_dir=output_dir,
            num_rounds=num_rounds or self.iterative_rounds,
            dry_run=dry_run,
        )
