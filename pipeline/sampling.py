import torch
import yaml
import random
import gc
import numpy as np
from pathlib import Path
from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from pipeline.device_utils import hf_device_map_value, resolve_device


class DiverseSampler:
    def __init__(
        self,
        config_path: str = "config/config.yaml",
        device: str | None = None,
        shared_model=None,
        shared_tokenizer=None,
    ):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config["model"]
        pipe_cfg = self.config["pipeline"]

        preferred_device = device or self.config.get("evaluation", {}).get("device")
        self.device = resolve_device(preferred_device)
        if shared_model is not None and shared_tokenizer is not None:
            self.model = shared_model
            self.tokenizer = shared_tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_cfg["name"],
                cache_dir=model_cfg.get("cache_dir"),
                padding_side="left",
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            load_in_4bit = model_cfg.get("load_in_4bit", False)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_cfg["name"], **self._build_model_kwargs(model_cfg, load_in_4bit)
                )
            except torch.cuda.OutOfMemoryError:
                if self.device.type != "cuda" or load_in_4bit:
                    raise
                torch.cuda.empty_cache()
                gc.collect()
                print("CUDA OOM while loading sampler model, retrying with 4-bit quantization...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_cfg["name"], **self._build_model_kwargs(model_cfg, True)
                )
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
        self.model.eval()

        self.num_answers = pipe_cfg["num_answers"]
        self.num_reasons = pipe_cfg["num_reasons"]
        self.max_new_tokens = pipe_cfg["max_new_tokens"]
        self.top_p = pipe_cfg["top_p"]

    def _build_model_kwargs(self, model_cfg: dict, load_in_4bit: bool) -> dict:
        model_kwargs = {
            "cache_dir": model_cfg.get("cache_dir"),
            "low_cpu_mem_usage": True,
        }
        if self.device.type == "cuda":
            model_kwargs["device_map"] = {"": hf_device_map_value(self.device)}
            model_kwargs["torch_dtype"] = torch.float16
            if load_in_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(
                        torch, model_cfg.get("bnb_4bit_compute_dtype", "float16")
                    ),
                    bnb_4bit_use_double_quant=model_cfg.get("bnb_4bit_use_double_quant", True),
                    bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
                )
                model_kwargs["quantization_config"] = bnb_config
                model_kwargs["torch_dtype"] = getattr(
                    torch, model_cfg.get("bnb_4bit_compute_dtype", "float16")
                )
        else:
            model_kwargs["torch_dtype"] = torch.float32
        return model_kwargs

    def _apply_chat_template(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": "user", "content": prompt},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def generate_with_contrastive_decode(
        self, prompt: str, temperature: float, alpha: float = 0.1
    ) -> str:
        chat_prompt = self._apply_chat_template(prompt)
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return generated.strip()

    def _parse_reason_answer(self, text: str):
        lines = text.strip().split("\n")
        answer = ""
        reason = text
        for line in lines:
            if "answer" in line.lower() or "therefore" in line.lower():
                answer = line
                reason_lines = [l for l in lines if l != line]
                reason = "\n".join(reason_lines)
                break
        return reason.strip(), answer.strip()

    def perturb_prompt(self, question: str) -> list[str]:
        perturbations = [
            f"Let's solve this step by step.\nQuestion: {question}",
            f"Think carefully and reason step by step.\nQuestion: {question}",
            f"Work through this problem logically.\nQuestion: {question}",
            f"Break this down and solve.\nQuestion: {question}",
        ]
        return perturbations

    def sample(self, question: str) -> list[dict]:
        all_samples = []
        perturbations = self.perturb_prompt(question)

        for prompt_temp in perturbations:
            for _ in range(self.num_answers):
                temp = random.uniform(
                    self.config["pipeline"]["temperature_range"][0],
                    self.config["pipeline"]["temperature_range"][1],
                )
                generated = self.generate_with_contrastive_decode(
                    prompt_temp, temperature=temp
                )
                reason, answer = self._parse_reason_answer(generated)
                all_samples.append({
                    "reason": reason,
                    "answer": answer,
                    "diversity_score": 0.0,
                    "temperature": temp,
                    "prompt_template": prompt_temp,
                })
        return all_samples
