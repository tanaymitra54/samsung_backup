import torch
import yaml
import random
import numpy as np
from pathlib import Path
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer


class DiverseSampler:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config["model"]
        pipe_cfg = self.config["pipeline"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["name"],
            cache_dir=model_cfg.get("cache_dir"),
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            cache_dir=model_cfg.get("cache_dir"),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        ).to(self.device)
        self.model.eval()

        self.num_answers = pipe_cfg["num_answers"]
        self.num_reasons = pipe_cfg["num_reasons"]
        self.max_new_tokens = pipe_cfg["max_new_tokens"]
        self.top_p = pipe_cfg["top_p"]

    def generate_with_contrastive_decode(
        self, prompt: str, temperature: float, alpha: float = 0.1
    ) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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
