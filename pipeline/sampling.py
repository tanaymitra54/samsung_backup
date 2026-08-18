"""
==============================================================================
FILE: pipeline/sampling.py
ROLE: Diverse Reasoning Trace Sampler
BRANCH ADDITION (abhyuday):
  1. KV Caching Activation: Enabled `use_cache=True` during candidate sampling to
     significantly accelerate reasoning trace generation.
  2. Candidate Pool Adjustment: Refined default candidate pool count parameters.
==============================================================================

pipeline/sampling.py  —  Diverse Reasoning Trace Sampler
=========================================================

ROLE IN THE PIPELINE
─────────────────────
The QUBO selection stage (qubo_builder.py) needs a LARGE, DIVERSE POOL of
candidate reasoning traces to select from. The bigger and more varied the
pool, the better the chance that the pool contains at least a few high-quality,
non-redundant reasons that the solver can pick.

This module generates that pool by sampling from the language model with two
forms of deliberate diversity:

  1. PROMPT PERTURBATIONS (perturb_prompt):
     The same mathematical/factual question is wrapped in 4 different prompt
     framings ("step by step", "think carefully", "work through logically",
     "break this down"). LLMs are highly sensitive to prompt framing — the
     same model given the same question but different system instructions
     often produces qualitatively different reasoning paths. This increases
     the structural diversity of the candidate pool.

  2. TEMPERATURE RANDOMISATION:
     For each perturbation, we sample `num_answers` completions at a randomly
     chosen temperature from [temperature_range_low, temperature_range_high].

     WHY RANDOM TEMPERATURE:
     • Low temperature (≈0.3): the model is more deterministic and confident —
       produces focused, consistent reasoning but low diversity.
     • High temperature (≈0.9): the model is more exploratory — produces
       creative/diverse reasoning but occasionally incoherent.
     • Randomising temperature across samples gets the best of both worlds:
       some samples are reliable anchors, others explore different paths.

     The verifier's quality score (correctness_score) later filters out the
     incoherent high-temperature samples, so the risk of including bad traces
     is managed by the QUBO scoring step, not the sampling step.

OUTPUT FORMAT:
  Each sampled trace is a dict with keys:
    'reason'           : the reasoning text (all lines except the answer line)
    'answer'           : the final answer line (extracted heuristically)
    'diversity_score'  : placeholder (0.0 at sampling time; computed later)
    'temperature'      : the sampling temperature used (for diagnostics)
    'prompt_template'  : which perturbation was used (for diagnostics)
"""

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

from pipeline.device_utils import resolve_device


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
        # Use config value directly, capped at max_new_tokens
        self.sampling_max_new_tokens = min(pipe_cfg.get("sampling_max_new_tokens", 256), self.max_new_tokens)
        self.sampling_max_input_tokens = pipe_cfg.get("sampling_max_input_tokens", 768)

    def _build_model_kwargs(self, model_cfg: dict, load_in_4bit: bool) -> dict:
        model_kwargs = {
            "cache_dir": model_cfg.get("cache_dir"),
            "low_cpu_mem_usage": True,
        }
        if self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"
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

    def generate_batch(
        self, prompt: str, temperature: float, n: int
    ) -> list[str]:
        """Generate `n` completions for one prompt in a single forward batch.

        WHY THIS EXISTS: the pool is 4 prompt framings x num_answers samples, and
        issuing those as individual batch-1 generate() calls left the GPU almost
        idle -- sampling measured at 90% of total benchmark wall time (21 of 23
        minutes on an H100). num_return_sequences produces the whole group in one
        call, so the pool costs 4 calls instead of 12 and each one actually fills
        the device.

        TEMPERATURE: all `n` sequences in a group share this temperature, so the
        pool carries one temperature per framing rather than one per sample.
        Diversity within a group still comes from do_sample=True drawing
        different tokens. Across the pool that is 4 distinct temperatures instead
        of 12, which is the intended axis of variation anyway -- framing is the
        structured axis, temperature is noise on top.

        Falls back to sequential single-sample generation if the batch OOMs.
        """
        chat_prompt = self._apply_chat_template(prompt)
        retry_limits = [
            (self.sampling_max_input_tokens, self.sampling_max_new_tokens),
            (min(self.sampling_max_input_tokens, 512), min(self.sampling_max_new_tokens, 128)),
        ]

        for max_input_tokens, max_new_tokens in retry_limits:
            inputs = None
            outputs = None
            try:
                inputs = self.tokenizer(
                    chat_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_tokens,
                ).to(self.device)
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        use_cache=True,
                        num_return_sequences=n,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                prompt_len = inputs["input_ids"].shape[1]
                return [
                    self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
                    for seq in outputs
                ]
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                gc.collect()
            finally:
                if inputs is not None:
                    del inputs
                if outputs is not None:
                    del outputs
                torch.cuda.empty_cache()
                gc.collect()

        # Batched path exhausted its retries -- fall back to one at a time.
        print("[Sampler] Batched generation OOMed; falling back to sequential.", flush=True)
        return [
            self.generate_with_contrastive_decode(prompt, temperature=temperature)
            for _ in range(n)
        ]

    def generate_with_contrastive_decode(
        self, prompt: str, temperature: float, alpha: float = 0.1
    ) -> str:
        chat_prompt = self._apply_chat_template(prompt)
        retry_limits = [
            (self.sampling_max_input_tokens, self.sampling_max_new_tokens),
            (min(self.sampling_max_input_tokens, 512), min(self.sampling_max_new_tokens, 32)),
            (min(self.sampling_max_input_tokens, 384), 16),
        ]

        for max_input_tokens, max_new_tokens in retry_limits:
            inputs = None
            outputs = None
            try:
                inputs = self.tokenizer(
                    chat_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_tokens,
                ).to(self.device)
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=self.top_p,
                        do_sample=True,
                        use_cache=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                generated = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                return generated.strip()
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                torch.cuda.empty_cache()
                gc.collect()
            finally:
                if inputs is not None:
                    del inputs
                if outputs is not None:
                    del outputs
                torch.cuda.empty_cache()
                gc.collect()

        return ""

    def _parse_reason_answer(self, text: str):
        lines = text.strip().split("\n")
        answer = ""
        reason = text
        
        for i in range(len(lines) - 1, -1, -1):
            line = lines[i]
            if line.strip().lower().startswith("answer:"):
                answer = line.split(":", 1)[1].strip()
                reason_lines = lines[:i] + lines[i+1:]
                reason = "\n".join(reason_lines)
                return reason.strip(), answer.strip()

        for line in lines:
            if "answer" in line.lower() or "therefore" in line.lower():
                answer = line
                reason_lines = [l for l in lines if l != line]
                reason = "\n".join(reason_lines)
                break
        return reason.strip(), answer.strip()

    def perturb_prompt(self, question: str, task_type: str = "math") -> list[str]:
        """
        Wrap the question in 4 structurally distinct prompt framings.

        WHY STRUCTURALLY DIVERSE PROMPTS:
        The original 4 templates ("Let's solve step by step", "Think carefully and
        reason step by step", "Work through this problem logically", "Break this down")
        all elicit the same forward-reasoning strategy. LLMs with identical training
        on any of these prompts will produce highly correlated outputs — the candidate
        pool ends up with 4 near-identical reasoning paths, reducing the value of
        QUBO selection.

        The new templates induce qualitatively different reasoning strategies:

        1. FORWARD (anchor)
           "Starting from the given information, work forward step by step to reach
           the answer."
           → Elicits the standard chain-of-thought: start with facts, derive conclusion.
             This is the most reliable template for small models. The "given information"
             framing nudges the model to enumerate knowns before computing.

        2. BACKWARD (monitor — weak on 3B-scale models)
           "Start from what the answer must satisfy and work backward to verify it
           from the given facts."
           → Elicits goal-directed reasoning. Theoretically powerful for constraint
             satisfaction and multi-step deduction, but 3B-scale models often produce
             circular or incoherent outputs when asked to reason backward.
           MONITOR: If chains from this template consistently score near zero in the
           verifier, disable it for that dataset by removing it from this list.

        3. ANALOGICAL
           "This problem is similar to one where you identify a pattern or analogy.
           Use that to reason through it."
           → Elicits pattern-matching and structural reasoning. Useful when the problem
             has a common structure (e.g., ratio problems, sequence problems). Less
             reliable for novel problem types.

        4. UNITS-FIRST (math) / BREAK-IT-DOWN (commonsense)
           Math:        "Identify the units and quantities involved, then compute step
                         by step."
           Commonsense: "Break this down and solve."
           → For math, the units-first framing anchors dimensional analysis before
             arithmetic, reducing unit-conversion errors. For commonsense, the
             "break it down" framing encourages enumeration of sub-problems.
             This template is task-type-aware.

        CALL SIGNATURE:
          task_type = "math"      → uses units-first variant for template 4
          task_type = anything else → uses commonsense variant for template 4
        """
        suffix = "\nPlease put your final answer on a new line starting with 'Answer:'"
        # Template 4: task-type-aware
        if task_type == "math":
            template_4 = (
                f"Identify the units and quantities involved, then compute step by step.\n"
                f"Question: {question}{suffix}"
            )
        else:
            template_4 = f"Break this down and solve.\nQuestion: {question}{suffix}"

        perturbations = [
            # Template 1: Forward reasoning (anchor)
            f"Starting from the given information, work forward step by step to reach the answer.\nQuestion: {question}{suffix}",
            # Template 2: Backward reasoning (monitor on small models)
            f"Start from what the answer must satisfy and work backward to verify it from the given facts.\nQuestion: {question}{suffix}",
            # Template 3: Analogical reasoning
            f"This problem is similar to one where you identify a pattern or analogy. Use that to reason through it.\nQuestion: {question}{suffix}",
            # Template 4: Units-first (math) or break-it-down (commonsense)
            template_4,
        ]
        return perturbations


    def sample(self, question: str, task_type: str = "math") -> list[dict]:
        """
        Generate a diverse pool of (reason, answer) pairs for a single question.

        TOTAL SAMPLES GENERATED:
            len(perturbations) × num_answers  (default: 4 × 3 = 12 candidates)

        Each sample is independently drawn at a random temperature drawn from
        [temperature_range[0], temperature_range[1]] (e.g., 0.3 to 0.9).
        This gives every sample a different exploration/exploitation trade-off.

        After sampling, the calling pipeline calls verifier.score_batch() to
        assign correctness_score to each sample, then qubo_builder.build_qubo()
        to select the best non-redundant subset.

        NOTE: 'diversity_score' is initialised to 0.0 here. It is not used in
        the current QUBO formulation (diversity is handled implicitly by the
        off-diagonal redundancy penalty). The field is preserved for potential
        future use (e.g., explicit diversity pre-filtering before QUBO).
        """
        all_samples = []
        perturbations = self.perturb_prompt(question, task_type=task_type)

        for prompt_temp in perturbations:
            # One temperature per framing group, drawn from the configured range.
            # The whole group is produced in a single batched call (see
            # generate_batch) rather than num_answers separate generate() calls.
            temp = random.uniform(
                self.config["pipeline"]["temperature_range"][0],
                self.config["pipeline"]["temperature_range"][1],
            )
            generations = self.generate_batch(
                prompt_temp, temperature=temp, n=self.num_answers
            )
            for generated in generations:
                # Parse the raw generated text into a structured (reason, answer) pair.
                # The reason is the reasoning chain; the answer is the final conclusion.
                reason, answer = self._parse_reason_answer(generated)
                all_samples.append({
                    "reason": reason,
                    "answer": answer,
                    "diversity_score": 0.0,   # populated later if needed
                    "temperature": temp,       # retained for diagnostics / ablations
                    "prompt_template": prompt_temp,  # retained for diagnostics
                    "task_type": task_type,    # retained for diagnostics
                })
        return all_samples

