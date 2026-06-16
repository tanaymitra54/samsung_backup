import gc
import yaml
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from pipeline.device_utils import candidate_cuda_devices, resolve_device


class InferencePipeline:
    def __init__(self, config_path: str = "config/config.yaml", device: str | None = None, use_vllm: bool | None = None):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config["model"]
        pipe_cfg = self.config["pipeline"]

        preferred_device = device or self.config.get("evaluation", {}).get("device")
        self.device = resolve_device(preferred_device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["name"], cache_dir=model_cfg.get("cache_dir")
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.use_vllm = model_cfg.get("use_vllm", False) if use_vllm is None else use_vllm

        if self.use_vllm:
            self.model = None
        else:
            load_in_4bit = model_cfg.get("load_in_4bit", False)
            self.model = self._load_model_with_fallbacks(model_cfg, load_in_4bit)
            if self.device.type == "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            self.model.generation_config.top_p = None
            self.model.generation_config.top_k = None
            self.model_input_device = self._get_model_input_device()
            self.generation_input_device = self._get_generation_input_device()

        self.subset_size = pipe_cfg["subset_size"]
        self.max_new_tokens = pipe_cfg["max_new_tokens"]
        self.fallback_max_new_tokens = min(pipe_cfg.get("fallback_max_new_tokens", 128), self.max_new_tokens)
        self.fallback_max_input_tokens = pipe_cfg.get("fallback_max_input_tokens", 1024)
        embedder_device = self.config.get("qubo", {}).get("embedder_device") or str(self.device)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=embedder_device)

    def _build_model_kwargs(self, model_cfg: dict, load_in_4bit: bool) -> dict:
        model_kwargs = {
            "cache_dir": model_cfg.get("cache_dir"),
            "low_cpu_mem_usage": True,
        }
        if self.device.type == "cuda":
            model_kwargs["device_map"] = "auto"
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
                model_kwargs["torch_dtype"] = torch.float16
            attn_impl = model_cfg.get("attn_implementation")
            if attn_impl:
                model_kwargs["attn_implementation"] = attn_impl
        else:
            model_kwargs["torch_dtype"] = torch.float32
        return model_kwargs

    def _load_model_with_fallbacks(self, model_cfg: dict, load_in_4bit: bool):
        candidate_devices = [self.device]
        if self.device.type == "cuda":
            candidate_devices = candidate_cuda_devices(str(self.device))

        attempted = []
        for candidate in candidate_devices:
            for quantized in ([load_in_4bit] if load_in_4bit else [False, True]):
                self.device = candidate
                attempted.append(f"{candidate}|4bit={quantized}")
                try:
                    if candidate != candidate_devices[0] or quantized != load_in_4bit:
                        print(f"Retrying main model load on {candidate} with 4-bit={quantized}...")
                    return AutoModelForCausalLM.from_pretrained(
                        model_cfg["name"], **self._build_model_kwargs(model_cfg, quantized)
                    )
                except torch.cuda.OutOfMemoryError:
                    if candidate.type != "cuda":
                        raise
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue

        raise RuntimeError(
            "Failed to load model on any single GPU. Attempted: " + ", ".join(attempted)
        )

    def _get_model_input_device(self):
        if self.use_vllm or self.model is None:
            return self.device
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            return next(self.model.parameters()).device

    def _get_generation_input_device(self):
        if self.use_vllm or self.model is None:
            return self.device
        return self.model_input_device

    def _rank_reasons_by_relevance(self, reasons: list[str], question: str) -> list[int]:
        reason_embs = self.embedder.encode(reasons, convert_to_numpy=True)
        query_emb = self.embedder.encode([question], convert_to_numpy=True)
        similarities = cosine_similarity(reason_embs, query_emb).flatten()
        return np.argsort(similarities)[::-1]

    def build_final_prompt(
        self, question: str, selected_reasons: list[str], is_mcq: bool = False
    ) -> str:
        K = min(self.subset_size, len(selected_reasons))
        top_reasons = selected_reasons[:K]

        prompt = "Here are some reasoning steps:\n"
        for i, reason in enumerate(top_reasons, 1):
            prompt += f"{i}. {reason}\n"
        if is_mcq:
            prompt += f"\nBased on these steps, output the correct answer choice (A, B, C, or D).\nQuestion: {question}\nAnswer:"
        else:
            prompt += f"\nBased on these steps, answer the following question.\nQuestion: {question}\nAnswer:"
        return prompt

    def _apply_chat_template(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": "user", "content": prompt},
            ]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return prompt

    def generate_answer(self, prompt: str) -> str:
        chat_prompt = self._apply_chat_template(prompt)
        if self.use_vllm:
            return self.generate_answers_vllm([chat_prompt])[0]

        retry_limits = [
            (2048, self.max_new_tokens),
            (self.fallback_max_input_tokens, self.fallback_max_new_tokens),
            (min(self.fallback_max_input_tokens, 768), min(self.fallback_max_new_tokens, 64)),
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
                ).to(self.generation_input_device)
                with torch.inference_mode():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        use_cache=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                answer = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                return answer.strip()
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

    def generate_answers_batch(self, prompts: list[str], batch_size: int = 1) -> list[str]:
        """Generate answers for multiple prompts with memory management.
        
        Args:
            prompts: List of prompts to process
            batch_size: Number of prompts to process at once (default 1 for maximum memory safety)
        
        Returns:
            List of generated answers
        """
        chat_prompts = [self._apply_chat_template(p) for p in prompts]
        if self.use_vllm:
            try:
                return self.generate_answers_vllm(chat_prompts)
            except ImportError:
                self.use_vllm = False
        
        all_answers = []
        
        # Process one prompt at a time for maximum memory safety
        for prompt_idx, chat_prompt in enumerate(chat_prompts):
            try:
                # Tokenize with truncation on CPU first
                inputs = self.tokenizer(
                    chat_prompt, 
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                ).to(self.generation_input_device)
                
                input_len = inputs["input_ids"].shape[1]
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        use_cache=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                
                # Decode answer
                ans = self.tokenizer.decode(
                    outputs[0][input_len:], skip_special_tokens=True
                )
                all_answers.append(ans.strip())
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    gc.collect()
                    # Return empty for this prompt
                    all_answers.append("")
                else:
                    raise
            finally:
                # Clean up after each prompt
                if 'inputs' in locals():
                    del inputs
                if 'outputs' in locals():
                    del outputs
                torch.cuda.empty_cache()
                gc.collect()
        
        return all_answers

    def generate_answers_vllm(self, prompts: list[str]) -> list[str]:
        from vllm import LLM, SamplingParams
        if not hasattr(self, "_vllm_model"):
            model_cfg = self.config["model"]
            vllm_cfg = model_cfg.get("vllm", {})
            self._vllm_model = LLM(
                model=model_cfg["name"],
                dtype=vllm_cfg.get("dtype", "auto"),
                quantization=vllm_cfg.get("quantization"),
                tensor_parallel_size=vllm_cfg.get("tensor_parallel_size", 1),
                gpu_memory_utilization=vllm_cfg.get("gpu_memory_utilization", 0.9),
            )
        params = SamplingParams(
            temperature=0.3,
            top_p=0.95,
            max_tokens=self.config["pipeline"]["max_new_tokens"],
        )
        outputs = self._vllm_model.generate(prompts, params)
        return [o.outputs[0].text.strip() for o in outputs]

    def run(
        self, question: str, selected_indices: list[int], samples: list[dict], is_mcq: bool = False
    ) -> str:
        selected_reasons = [samples[i]["reason"] for i in selected_indices]
        ranked_order = self._rank_reasons_by_relevance(selected_reasons, question)
        ordered_reasons = [selected_reasons[i] for i in ranked_order]

        final_prompt = self.build_final_prompt(question, ordered_reasons, is_mcq=is_mcq)
        return self.generate_answer(final_prompt)
