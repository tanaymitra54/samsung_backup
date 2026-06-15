import gc
import yaml
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class InferencePipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config["model"]
        pipe_cfg = self.config["pipeline"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg["name"], cache_dir=model_cfg.get("cache_dir")
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.use_vllm = model_cfg.get("use_vllm", False)

        if self.use_vllm:
            self.model = None
        else:
            model_kwargs = {
                "cache_dir": model_cfg.get("cache_dir"),
                "device_map": "auto" if self.device == "cuda" else None,
            }
            if self.device == "cuda":
                if model_cfg.get("load_in_4bit"):
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
            self.model = AutoModelForCausalLM.from_pretrained(
                model_cfg["name"], **model_kwargs
            )
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            self.model.eval()

        self.subset_size = pipe_cfg["subset_size"]
        embedder_device = self.config.get("qubo", {}).get("embedder_device", self.device)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=embedder_device)

    def _rank_reasons_by_relevance(self, reasons: list[str], question: str) -> list[int]:
        reason_embs = self.embedder.encode(reasons, convert_to_numpy=True)
        query_emb = self.embedder.encode([question], convert_to_numpy=True)
        similarities = cosine_similarity(reason_embs, query_emb).flatten()
        return np.argsort(similarities)[::-1]

    def build_final_prompt(
        self, question: str, selected_reasons: list[str]
    ) -> str:
        K = min(self.subset_size, len(selected_reasons))
        top_reasons = selected_reasons[:K]

        prompt = "Here are some reasoning steps:\n"
        for i, reason in enumerate(top_reasons, 1):
            prompt += f"{i}. {reason}\n"
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
        
        # Tokenize with truncation to prevent OOM
        inputs = self.tokenizer(
            chat_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config["pipeline"]["max_new_tokens"],
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Clear GPU memory and retry
                torch.cuda.empty_cache()
                gc.collect()
                
                # Re-tokenize with truncation on CPU
                inputs = self.tokenizer(
                    chat_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=2048
                )
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config["pipeline"]["max_new_tokens"],
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            else:
                raise
        
        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        
        # Clean up
        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()
        
        return answer.strip()

    def generate_answers_batch(self, prompts: list[str], batch_size: int = 4) -> list[str]:
        """Generate answers for multiple prompts with memory management.
        
        Args:
            prompts: List of prompts to process
            batch_size: Number of prompts to process at once (default 4 to prevent OOM)
        
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
        
        # Process in smaller batches to avoid OOM
        for batch_idx in range(0, len(chat_prompts), batch_size):
            batch_prompts = chat_prompts[batch_idx:batch_idx + batch_size]
            
            # Tokenize with max_length to prevent excessive padding
            inputs = self.tokenizer(
                batch_prompts, 
                padding=True, 
                truncation=True,
                max_length=2048,
                return_tensors="pt"
            ).to(self.device)
            input_len = inputs["input_ids"].shape[1]
            
            try:
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.config["pipeline"]["max_new_tokens"],
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Try with even smaller batch on OOM
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Recursively call with smaller batch size
                    if batch_size > 1:
                        return self.generate_answers_batch(prompts, batch_size=max(1, batch_size // 2))
                    else:
                        # Single item OOM - try moving to CPU temporarily
                        inputs = self.tokenizer(
                            batch_prompts,
                            padding=True,
                            truncation=True,
                            max_length=2048,
                            return_tensors="pt"
                        )
                        with torch.no_grad():
                            outputs = self.model.generate(
                                **inputs,
                                max_new_tokens=self.config["pipeline"]["max_new_tokens"],
                                temperature=0.0,
                                do_sample=False,
                                pad_token_id=self.tokenizer.pad_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
                            )
                else:
                    raise
            
            # Decode answers
            for i in range(len(batch_prompts)):
                ans = self.tokenizer.decode(
                    outputs[i][input_len:], skip_special_tokens=True
                )
                all_answers.append(ans.strip())
            
            # Clean up after each batch
            del inputs, outputs
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
        self, question: str, selected_indices: list[int], samples: list[dict]
    ) -> str:
        selected_reasons = [samples[i]["reason"] for i in selected_indices]
        ranked_order = self._rank_reasons_by_relevance(selected_reasons, question)
        ordered_reasons = [selected_reasons[i] for i in ranked_order]

        final_prompt = self.build_final_prompt(question, ordered_reasons)
        return self.generate_answer(final_prompt)
