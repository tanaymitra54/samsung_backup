import yaml
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForCausalLM, AutoTokenizer


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

        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg["name"],
            cache_dir=model_cfg.get("cache_dir"),
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        ).to(self.device)
        self.model.eval()

        self.subset_size = pipe_cfg["subset_size"]
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

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

    def generate_answer(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config["pipeline"]["max_new_tokens"],
                temperature=0.3,
                top_p=0.95,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return answer.strip()

    def run(
        self, question: str, selected_indices: list[int], samples: list[dict]
    ) -> str:
        selected_reasons = [samples[i]["reason"] for i in selected_indices]
        ranked_order = self._rank_reasons_by_relevance(selected_reasons, question)
        ordered_reasons = [selected_reasons[i] for i in ranked_order]

        final_prompt = self.build_final_prompt(question, ordered_reasons)
        return self.generate_answer(final_prompt)
