"""Actual PyTorch/TRL CPU checks on tiny random models, not benchmark accuracy."""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")


def tiny_model_and_tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
    words = ["<pad>", "<eos>", "<unk>", "user", "assistant", "Question", "Final", "answer", ":", "1", "2", "reason", "step"]
    tokenizer_object = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="<unk>"))
    tokenizer_object.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_object, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}assistant {% endif %}"
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(words), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
        max_position_embeddings=128, bos_token_id=1, eos_token_id=1, pad_token_id=0))
    return model, tokenizer


def test_real_generation_confidence_and_context_guard():
    from types import SimpleNamespace
    from evaluation.diagnostic_runner import GenerationBackend
    from pipeline.sampling import DiverseSampler
    torch.set_num_threads(1)
    model, tokenizer = tiny_model_and_tokenizer()
    model.eval()
    sampler = DiverseSampler.__new__(DiverseSampler)
    sampler.tokenizer = tokenizer
    backend = GenerationBackend.__new__(GenerationBackend)
    backend.torch = torch
    backend.sampler = sampler
    backend.config = dict(pipeline=dict(sampling_max_new_tokens=8, max_new_tokens=8, top_p=0.95))
    backend.inference = SimpleNamespace(tokenizer=tokenizer, model=model, generation_input_device=torch.device("cpu"),
                                        _apply_chat_template=lambda p: tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True))
    torch.manual_seed(42)
    sampled = backend.generate("Question reason", True, 0.7)
    assert sampled["output_tokens"] > 0 and 0 <= sampled["confidence_score"] <= 1
    first = backend.generate("Question reason", False)
    second = backend.generate("Question reason", False)
    assert first["raw_text"] == second["raw_text"]
    with pytest.raises(ValueError, match="Context overflow"):
        backend.generate("Question " * 130, False)


def test_exact_selection_matches_torch_energy():
    from pipeline.qubo_math import exhaustive_solve
    torch.manual_seed(11)
    matrix = torch.randn(16, 16)
    matrix = (matrix + matrix.T) / 2
    state, energy = exhaustive_solve(matrix.tolist(), 6)
    x = torch.tensor(state, dtype=matrix.dtype)
    reference = torch.dot(matrix.diag(), x) + (torch.triu(matrix, diagonal=1) * x[:, None] * x[None, :]).sum()
    assert int(x.sum()) == 6
    assert energy == pytest.approx(reference.item(), abs=1e-5)


def test_torch_numpy_bridge_used_by_embeddings():
    import numpy as np
    array = torch.arange(4, dtype=torch.float32).numpy()
    assert np.array_equal(array, np.arange(4))
    assert torch.equal(torch.from_numpy(array), torch.arange(4, dtype=torch.float32))


def test_sft_real_update_with_pinned_config_api(tmp_path):
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    torch.set_num_threads(1)
    model, tokenizer = tiny_model_and_tokenizer()
    model.config.use_cache = False
    data = Dataset.from_list([{"text": "user Question assistant Final answer : 1"} for _ in range(4)])
    args = SFTConfig(output_dir=str(tmp_path / "sft"), use_cpu=True, bf16=False, fp16=False,
                     max_length=32, max_steps=1, per_device_train_batch_size=2,
                     learning_rate=0.01, report_to="none", save_strategy="no", logging_steps=1,
                     dataset_text_field="text", packing=False, dataloader_pin_memory=False)
    trainer = SFTTrainer(model=model, args=args, train_dataset=data, processing_class=tokenizer,
                         peft_config=LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    before = {n: p.detach().clone() for n, p in trainer.model.named_parameters() if p.requires_grad}
    trainer.train()
    assert any(not torch.equal(before[n], p) for n, p in trainer.model.named_parameters() if n in before)


def test_device_mapping_respects_scheduler_allocation(monkeypatch):
    from pipeline.device_utils import mask_worker_gpu, resolve_device
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,7")
    assert mask_worker_gpu(1) == "cuda:0"
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "7"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaa,GPU-bbb")
    assert mask_worker_gpu(0) == "cuda:0"
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == "GPU-aaa"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError, match="only 2"):
        resolve_device("cuda:7")


def test_dpo_real_update_frozen_sft_reference_and_reload(tmp_path):
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from trl import DPOConfig, DPOTrainer
    torch.set_num_threads(1)
    torch.manual_seed(101)
    base, tokenizer = tiny_model_and_tokenizer()
    base_state = copy.deepcopy(base.state_dict())
    sft = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    # A nonzero saved adapter emulates an SFT starting point; no SFT quality claim.
    with torch.no_grad():
        for name, param in sft.named_parameters():
            if "lora_B" in name:
                param.fill_(0.03)
    sft_path = tmp_path / "sft"
    sft.save_pretrained(sft_path)
    fresh, _ = tiny_model_and_tokenizer()
    fresh.load_state_dict(base_state)
    model = PeftModel.from_pretrained(fresh, sft_path, adapter_name="policy", is_trainable=True)
    model.load_adapter(sft_path, adapter_name="reference", is_trainable=False)
    model.set_adapter("policy")
    before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
    dataset = Dataset.from_list([dict(prompt=[{"role": "user", "content": "Question reason"}],
        chosen=[{"role": "assistant", "content": "Final answer : 1"}],
        rejected=[{"role": "assistant", "content": "Final answer : 2"}]) for _ in range(4)])
    args = DPOConfig(output_dir=str(tmp_path / "dpo"), beta=0.1, max_length=32,
        model_adapter_name="policy", ref_adapter_name="reference", use_cpu=True, bf16=False, fp16=False,
        max_steps=2, per_device_train_batch_size=2, learning_rate=0.01,
        logging_steps=1, logging_nan_inf_filter=False, report_to="none", save_strategy="no", optim="adamw_torch",
        dataloader_pin_memory=False)
    trainer = DPOTrainer(model=model, ref_model=None, args=args, train_dataset=dataset, processing_class=tokenizer)
    trainer.train()
    policy_changed = any(not torch.equal(before[n], p) for n, p in model.named_parameters() if ".policy." in n and "lora_" in n)
    reference_unchanged = all(torch.equal(before[n], p) for n, p in model.named_parameters() if ".reference." in n and "lora_" in n)
    assert policy_changed and reference_unchanged
    assert all(__import__("math").isfinite(log["loss"]) for log in trainer.state.log_history if "loss" in log)
    output = tmp_path / "final_adapter"
    model.save_pretrained(output, selected_adapters=["policy"])
    reloaded_base, _ = tiny_model_and_tokenizer()
    reloaded_base.load_state_dict(base_state)
    loaded = PeftModel.from_pretrained(reloaded_base, output / "policy")
    assert "default" in loaded.peft_config


class TorchTransportBackend:
    """Real PyTorch operations in spawned workers; outputs are transport fixtures."""
    revisions = {"fixture": "torch-cpu"}
    def __init__(self, *args):
        torch.set_num_threads(1)

    def pool(self, example, seed):
        from evaluation.selection_protocol import N_TRACES
        generator = torch.Generator().manual_seed(seed)
        embeddings = torch.nn.functional.normalize(torch.randn(N_TRACES, 4, generator=generator), dim=1)
        samples = [dict(raw_text="Final answer: 1", canonical_answer="1", answer="1", reason="transport fixture",
                        sc_score=1.0, nli_score=0.5, conf_score=0.5, arith_score=0.0) for _ in range(N_TRACES)]
        return samples, (embeddings @ embeddings.T).tolist()


def test_two_spawned_torch_workers_and_durable_merge(tmp_path):
    from evaluation.diagnostic_data import question_hash
    from evaluation.parallel_execution import execute_tasks, merge_records
    tasks = []
    for index in range(4):
        question = f"Transport question {index}"
        tasks.append((dict(question=question, question_hash=question_hash(question), source_id=str(index),
                           benchmark="gsm8k", gold="1", answer_type="numeric", role="fit"), 42))
    statuses = execute_tasks(tasks, ["cpu", "cpu"], tmp_path, None, None, {}, "fixture", backend_factory=TorchTransportBackend)
    assert [status["records"] for status in statuses] == [2, 2]
    records = merge_records(tmp_path, [e for e, _ in tasks], [42])
    assert len(records) == 4 and len({r["worker_index"] for r in records}) == 2
    assert all(not r.get("error") for r in records)
