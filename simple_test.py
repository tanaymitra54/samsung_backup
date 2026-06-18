#!/usr/bin/env python3
"""Direct test to see what the model actually outputs"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading model...")
model_name = "Qwen/Qwen3.5-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache/models")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="./cache/models",
    torch_dtype=torch.float16,
    device_map="auto"
)

question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

# Test different prompts
prompts = {
    "Direct": f"Q: {question}\nA:",
    "Boxed": f"{question}\n\nPlease solve this step by step and put your final numerical answer in \\boxed{{}}.\n\nSolution:",
    "Simple": f"{question}\n\nThe answer is:",
    "CoT": f"Let's solve this step by step:\n{question}\n\nStep 1:",
}

for name, prompt in prompts.items():
    print(f"\n{'='*70}")
    print(f"Prompt: {name}")
    print(f"{'='*70}")
    print(f"Input: {prompt[:100]}...")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"\nModel output:\n{response}")
    print(f"\nExtracted numbers: {[x for x in response.split() if x.replace('.','').replace(',','').isdigit()]}")
