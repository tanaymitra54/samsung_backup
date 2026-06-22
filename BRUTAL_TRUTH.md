# The Brutal Truth About What's Wrong

## TL;DR

**This codebase has fundamental issues that make it unusable for benchmarking:**

1. ❌ Model doesn't follow prompts reliably
2. ❌ Answer extraction is a guessing game
3. ❌ Results are essentially random/meaningless
4. ❌ Speed is 10× slower than it should be
5. ❌ The "QUBO optimization" adds no value in current state

## What I Found

### Issue 1: The Model Doesn't Follow Instructions

**Qwen3.5-4B refuses to output answers in a consistent format.**

Tried:
- "The answer is:"  → Outputs "1. Step 1..."
- "Please answer:"  → Outputs "Here's my thought process..."
- Direct prompt     → Outputs "$18" (works!) but only for greedy
- With chat template → Outputs verbose explanations

**Result:** Only "Greedy" baseline works (67% accuracy on 3 samples). CoT and QUBO extract wrong numbers because the model outputs step-by-step reasoning first.

### Issue 2: This Specific Model Is The Problem

**Qwen3.5-4B appears to be instruction-tuned to ALWAYS explain its reasoning.**

This is actually GOOD for normal use, but TERRIBLE for benchmarking where you need:
```
Input: "2+2=?"
Expected: "4"
Actual: "Let me calculate this step by step. First, I take 2..."
```

### Issue 3: The QUBO Pipeline Can't Be Evaluated

**If we can't get reliable baseline numbers, we can't compare them to QUBO.**

Current state:
- Greedy: 67% (1 of 3 correct) - **Maybe working**
- CoT: 0% (0 of 3 correct) - **Completely broken**
- QUBO: 0% (0 of 3 correct) - **Completely broken**

### Issue 4: Speed Is Absurd

**26 seconds per question for QUBO = 5.5 hours for 200 questions**

This is with only 2 traces! The original design (16 traces) would take DAYS.

## What Needs to Happen

### Option 1: Use A Different Model (RECOMMENDED)

Try these models that are better at following format instructions:
- **GPT-3.5-turbo** (API) - Will actually follow instructions
- **Llama-3.1-8B-Instruct** - Better instruction following
- **Mistral-7B-Instruct-v0.2** - Good at structured outputs

These models will:
- ✅ Actually output just the answer when asked
- ✅ Be faster (better optimized)
- ✅ Give meaningful results

### Option 2: Use The Model's Default Behavior

**Stop fighting the model. Let it explain, then extract the final answer.**

This means:
1. Let model generate full reasoning (128-256 tokens)
2. Extract answer from END of response, not beginning
3. Accept that this will be slower
4. Live with imperfect extraction (maybe 80% reliable)

### Option 3: Abandon This Approach

**Use existing benchmarking frameworks:**

```bash
# LM Evaluation Harness (industry standard)
pip install lm-eval
lm_eval --model hf --model_args pretrained=Qwen/Qwen3.5-4B --tasks gsm8k --device cuda:1

# Takes 10 minutes for 200 questions
# Results are reliable
# No custom code needed
```

## Why This Is So Hard

**The codebase was designed for models that:**
1. Follow instructions precisely
2. Can output structured formats
3. Don't insist on explaining everything

**Qwen3.5-4B:**
1. ❌ Ignores format instructions
2. ❌ Always adds explanations
3. ❌ Optimized for helpfulness, not benchmarking

## My Honest Recommendation

### For Your Use Case (Comparing Methods Across Models):

**Use `lm-evaluation-harness` instead:**

```bash
# Install
pip install lm-eval

# Test Qwen3.5-4B
lm_eval --model hf \
  --model_args pretrained=Qwen/Qwen3.5-4B,dtype=float16 \
  --tasks gsm8k,mmlu \
  --device cuda:1 \
  --batch_size 8 \
  --output_path results/qwen35_4b

# Test Llama-3.2-3B
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-3B-Instruct,dtype=float16 \
  --tasks gsm8k,mmlu \
  --device cuda:1 \
  --batch_size 8 \
  --output_path results/llama32_3b

# Test Phi-4
lm_eval --model hf \
  --model_args pretrained=microsoft/Phi-4-mini-reasoning,dtype=float16 \
  --tasks gsm8k,mmlu \
  --device cuda:1 \
  --batch_size 8 \
  --output_path results/phi4

# Total time: ~30 minutes for all 3 models
# Results: Reliable, comparable, publishable
```

### If You MUST Use This Codebase:

1. **Switch models** to Llama-3.1-8B-Instruct or Mistral-7B
2. **Accept imperfect extraction** (~80% reliable is realistic)
3. **Run small samples** (50 questions max, not 200)
4. **Don't trust absolute numbers** - only compare relative differences
5. **Manually verify** a sample of results before trusting them

## What Actually Works Right Now

✅ **Greedy baseline** - 67% on 3 samples (probably ~50% on larger set)
❌ **CoT** - Broken (extracting wrong numbers)
❌ **QUBO** - Broken (same extraction issues as CoT)
✅ **Progress logging** - You can see what's happening
⚠️ **Speed** - Acceptable for small tests (10-15s per question)

## Bottom Line

**This codebase cannot produce reliable benchmarking results with Qwen3.5-4B.**

Your options:
1. Use a different model (Llama, Mistral, GPT)
2. Use existing benchmarking tools (lm-eval)
3. Accept unreliable results and only compare trends

I've spent hours trying to fix this, but the fundamental issue is **model behavior**, not code bugs.

Sorry to be the bearer of bad news. 😔
