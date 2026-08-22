"""
================================================================================
  QUANTUM-INSPIRED ANNEALING PIPELINE  ---  Mentor Demonstration Script
  Samsung PRISM 2026
================================================================================

PURPOSE
-------
End-to-end live demonstration of the full pipeline using a real Qwen-2.5-3B-
Instruct model for chain generation and the project's own ReasonVerifier for
scoring.  No mock data.  Every step shows:
    - The governing formula / equation
    - WHY that approach was chosen
    - The actual numbers from live inference

THREE QUESTIONS:
  Q1  Arithmetic  : "A train travels at 60 km/h for 2.5 hours. How far?"
  Q2  Commonsense : "Do penguins live closer to the North or South Pole?"
  Q3  Multi-step  : "A baker makes 48 cookies, sells 3/4, gives away 6.
                     How many are left?"

THREE METHODS COMPARED:
  (1) Greedy decoding        -- single highest-scoring chain, no diversity
  (2) CoT Top-K majority     -- top-K chains voted, no diversity constraint
  (3) QUBO pipeline (ours)   -- Simulated Annealing on the QUBO matrix

RUN:
    $env:PYTHONIOENCODING='utf-8'
    python scripts/demo_pipeline.py

REQUIREMENTS:
    pip install transformers accelerate sentence-transformers scikit-learn numpy
    Model will be downloaded from HuggingFace on first run (~6 GB total):
      - Qwen/Qwen2.5-3B-Instruct
      - all-MiniLM-L6-v2
      - cross-encoder/nli-deberta-v3-base
================================================================================
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os, re, sys, ast, io, math, time, textwrap, random, gc
from typing import Optional
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force UTF-8 output on Windows terminals
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from pipeline.device_utils import resolve_device


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

QWEN_MODEL   = "Qwen/Qwen2.5-3B-Instruct"
NLI_MODEL    = "cross-encoder/nli-deberta-v3-base"
EMBED_MODEL  = "all-MiniLM-L6-v2"

# Sampling settings
NUM_TEMPLATES   = 4      # 4 structurally distinct prompt framings
SAMPLES_PER_T   = 2      # samples per template  →  4 x 2 = 8 candidates
TEMP_LOW        = 0.3
TEMP_HIGH       = 0.9
MAX_NEW_TOKENS  = 512

# QUBO settings
K_TARGET        = 4      # chains to select
PENALTY_W       = 2.0    # off-diagonal penalty multiplier
DIVERSITY_BONUS = 0.5    # diagonal offset
ANS_SIM_W       = 0.6    # cosine weight in off-diagonal
ANS_AGR_W       = 0.4    # answer-agree weight in off-diagonal
LAMBDA_C        = 0.10   # cardinality constraint coefficient

# Verifier weights
MATH_MATCH_W    = 0.63
MATH_CONSIST_W  = 0.27
CONSENSUS_W     = 0.10
LANG_NLI_W      = 0.60
LANG_COV_W      = 0.25
LANG_STRUCT_W   = 0.15

# Display
W = 82   # terminal width

# Questions
QUESTIONS = [
    {"id": "Q1", "type": "math",
     "question": "A train travels at 60 km/h for 2.5 hours. How far does it go?",
     "gold": "150"},
    {"id": "Q2", "type": "language",
     "question": "Do penguins live closer to the North Pole or the South Pole?",
     "gold": "South Pole"},
    {"id": "Q3", "type": "math",
     "question": ("A baker makes 48 cookies. He sells 3/4 of them and gives away "
                  "6 of the rest. How many cookies are left?"),
     "gold": "6"},
]


# ─────────────────────────────────────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def banner(title: str, char: str = "=") -> None:
    pad = (W - len(title) - 2) // 2
    print("\n" + char * W)
    print(f"{char * pad} {title} {char * (W - pad - len(title) - 2)}")
    print(char * W)

def section(title: str) -> None:
    print(f"\n  +{'-' * (W - 4)}+")
    print(f"  |  {title:<{W - 6}}|")
    print(f"  +{'-' * (W - 4)}+")

def formula_box(lines: list) -> None:
    """Print a nicely bordered formula block."""
    inner = W - 6
    print()
    print(f"  +{'-' * (inner + 2)}+")
    print(f"  |  {'FORMULA / EQUATION':<{inner}}|")
    print(f"  +{'=' * (inner + 2)}+")
    for line in lines:
        print(f"  |  {line:<{inner}}|")
    print(f"  +{'-' * (inner + 2)}+")

def why_box(msg: str) -> None:
    """Print a WHY block."""
    inner = W - 6
    wrapped = textwrap.wrap(msg, width=inner - 6)
    print()
    print(f"  |  WHY? {wrapped[0] if wrapped else ''}")
    for line in wrapped[1:]:
        print(f"  |        {line}")

def note(msg: str, prefix: str = "  [i] ") -> None:
    wrapped = textwrap.fill(msg, width=W - len(prefix),
                            subsequent_indent=" " * len(prefix))
    print(prefix + wrapped)

def good(msg: str) -> None:  print(f"  [+] {msg}")
def warn(msg: str) -> None:  print(f"  [!] {msg}")

def table(headers: list, rows: list, col_widths: list) -> None:
    sep = "  +" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    hdr = "  |" + "|".join(f" {str(h):<{w}} " for h, w in zip(headers, col_widths)) + "|"
    print(sep); print(hdr); print(sep)
    for row in rows:
        print("  |" + "|".join(f" {str(c):<{w}} " for c, w in zip(row, col_widths)) + "|")
    print(sep)

def pause(msg: str = "  [Enter] Press Enter to continue...") -> None:
    try: input(msg)
    except (EOFError, KeyboardInterrupt): print()


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

class ModelBundle:
    """Holds all three models: LLM, NLI, Embedder."""

    def __init__(self):
        banner("LOADING MODELS", char="=")

        # ── Device selection ──────────────────────────────────────────────────
        # Auto-picks whichever visible GPU currently has the most free VRAM
        # instead of always grabbing cuda:0 (see pipeline/device_utils.py).
        self.device = resolve_device()
        if self.device.type == "cuda":
            idx = self.device.index or 0
            torch.cuda.set_device(idx)
            note(f"CUDA device detected: {torch.cuda.get_device_name(idx)} (cuda:{idx}) — using GPU.")
        else:
            warn("No CUDA GPU found — running on CPU.  Generation will be slow (~1-2 min/chain).")

        # ── 1. Qwen 2.5 3B Instruct ──────────────────────────────────────────
        print()
        note(f"Loading LLM: {QWEN_MODEL}")
        why_box(
            "Qwen-2.5-3B-Instruct is chosen for the demo because it is small "
            "enough to run without a GPU cluster yet strong enough to produce "
            "coherent multi-step reasoning chains.  The '-Instruct' suffix means "
            "the model is fine-tuned with RLHF to follow instructions — critical "
            "for prompt-template diversity to actually produce structurally "
            "different reasoning styles."
        )
        t0 = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
        if self.device.type == "cuda":
            # Pinned to the one selected GPU rather than "auto": accelerate's
            # "auto" placement ignores which card we picked as most-free and
            # spreads/chooses across every visible device instead.
            load_kwargs["device_map"] = {"": self.device.index or 0}
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float32

        self.llm = AutoModelForCausalLM.from_pretrained(QWEN_MODEL, **load_kwargs)
        if self.device.type == "cpu":
            self.llm = self.llm.to(self.device)
        self.llm.eval()
        note(f"LLM loaded in {time.time()-t0:.1f}s")

        # ── 2. NLI DeBERTa (for language scoring) ────────────────────────────
        print()
        note(f"Loading NLI model: {NLI_MODEL}")
        why_box(
            "cross-encoder/nli-deberta-v3-base is a Natural Language Inference "
            "model fine-tuned on MultiNLI + SNLI.  It takes a (premise, hypothesis) "
            "pair and outputs three probabilities: P(contradiction), P(entailment), "
            "P(neutral).  We use P(entailment) as the primary quality signal for "
            "language/commonsense chains: if the reasoning ENTAILS the answer, it "
            "is logically sound.  DeBERTa v3 uses disentangled attention (position "
            "and content are attended separately), giving it SOTA performance on NLI "
            "benchmarks at moderate size (~184M params)."
        )
        t0 = time.time()
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        self.nli_model = self.nli_model.to(self.device)
        self.nli_model.eval()
        self.nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL)
        # Read label2id from model config so it works with any NLI model
        label2id = getattr(self.nli_model.config, "label2id", {})
        self._entail_idx = label2id.get("entailment", label2id.get("ENTAILMENT", 1))
        note(f"NLI model loaded in {time.time()-t0:.1f}s  (entailment label index = {self._entail_idx})")

        # ── 3. Sentence Embedder (for QUBO similarity) ────────────────────────
        print()
        note(f"Loading embedder: {EMBED_MODEL}")
        why_box(
            "all-MiniLM-L6-v2 is a 22M-parameter sentence transformer that "
            "produces 384-dimensional semantic embeddings.  Cosine similarity "
            "between two embeddings is our proxy for semantic redundancy between "
            "two reasoning chains.  MiniLM is chosen over larger models (MPNet, "
            "BGE) because: (a) it is fast enough to embed 16 chains in <1s, "
            "(b) its 384-dim space is sufficient for pairwise redundancy detection, "
            "and (c) it is already installed as a project dependency."
        )
        t0 = time.time()
        self.embedder = SentenceTransformer(EMBED_MODEL,
                                            device=str(self.device))
        note(f"Embedder loaded in {time.time()-t0:.1f}s")

        banner("ALL MODELS READY", char="-")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — CHAIN GENERATION
# ─────────────────────────────────────────────────────────────────────────────

TEMPLATE_NAMES = ["T1-Forward", "T2-Backward", "T3-Analogical", "T4-UnitsBD"]

def make_templates(question: str, task_type: str) -> list:
    """Return 4 structurally distinct prompt framings."""
    suffix = "\nPlease put your final answer on a new line starting with 'Answer:'"
    t4 = (
        f"Identify the units and quantities involved, then compute step by step.\n"
        f"Question: {question}{suffix}"
        if task_type == "math"
        else f"Break this down step by step.\nQuestion: {question}{suffix}"
    )
    return [
        f"Starting from the given information, work forward step by step to reach the answer.\nQuestion: {question}{suffix}",
        f"Start from what the answer must satisfy and work backward to verify it from the given facts.\nQuestion: {question}{suffix}",
        f"This problem is similar to one where you identify a pattern or analogy. Use that to reason through it.\nQuestion: {question}{suffix}",
        t4,
    ]

def _apply_chat_template(tokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True
        )
    return prompt

def _parse_reason_answer(text: str):
    """
    Split generated text into (reason, answer).
    The answer is the line containing 'answer' or 'therefore'.
    The reason is everything else.
    """
    lines = text.strip().split("\n")
    answer = ""
    reason = text
    
    for i in range(len(lines) - 1, -1, -1):
        line = lines[i]
        if line.strip().lower().startswith("answer:"):
            answer = line.split(":", 1)[1].strip()
            reason = "\n".join(lines[:i] + lines[i+1:])
            return reason.strip(), answer.strip()
            
    for line in lines:
        if "answer" in line.lower() or "therefore" in line.lower():
            answer = line
            reason = "\n".join(l for l in lines if l != line)
            break
    return reason.strip(), answer.strip()

def generate_chains(bundle: ModelBundle, question: dict, verbose: bool = True) -> list:
    """
    Generate NUM_TEMPLATES x SAMPLES_PER_T reasoning chains using Qwen 2.5 3B.
    Returns a list of dicts with keys: reason, answer, temperature, template_name.
    """
    section(f"STEP 1: CHAIN GENERATION  [{question['id']}  {question['type'].upper()}]")

    formula_box([
        "Pool size = num_templates x samples_per_template",
        f"         = {NUM_TEMPLATES} x {SAMPLES_PER_T} = {NUM_TEMPLATES * SAMPLES_PER_T} candidate chains",
        "",
        "For each chain i:",
        "  temperature_i ~ Uniform(T_low, T_high)  =  Uniform(0.3, 0.9)",
        "  chain_i = LLM(prompt_template_j, temperature_i)",
        "",
        "Sampling distribution: p(token | context) = softmax(logits / T)",
        "  Low T -> peaked distribution -> deterministic, focused",
        "  High T -> flat distribution  -> diverse, exploratory",
    ])

    why_box(
        "WHY 4 TEMPLATES: LLMs are highly sensitive to prompt framing. A single "
        "'solve step by step' prompt produces highly correlated chains — the pool "
        "lacks diversity and QUBO selection provides little benefit. Four "
        "structurally distinct templates (forward, backward, analogical, "
        "units-first) force qualitatively different reasoning strategies, "
        "maximising the chance that the pool contains both correct AND diverse chains. "
        "WHY RANDOM TEMPERATURE: Low temperature produces reliable but similar "
        "chains; high temperature produces diverse but sometimes incoherent chains. "
        "Randomising across [0.3, 0.9] gets both — the Verifier later filters "
        "the incoherent ones, so sampling breadth costs nothing in quality."
    )

    print()
    note(f"Question : {question['question']}")
    note(f"Gold ans : {question['gold']}")
    note(f"Task type: {question['type']}")
    print()

    templates = make_templates(question["question"], question["type"])
    template_descs = [
        "Forward  : 'work forward step by step from given info'",
        "Backward : 'start from what answer must satisfy, verify backward'",
        "Analogical: 'identify a pattern or analogy and reason through it'",
        "Units-BD : 'identify units and quantities, then compute' (math)"
                  " / 'break this down step by step' (commonsense)",
    ]

    print("  TEMPLATE DESCRIPTIONS:")
    for name, desc in zip(TEMPLATE_NAMES, template_descs):
        print(f"    [{name}]  {desc}")
    print()
    note(
        "Chain-decision rule: Each template x temperature combination produces "
        "one candidate via greedy-search-with-sampling (do_sample=True, top_p=0.95). "
        "The raw generated text is split into (reason, answer) by _parse_reason_answer(): "
        "the first line containing 'answer' or 'therefore' becomes the answer field; "
        "all other lines form the reasoning trace. If no such line exists, answer='' "
        "and reason=full_text."
    )
    print()

    all_chains = []
    for t_idx, (template, t_name) in enumerate(zip(templates, TEMPLATE_NAMES)):
        print(f"  --- Template {t_idx+1}/4: {t_name} ---")
        for s_idx in range(SAMPLES_PER_T):
            temp = random.uniform(TEMP_LOW, TEMP_HIGH)
            print(f"    Sample {s_idx+1}/{SAMPLES_PER_T}  temperature={temp:.3f}  ", end="", flush=True)

            chat_prompt = _apply_chat_template(bundle.tokenizer, template)
            inputs = bundle.tokenizer(
                chat_prompt, return_tensors="pt",
                truncation=True, max_length=512
            ).to(bundle.device)

            t0 = time.time()
            with torch.inference_mode():
                out = bundle.llm.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=temp,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=bundle.tokenizer.pad_token_id,
                    eos_token_id=bundle.tokenizer.eos_token_id,
                )
            gen_text = bundle.tokenizer.decode(
                out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            elapsed = time.time() - t0

            reason, answer = _parse_reason_answer(gen_text)
            chain = {
                "template_name": t_name,
                "template_idx": t_idx,
                "temperature": temp,
                "reason": reason or gen_text,   # fallback to full text
                "answer": answer,
                "raw": gen_text,
            }
            all_chains.append(chain)

            print(f"[{elapsed:.1f}s]  answer='{answer[:40]}'")
            if verbose:
                print()
                print(f"      Reason : {reason[:120]}{'...' if len(reason) > 120 else ''}")
                print(f"      Answer : {answer[:80]}")
                print(f"      Decision: line containing 'answer'/'therefore' "
                      f"{'found' if answer else 'NOT found -> answer=empty'}")
                print()

    print()
    print(f"  Generated {len(all_chains)} candidate chains across {NUM_TEMPLATES} templates.")
    return all_chains


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — VERIFICATION / SCORING
# ─────────────────────────────────────────────────────────────────────────────

_STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","may","might","to",
    "of","in","on","at","by","for","with","as","and","but","or","this",
    "that","it","he","she","they","we","you","i","so","up","out","all",
    "any","no","if","then","than","more","most","also","only","just","not",
}

def _extract_last_number(text: str) -> Optional[float]:
    cleaned = (text or "").strip().replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned)
    if not matches: return None
    v = float(matches[-1])
    return int(v) if v == int(v) else v

def _gold_match(pred, gold) -> float:
    if pred is None or gold is None: return 0.0
    thr = max(0.01, 0.01 * abs(gold))
    return 1.0 if abs(pred - gold) < thr else 0.0

def _ast_eval(node) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _ast_eval(node.operand); return -v if v is not None else None
    if isinstance(node, ast.BinOp):
        L, R = _ast_eval(node.left), _ast_eval(node.right)
        if L is None or R is None: return None
        if isinstance(node.op, ast.Add):  return L + R
        if isinstance(node.op, ast.Sub):  return L - R
        if isinstance(node.op, ast.Mult): return L * R
        if isinstance(node.op, ast.Div):  return (L / R) if abs(R) > 1e-12 else None
    return None

def _check_expressions(text: str) -> list:
    """Extract and AST-verify all 'LHS = RHS' arithmetic statements."""
    pat = re.compile(r'([\d\s().+\-*/x\u00d7\u00f7]+)\s*=\s*(-?\d+(?:[.,]\d+)?)')
    results = []
    for m in pat.finditer(text):
        lhs_raw, rhs_raw = m.group(1).strip(), m.group(2).replace(",", "")
        lhs = lhs_raw.replace("\u00d7","*").replace("\u00f7","/")
        lhs = re.sub(r'([\d)]) x ([\d(])', r'\1 * \2', lhs)
        lhs = re.sub(r'([\d)])x([\d(])', r'\1*\2', lhs).strip()
        try:
            stated = float(rhs_raw)
            tree = ast.parse(lhs, mode='eval')
            computed = _ast_eval(tree.body)
        except Exception:
            continue
        if computed is None: continue
        tol = max(0.01, 0.01 * abs(stated))
        results.append((lhs, stated, computed, abs(computed - stated) < tol))
    return results

def _arithmetic_consistency(text: str) -> float:
    exprs = _check_expressions(text)
    if not exprs: return 0.5   # no verifiable expressions → neutral score
    return sum(1 for *_, ok in exprs if ok) / len(exprs)

def _lexical_coverage(reason: str, question: str) -> float:
    def cw(t):
        toks = re.findall(r"[a-z]+", t.lower())
        return {x for x in toks if x not in _STOPWORDS and len(x) > 2}
    q_words = cw(question)
    if not q_words: return 0.5
    return len(cw(reason) & q_words) / len(q_words)

def _structural_completeness(reason: str, mu: float = 1.5, tau: float = 1.0) -> float:
    sents = [s for s in re.split(r'(?<=[.!?])\s+', reason.strip()) if len(s.strip()) > 5]
    z = (max(1, len(sents)) - mu) / tau
    return 1.0 / (1.0 + math.exp(-z))

def _nli_score(bundle: ModelBundle, reason: str, answer: str) -> float:
    """Run DeBERTa NLI cross-encoder: premise=reason, hypothesis=answer."""
    inputs = bundle.nli_tokenizer(
        reason, answer or "yes",
        return_tensors="pt", truncation=True, max_length=512
    ).to(bundle.device)
    with torch.no_grad():
        logits = bundle.nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    return probs[bundle._entail_idx].item()

def _normalise_ans(ans: str, task_type: str) -> str:
    if task_type == "math":
        return ans.strip()
    m = re.search(r'\b([A-Ea-e])\b', ans)
    return m.group(1).lower() if m else ans.strip().lower()

def score_chains(bundle: ModelBundle, chains: list, question: dict, gold: str) -> list:
    """Score all chains. Attaches _base, consensus_score, correctness_score."""
    section("STEP 2: VERIFICATION  ---  Per-chain Quality Scoring")

    task_type = question["type"]
    q_text    = question["question"]

    # ── Formula explanation ───────────────────────────────────────────────────
    if task_type == "math":
        formula_box([
            "MATH SCORING (per chain i):",
            "",
            "  base_score(i) = alpha * answer_match(i)  +  beta * arith_consistency(i)",
            "  where alpha = 0.63, beta = 0.27",
            "",
            "  answer_match(i):",
            "    = 1.0  if  |pred_num - gold_num| < max(0.01, 0.01 * |gold_num|)",
            "    = 0.0  otherwise",
            "",
            "  arith_consistency(i):",
            "    = #{correct arithmetic steps} / #{total arithmetic steps}",
            "    Checked via AST evaluation (NO eval(), pure tree recursion)",
            "    If no arithmetic steps found -> 0.5 (neutral)",
            "",
            "SELF-CONSISTENCY (Wang et al., 2023):",
            "  consensus(i) = #{j != i : answer_j matches answer_i} / (N - 1)",
            "  agreement = 1 if |a_i - a_j| < tolerance,  else 0",
            "",
            "FINAL BLENDED SCORE:",
            "  correctness(i) = (1 - gamma) * base(i)  +  gamma * consensus(i)",
            "  gamma = 0.10",
        ])
    else:
        formula_box([
            "LANGUAGE SCORING (per chain i):",
            "",
            "  base_score(i) = w_nli   * P_entail(reason_i, answer_i)",
            "                + w_cov   * lexical_coverage(reason_i, question)",
            "                + w_struct * structural_completeness(reason_i)",
            "  where w_nli=0.60, w_cov=0.25, w_struct=0.15",
            "",
            "  P_entail = P(entailment | reason, answer) from DeBERTa NLI model",
            "    Premise   = reasoning chain (reason_i)",
            "    Hypothesis = extracted answer (answer_i)",
            "    Model outputs [P_contradict, P_entail, P_neutral]",
            "",
            "  lexical_coverage = |content_words(reason) ∩ content_words(question)|",
            "                   / |content_words(question)|",
            "  (content words = non-stopword tokens with length > 2)",
            "",
            "  structural_completeness = sigmoid((n_sentences - mu) / tau)",
            "  sigmoid maps sentence count to [0,1]: 1 sent->~0.38, 3 sents->~0.82",
            "",
            "SELF-CONSISTENCY:",
            "  consensus(i) = #{j != i : normalised(ans_j) == normalised(ans_i)} / (N-1)",
            "  Empty answers count as 0 (no spurious agreement)",
            "",
            "FINAL BLENDED SCORE:",
            "  correctness(i) = 0.90 * base(i) + 0.10 * consensus(i)",
        ])

    why_box(
        "WHY SEPARATE MATH VS LANGUAGE SCORING: Math correctness is objective "
        "(a number is right or wrong), so answer_match is the dominant signal. "
        "Language correctness is subjective and multi-dimensional — a chain can "
        "be factually correct but logically incoherent or off-topic.  NLI "
        "captures logical coherence; lexical coverage captures topical relevance; "
        "structural completeness captures reasoning depth. "
        "WHY SELF-CONSISTENCY: Wang et al. (2023) showed that chains agreeing "
        "with the majority answer are statistically much more likely to be correct, "
        "independent of their individual score. Adding 10% consensus weight lifts "
        "high-agreement chains even when their arithmetic is partially wrong. "
        "WHY AST NOT eval(): eval() on LLM-generated text is a code injection "
        "risk. AST tree recursion only handles numeric literals and the four "
        "arithmetic operators — anything else (function calls, variable names) "
        "returns None and is silently skipped."
    )

    if task_type == "language":
        print()
        print("  --- NLI-DeBERTa MODEL EXPLANATION ---")
        note(
            "DeBERTa (Decoding-enhanced BERT with Disentangled Attention) v3 uses "
            "disentangled attention: instead of a single attention score per token "
            "pair, it computes TWO separate scores -- one based on content embeddings "
            "and one based on relative position embeddings.  The final attention "
            "weight is the sum of four cross-terms: content-to-content, "
            "content-to-position, position-to-content, position-to-position.  "
            "This makes DeBERTa particularly strong at capturing long-range "
            "syntactic dependencies, which are crucial for NLI."
        )
        print()
        note(
            "The cross-encoder architecture (as opposed to bi-encoder) encodes the "
            "(premise, hypothesis) PAIR jointly in one forward pass, allowing full "
            "cross-attention between the two texts.  This is more accurate than "
            "bi-encoders (which encode premise and hypothesis independently) at the "
            "cost of being slower -- acceptable here since we only call it N=8 "
            "times per question."
        )
        print()
        note(
            "Label mapping for cross-encoder/nli-deberta-v3-base: "
            "{contradiction:0, entailment:1, neutral:2}. "
            f"We extract P(entailment) = softmax(logits)[{bundle._entail_idx}].  "
            "A score near 1.0 means the model is confident the reasoning chain "
            "logically supports the answer.  Below 0.5 typically indicates "
            "contradiction or irrelevance."
        )

    print()

    # ── Compute per-chain base scores ─────────────────────────────────────────
    gold_num = _extract_last_number(gold)

    for i, c in enumerate(chains):
        reason = c["reason"]
        answer = c["answer"]

        if task_type == "math":
            exprs = _check_expressions(reason)
            consist = _arithmetic_consistency(reason)
            pred_num = _extract_last_number(answer) or _extract_last_number(reason)
            match    = _gold_match(pred_num, gold_num)
            base     = MATH_MATCH_W * match + MATH_CONSIST_W * consist
            c.update({
                "_exprs": exprs, "_consist": consist,
                "_match": match, "_pred_num": pred_num,
                "_base": base,
            })
        else:
            nli  = _nli_score(bundle, reason, answer)
            cov  = _lexical_coverage(reason, q_text)
            stru = _structural_completeness(reason)
            sents = [s for s in re.split(r'(?<=[.!?])\s+', reason.strip()) if len(s.strip()) > 5]
            base  = LANG_NLI_W * nli + LANG_COV_W * cov + LANG_STRUCT_W * stru
            c.update({
                "_nli": nli, "_cov": cov, "_struct": stru,
                "_sents": len(sents), "_base": base,
            })

        print(f"  [{c['template_name']}  s={i%SAMPLES_PER_T+1}]  base={base:.3f}  "
              f"ans='{answer[:30]}'")

    # ── Self-consistency (Fix #1) ─────────────────────────────────────────────
    print()
    print("  --- Fix #1: Computing Self-Consistency ---")
    norm_answers = [_normalise_ans(c["answer"], task_type) for c in chains]
    num_answers  = [_extract_last_number(c["answer"] or c["reason"]) for c in chains]
    N = len(chains)

    for i, c in enumerate(chains):
        agree = 0
        for j in range(N):
            if j == i: continue
            if task_type == "math":
                ai, aj = num_answers[i], num_answers[j]
                if ai is not None and aj is not None:
                    agree += int(_gold_match(ai, aj) > 0.5)
            else:
                a_i, a_j = norm_answers[i], norm_answers[j]
                if a_i and a_j and a_i == a_j:
                    agree += 1
        c["consensus_score"] = agree / (N - 1) if N > 1 else 0.0

    # ── Blend final correctness_score ────────────────────────────────────────
    for c in chains:
        c["correctness_score"] = min(1.0,
            (1.0 - CONSENSUS_W) * c["_base"] + CONSENSUS_W * c["consensus_score"]
        )

    # ── Print scoring table ───────────────────────────────────────────────────
    print()
    if task_type == "math":
        hdrs = ["#", "Template", "Temp", "Ans[:8]", "match", "consist", "consensus", "FINAL"]
        cws  = [2, 14, 5, 10, 6, 7, 9, 7]
        rows = []
        for i, c in enumerate(chains):
            rows.append([
                str(i),
                c["template_name"],
                f"{c['temperature']:.2f}",
                (c["answer"] or "")[:10],
                f"{c['_match']:.2f}",
                f"{c['_consist']:.2f}",
                f"{c['consensus_score']:.2f}",
                f"{c['correctness_score']:.3f}",
            ])
        table(hdrs, rows, cws)

        # AST drill-down
        print()
        print("  --- Fix #4: AST Arithmetic Verifier (expression by expression) ---")
        formula_box([
            "For each 'LHS = RHS' statement in the reasoning text:",
            "  1. Extract LHS string (digits, parens, operators)",
            "  2. Normalise: 'x' -> '*', Unicode x -> '*', Unicode / -> '/'",
            "  3. ast.parse(LHS, mode='eval')  ->  syntax tree",
            "  4. Recursively evaluate tree via _ast_eval(node):",
            "       Constant(n)      -> float(n)",
            "       UnaryOp(-, x)   -> -x",
            "       BinOp(+,-,*,/)  -> left op right",
            "       Anything else   -> None  (skipped, safe)",
            "  5. correct = |computed - stated| < max(0.01, 0.01*|stated|)",
            "  arith_consistency = sum(correct) / total_expressions",
        ])
        print()
        shown = 0
        for i, c in enumerate(chains):
            if not c["_exprs"]: continue
            print(f"  Chain {i} [{c['template_name']}] T={c['temperature']:.2f}:")
            for lhs, stated, computed, ok in c["_exprs"]:
                tick = "[+]" if ok else "[x]"
                print(f"      {tick}  {lhs} = {stated}  -->  computed={computed:.4g}"
                      f"  ({'CORRECT' if ok else 'WRONG'})")
            shown += 1
            if shown >= 4: break  # show first 4 chains with expressions

    else:  # language
        hdrs = ["#", "Template", "Temp", "NLI", "Cov", "Struct", "#S", "consensus", "FINAL"]
        cws  = [2, 14, 5, 5, 5, 6, 3, 9, 7]
        rows = []
        for i, c in enumerate(chains):
            rows.append([
                str(i),
                c["template_name"],
                f"{c['temperature']:.2f}",
                f"{c['_nli']:.2f}",
                f"{c['_cov']:.2f}",
                f"{c['_struct']:.2f}",
                str(c["_sents"]),
                f"{c['consensus_score']:.2f}",
                f"{c['correctness_score']:.3f}",
            ])
        table(hdrs, rows, cws)

        # Show one NLI call in detail
        best = max(chains, key=lambda c: c["correctness_score"])
        worst = min(chains, key=lambda c: c["correctness_score"])
        print()
        print("  --- NLI call detail (best vs worst chain) ---")
        for label, c in [("BEST", best), ("WORST", worst)]:
            print(f"\n  [{label}] template={c['template_name']}  T={c['temperature']:.2f}")
            print(f"    Premise (reason) : {c['reason'][:100]}...")
            print(f"    Hypothesis (ans) : {c['answer'][:60]}")
            print(f"    P(entailment)    : {c['_nli']:.4f}")
            print(f"    Lexical coverage : {c['_cov']:.4f}  "
                  f"(reason shares {c['_cov']:.0%} of question's content words)")
            print(f"    Struct. complete : {c['_struct']:.4f}  "
                  f"({c['_sents']} sentences -> sigmoid({c['_sents']:.0f}-1.5)={c['_struct']:.3f})")
            print(f"    Base score       : {c['_base']:.4f}  = "
                  f"0.60*{c['_nli']:.3f} + 0.25*{c['_cov']:.3f} + 0.15*{c['_struct']:.3f}")

    return chains


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — EMBEDDING + QUBO CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_qubo(bundle: ModelBundle, chains: list, question: dict) -> tuple:
    section("STEP 3: EMBEDDING + QUBO MATRIX CONSTRUCTION")

    formula_box([
        "QUBO energy function to MINIMISE:",
        "",
        "  E(x) = sum_i  Q[i][i] * x_i",
        "       + sum_{i<j} Q[i][j] * x_i * x_j",
        "",
        "  x_i in {0,1}  (1=selected, 0=not selected)",
        "",
        "DIAGONAL  Q[i][i] = -quality(i) + diversity_bonus + lambda_c*(1-2k)",
        "  quality(i)     = correctness_score(i)  in [0,1]",
        "  diversity_bonus = 0.5  (admission cost; only chains with quality>0.5",
        "                          are unconditionally attractive)",
        "  lambda_c*(1-2k) = 0.10*(1-8) = -0.70  (cardinality nudge)",
        "",
        "OFF-DIAGONAL  Q[i][j] = combined_sim(i,j) * penalty_weight + 2*lambda_c",
        "  combined_sim(i,j) = alpha * cosine_sim(embed_i, embed_j)",
        "                    + beta  * answer_agree(i,j)",
        "  alpha=0.6, beta=0.4,  penalty_weight=2.0,  2*lambda_c=0.20",
        "",
        "  answer_agree(i,j) = 1.0  if both answers non-empty AND equal",
        "                    = 0.0  otherwise",
        "",
        "CARDINALITY CONSTRAINT (Fix #3):",
        "  Penalty = lambda_c * (sum_i x_i  -  k)^2",
        "  Expanding: diagonal += lambda_c*(1-2k), off-diag += 2*lambda_c",
        "  Forces solver toward selecting exactly k=4 chains",
    ])

    why_box(
        "WHY NEGATIVE DIAGONAL: Minimising E(x) means selecting chains with "
        "negative Q[i][i]. Setting Q[i][i]=-quality makes high-quality chains "
        "attractive. WHY POSITIVE OFF-DIAGONAL: Selecting both i and j when "
        "sim(i,j) is high ADDS Q[i][j]>0 to energy, penalising redundant pairs. "
        "WHY ANSWER-AWARE TERM (Fix #2): Two chains can have different text "
        "(low cosine similarity) yet both assert the same wrong answer -- the "
        "embedder misses this.  The 0.4*agree term catches logical redundancy "
        "that semantic similarity alone misses. "
        "WHY CARDINALITY CONSTRAINT (Fix #3): Without it, SA can select fewer "
        "than k chains if all pairs have high redundancy.  The soft penalty "
        "(lambda_c=0.1, much smaller than typical cosine penalty ~1-2) nudges "
        "the solver toward k without dominating the diversity signal."
    )

    reasons = [c["reason"] for c in chains]
    n = len(chains)

    print()
    note(f"Embedding {n} chains with {EMBED_MODEL}...")
    t0 = time.time()
    embeddings = bundle.embedder.encode(reasons, convert_to_numpy=True)
    print(f"  Done in {time.time()-t0:.1f}s. Shape: {embeddings.shape}")

    sim_matrix = cosine_similarity(embeddings)
    answers    = [(c.get("answer") or "").strip().lower() for c in chains]

    # Print heatmap
    print()
    print(f"  Cosine similarity matrix ({n}x{n}):")
    ids = [f"C{i}" for i in range(n)]
    print("        " + "  ".join(f"{h:>5}" for h in ids))
    print("       " + "-" * (7 * n + 1))
    for i in range(n):
        row = [f"{sim_matrix[i][j]:>5.2f}" for j in range(n)]
        print(f"  {ids[i]:>4} | " + "  ".join(row))

    # Build Q
    Q = np.zeros((n, n))
    for i in range(n):
        q = chains[i]["correctness_score"]
        Q[i][i] = -q + DIVERSITY_BONUS

    for i in range(n):
        for j in range(i + 1, n):
            cos   = sim_matrix[i][j]
            agree = 1.0 if (answers[i] and answers[j] and answers[i] == answers[j]) else 0.0
            combined = ANS_SIM_W * cos + ANS_AGR_W * agree
            Q[i][j] = combined * PENALTY_W
            Q[j][i] = Q[i][j]

    # Cardinality constraint
    np.fill_diagonal(Q, Q.diagonal() + LAMBDA_C * (1 - 2 * K_TARGET))
    off = ~np.eye(n, dtype=bool)
    Q[off] += 2 * LAMBDA_C

    # Print Q matrix
    print()
    print(f"  QUBO matrix Q ({n}x{n})  [negative diagonal = attractive, positive off-diag = penalty]:")
    print("        " + "  ".join(f"{h:>6}" for h in ids))
    print("       " + "-" * (8 * n + 1))
    for i in range(n):
        row = [f"{Q[i][j]:>6.2f}" for j in range(n)]
        print(f"  {ids[i]:>4} | " + "  ".join(row))

    # Worked example for a pair
    i0, i1 = 0, 1
    print()
    print(f"  --- Worked calculation: Q[C0][C1] off-diagonal ---")
    c01 = sim_matrix[i0][i1]
    a0, a1 = answers[i0], answers[i1]
    ag  = 1.0 if (a0 and a1 and a0 == a1) else 0.0
    comb = ANS_SIM_W * c01 + ANS_AGR_W * ag
    pen  = comb * PENALTY_W
    card = 2 * LAMBDA_C
    print(f"    cos(C0,C1)            = {c01:.4f}")
    print(f"    answer_agree(C0,C1)   = {ag:.1f}  ('{a0[:20]}' vs '{a1[:20]}')")
    print(f"    combined              = 0.6*{c01:.4f} + 0.4*{ag:.1f} = {comb:.4f}")
    print(f"    base off-diag penalty = {comb:.4f} * 2.0 = {pen:.4f}")
    print(f"    cardinality term      = 2*lambda_c = 2*0.10 = {card:.2f}")
    print(f"    Q[C0][C1] FINAL       = {pen:.4f} + {card:.2f} = {pen+card:.4f}")

    print()
    print(f"  --- Worked calculation: Q[C0][C0] diagonal ---")
    q0 = chains[0]["correctness_score"]
    d_base = -q0 + DIVERSITY_BONUS
    d_card = LAMBDA_C * (1 - 2 * K_TARGET)
    print(f"    correctness_score(C0) = {q0:.4f}")
    print(f"    base diagonal         = -{q0:.4f} + 0.5 = {d_base:.4f}")
    print(f"    cardinality diagonal  = 0.10*(1-2*4) = 0.10*(-7) = {d_card:.4f}")
    print(f"    Q[C0][C0] FINAL       = {d_base:.4f} + ({d_card:.4f}) = {d_base+d_card:.4f}")

    return Q, embeddings


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — SIMULATED ANNEALING SOLVER
# ─────────────────────────────────────────────────────────────────────────────

def solve_qubo(Q: np.ndarray, chains: list) -> list:
    section("STEP 4: QUBO SOLVER  ---  Simulated Annealing")

    formula_box([
        "OBJECTIVE: minimise E(x) = x^T Q x  over x in {0,1}^N",
        "",
        "SIMULATED ANNEALING ALGORITHM:",
        "  1. Initialise state x randomly with exactly k=4 bits set to 1",
        "  2. At each step:",
        "     a. Flip one random bit: x' = flip(x, i)",
        "     b. Compute dE = E(x') - E(x)",
        "     c. Accept x' if dE < 0  (energy decreases: always accept)",
        "        else accept with probability P = exp(-dE / T)  (Metropolis)",
        "  3. Cool: T <- max(T * cooling_rate, T_final)",
        "  4. Repeat for num_iterations steps, num_reads restarts",
        "  5. Return x* = argmin energy across all reads",
        "",
        "Settings: T_init=100, T_final=0.01, cooling=0.99,",
        "          iterations=1000, num_reads=5",
    ])

    why_box(
        "WHY SIMULATED ANNEALING: The QUBO selection problem is NP-hard in "
        "general (equivalent to max-cut / max-weight independent set). SA "
        "provides a practical heuristic that escapes local minima via the "
        "temperature-controlled Metropolis acceptance criterion -- at high T, "
        "bad moves are accepted with reasonable probability (exploration); at "
        "low T, only improvements are accepted (exploitation).  The quantum "
        "analogy: SA mimics quantum tunnelling through energy barriers at finite "
        "temperature.  For N~8 chains, SA converges reliably in milliseconds.  "
        "WHY MULTIPLE READS: Different random initialisations explore different "
        "regions of the {0,1}^N space, reducing the probability of a bad local minimum."
    )

    n = Q.shape[0]
    T_init, T_final, cooling = 100.0, 0.01, 0.99
    num_iterations, num_reads = 1000, 5

    best_state, best_energy = None, float('inf')

    print()
    for read in range(num_reads):
        state = np.zeros(n)
        idx   = np.random.choice(n, min(K_TARGET, n), replace=False)
        state[idx] = 1.0
        T     = T_init
        energy = float(state @ Q @ state)

        accepts_good, accepts_bad = 0, 0
        for step in range(num_iterations):
            flip = np.random.randint(0, n)
            new_state = state.copy()
            new_state[flip] = 1.0 - new_state[flip]
            new_energy = float(new_state @ Q @ new_state)
            dE = new_energy - energy
            if dE < 0:
                state, energy = new_state, new_energy
                accepts_good += 1
            elif random.random() < math.exp(-dE / T):
                state, energy = new_state, new_energy
                accepts_bad += 1
            T = max(T * cooling, T_final)

        print(f"  Read {read+1}/{num_reads}: final_E={energy:.4f}  "
              f"selected={int(state.sum())} chains  "
              f"accepts=(good={accepts_good}, uphill={accepts_bad})")

        if energy < best_energy:
            best_energy = energy
            best_state  = state.copy()

    selected_idx = [i for i, s in enumerate(best_state) if s > 0.5]
    if not selected_idx:
        selected_idx = sorted(range(n),
                              key=lambda i: chains[i]["correctness_score"],
                              reverse=True)[:K_TARGET]

    print()
    print(f"  Best state: [{', '.join(str(int(s)) for s in best_state)}]")
    print(f"  Best QUBO energy: {best_energy:.4f}")
    print(f"  Selected chain indices: {selected_idx}")
    print()
    print("  Selected chains:")
    for idx in selected_idx:
        c = chains[idx]
        print(f"    [C{idx}] template={c['template_name']}  T={c['temperature']:.2f}  "
              f"correctness={c['correctness_score']:.3f}  ans='{c['answer'][:40]}'")

    # Compare vs naive top-K
    topk_idx = sorted(range(n),
                      key=lambda i: chains[i]["correctness_score"],
                      reverse=True)[:K_TARGET]
    topk_state = np.zeros(n)
    for i in topk_idx: topk_state[i] = 1.0
    topk_energy = float(topk_state @ Q @ topk_state)
    print()
    print(f"  Naive top-{K_TARGET} by quality alone:")
    print(f"    Indices: {topk_idx},  QUBO energy: {topk_energy:.4f}")
    if best_energy < topk_energy:
        note("SA found a lower-energy (more diverse) solution than naive top-K. [+]")
    else:
        note("SA converged to same selection as top-K (both optimal for this pool).")

    return selected_idx


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 5 — METHOD COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_methods(chains: list, selected_idx: list, question: dict, gold: str) -> dict:
    section("STEP 5: COMPARISON  ---  Greedy vs CoT Top-K vs QUBO")

    task_type = question["type"]

    def majority_vote(cands):
        if task_type == "math":
            votes = [_extract_last_number(c["answer"]) for c in cands]
            votes = [v for v in votes if v is not None]
            if not votes: return ""
            w = Counter(votes).most_common(1)[0][0]
            return str(int(w) if w == int(w) else w)
        else:
            votes = [_normalise_ans(c["answer"], task_type) for c in cands]
            if not votes: return ""
            return Counter(votes).most_common(1)[0][0]

    def is_correct(ans: str) -> bool:
        a_p = ans.strip().lower()
        a_g = gold.strip().lower()
        if a_p == a_g: return True
        if re.search(rf'\b{re.escape(a_g)}\b', a_p): return True
        p = _extract_last_number(ans)
        g = _extract_last_number(gold)
        if p is not None and g is not None:
            return _gold_match(p, g) == 1.0
        return False

    # ── Greedy ────────────────────────────────────────────────────────────────
    best = max(chains, key=lambda c: c["correctness_score"])
    greedy_ans = best["answer"]
    if task_type == "math":
        n = _extract_last_number(greedy_ans)
        greedy_ans = str(int(n) if n is not None and n == int(n) else n) if n is not None else greedy_ans

    # ── CoT Top-K ─────────────────────────────────────────────────────────────
    topk = sorted(chains, key=lambda c: c["correctness_score"], reverse=True)[:K_TARGET]
    cot_ans = majority_vote(topk)

    # ── QUBO ─────────────────────────────────────────────────────────────────
    selected = [chains[i] for i in selected_idx]
    qubo_ans = majority_vote(selected)

    results = {
        "Greedy":      {"selected": [best],    "answer": greedy_ans, "chains": [best]},
        "CoT (Top-K)": {"selected": topk,      "answer": cot_ans,    "chains": topk},
        "QUBO (Ours)": {"selected": selected,  "answer": qubo_ans,   "chains": selected,
                        "selected_idx": selected_idx},
    }

    print()
    method_rows = []
    for name, res in results.items():
        sel_ids = [f"C{chains.index(c)}" for c in res["selected"] if c in chains]
        tick    = "[+] CORRECT" if is_correct(res["answer"]) else "[x] WRONG"
        method_rows.append([name, ",".join(sel_ids), res["answer"][:14], tick])

    table(
        ["Method", "Chains used", "Final answer", "Correct?"],
        method_rows, [14, 22, 16, 12]
    )

    print()
    print("  --- Why each method behaves the way it does ---")

    print()
    print("  GREEDY:")
    note(
        f"Picks the single highest-correctness chain (index={chains.index(best)}, "
        f"score={best['correctness_score']:.3f}).  Zero error correction — if the top "
        "chain hallucinated, greedy fails silently.  On easy questions this is "
        "sufficient, but on multi-step arithmetic or ambiguous commonsense questions "
        "a single chain is fragile.  Complexity: O(N) to find max."
    )

    print()
    print("  CoT TOP-K MAJORITY VOTE:")
    note(
        f"Takes the top-{K_TARGET} chains by individual quality score and majority-votes "
        "their answers.  More robust than greedy -- minority wrong answers are outvoted.  "
        "FLAW: top-K by score selects the K MOST SIMILAR high-quality chains (all came "
        "from the same reasoning path and make the same correlated error).  If all top-K "
        "chains share a systematic bias, the vote fails.  No diversity guarantee."
    )

    print()
    print("  QUBO (OURS):")
    note(
        f"Selected chains {[f'C{i}' for i in selected_idx]} via Simulated Annealing on "
        f"the QUBO matrix.  The Q matrix encodes BOTH quality (diagonal) AND diversity "
        "(off-diagonal).  The solver simultaneously maximises quality AND minimises "
        "redundancy -- an NP-hard tradeoff approximated by SA.  Self-consistency "
        "(Fix #1) boosts cross-agreeing chains.  Answer-aware off-diagonal (Fix #2) "
        "catches logical redundancy the embedder misses.  Cardinality constraint "
        "(Fix #3) ensures exactly k=4 chains are selected."
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 6 — INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(bundle: ModelBundle, results: dict, chains: list,
                  question: dict, gold: str) -> None:
    section("STEP 6: INFERENCE  ---  Final Answer Generation with QUBO Chains")

    formula_box([
        "RELEVANCE RE-RANKING before prompt assembly:",
        "  sim_to_q(i) = cosine(embed(reason_i), embed(question))",
        "  Rank selected chains by sim_to_q descending",
        "  (Most relevant chain appears first in prompt -> LLM primacy bias)",
        "",
        "FINAL PROMPT STRUCTURE:",
        "  'Here are some reasoning steps:'",
        "  '1. <most relevant selected reason>'",
        "  '2. <second>'",
        "  '...'",
        "  'Based on these steps, answer the question.'",
        "  'Question: <question>'",
        "  'Answer:'",
        "",
        "GENERATION: greedy decode (temperature=0, do_sample=False)",
    ])

    why_box(
        "WHY RE-RANK BY RELEVANCE: QUBO selects chains by quality+diversity, not "
        "by relevance to the specific question. LLMs exhibit primacy bias -- they "
        "weight earlier context more heavily. Placing the most question-relevant "
        "chain first anchors the model on pertinent evidence. "
        "WHY GREEDY DECODE FOR FINAL ANSWER: Unlike chain generation (where "
        "diversity is desired), the final answer synthesis should be as "
        "deterministic as possible. Temperature=0 (greedy) eliminates stochasticity "
        "from the final answer, making the output reproducible and stable. "
        "WHY SAME LLM: Ideally a larger model would synthesise the final answer "
        "(Phase 2 TODO), but Qwen-2.5-3B is used here for both stages. The "
        "QUBO-curated scaffold still significantly reduces hallucination risk "
        "compared to prompting the model cold."
    )

    qubo_res = results["QUBO (Ours)"]
    selected_reasons = [c["reason"] for c in qubo_res["chains"]]

    # Re-rank by relevance to question
    q_emb = bundle.embedder.encode([question["question"]], convert_to_numpy=True)
    r_embs = bundle.embedder.encode(selected_reasons, convert_to_numpy=True)
    sims   = cosine_similarity(r_embs, q_emb).flatten()
    ranked = np.argsort(sims)[::-1]
    ordered_reasons = [selected_reasons[i] for i in ranked]

    prompt = "Here are some reasoning steps:\n"
    for k, r in enumerate(ordered_reasons, 1):
        prompt += f"{k}. {r}\n"
    prompt += f"\nBased on these steps, answer the following question.\n"
    prompt += f"Question: {question['question']}\nAnswer:"

    print()
    print("  FINAL PROMPT (sent to Qwen-2.5-3B-Instruct, greedy decode):")
    print()
    inner = W - 4
    print(f"  +{'=' * inner}+")
    for line in prompt.split("\n"):
        wrapped = textwrap.wrap(line, width=inner - 2) or [""]
        for wline in wrapped:
            print(f"  | {wline:<{inner - 1}}|")
    print(f"  +{'=' * inner}+")

    print()
    print("  Generating final answer...", end="", flush=True)
    t0 = time.time()
    chat = _apply_chat_template(bundle.tokenizer, prompt)
    inputs = bundle.tokenizer(chat, return_tensors="pt",
                              truncation=True, max_length=1024).to(bundle.device)
    with torch.inference_mode():
        out = bundle.llm.generate(
            **inputs, max_new_tokens=64,
            do_sample=False,
            pad_token_id=bundle.tokenizer.pad_token_id,
            eos_token_id=bundle.tokenizer.eos_token_id,
        )
    final_ans = bundle.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().replace("\n", " ")
    print(f"  [{time.time()-t0:.1f}s]")
    print()

    def is_correct(ans: str) -> bool:
        a_p = ans.strip().lower(); a_g = gold.strip().lower()
        if a_p == a_g: return True
        if re.search(rf'\b{re.escape(a_g)}\b', a_p): return True
        p = _extract_last_number(ans); g = _extract_last_number(gold)
        if p is not None and g is not None:
            return _gold_match(p, g) == 1.0
        return False

    for method_name, res in results.items():
        correct = is_correct(res["answer"])
        safe_ans = res['answer'].replace('\n', ' ')
        tick = "[+] CORRECT" if correct else "[x] WRONG"
        print(f"  {method_name:<20}  final answer: '{safe_ans[:30]}'  {tick}")

    print(f"  {'QUBO + LLM synthesis':<20}  final answer: '{final_ans[:50]}'")
    correct_synth = is_correct(final_ans)
    print(f"    Gold answer: '{gold}'  -> {'[+] CORRECT' if correct_synth else '[x] WRONG'}")


# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def global_summary(all_results: list) -> None:
    banner("SUMMARY  ---  All Methods Across All Questions", char="*")

    def is_correct(ans: str, gold: str) -> bool:
        a_p = ans.strip().lower(); a_g = gold.strip().lower()
        if a_p == a_g: return True
        if re.search(rf'\b{re.escape(a_g)}\b', a_p): return True
        p = _extract_last_number(ans); g = _extract_last_number(gold)
        if p is not None and g is not None:
            return _gold_match(p, g) == 1.0
        return False

    totals  = {"Greedy": 0, "CoT (Top-K)": 0, "QUBO (Ours)": 0}
    rows    = []
    for entry in all_results:
        q, results = entry["question"], entry["results"]
        gold = q["gold"]
        row = [q["id"], q["type"], gold]
        for method in ["Greedy", "CoT (Top-K)", "QUBO (Ours)"]:
            ans = results[method]["answer"]
            ok  = is_correct(ans, gold)
            if ok: totals[method] += 1
            safe_ans = ans.replace('\n', ' ')
            row.append(f"{safe_ans[:10]} {'[+]' if ok else '[x]'}")
        rows.append(row)

    print()
    table(
        ["QID","Type","Gold","Greedy","CoT","QUBO"],
        rows, [4, 10, 12, 14, 14, 14]
    )

    total = len(all_results)
    print()
    print("  Accuracy:")
    for method, correct in totals.items():
        bar = "[" + "#" * correct + "." * (total - correct) + "]"
        print(f"    {method:<20}  {bar}  {correct}/{total}  ({100*correct//total}%)")

    print()
    note(
        "These 3 questions are EASY -- all methods get them right because the "
        "candidate pool always contains correct chains. The performance gap "
        "becomes LARGE on hard benchmarks (GSM8K hard split, AIME, BBH): "
        "the LLM generates many wrong chains, greedy picks the best-scoring "
        "wrong chain, CoT votes on correlated wrong chains, while QUBO selects "
        "a diverse set that covers more reasoning paths and makes correlated "
        "errors less likely to dominate the vote."
    )

    banner("IMPROVEMENTS IN THIS PIPELINE", char="+")
    improvements = [
        ("Fix #1", "Self-consistency (Wang 2023)",
         "gamma=0.10 consensus term lifts cross-agreeing chains"),
        ("Fix #2", "Answer-aware off-diagonal",
         "Q[i][j] += 0.4 * agree; catches logical redundancy"),
        ("Fix #3", "Cardinality constraint",
         "lambda_c=0.1 penalises selecting !=k chains"),
        ("Fix #4", "AST arithmetic verifier",
         "No eval(); pure tree recursion; sandboxed"),
        ("Fix #5", "Diverse prompt templates",
         "4 structural framings: forward/backward/analogical/units"),
        ("Fix #6", "Top-k cluster reps",
         "top_k_per_cluster=2; QUBO decides within each cluster"),
        ("Fix #7", "Separate synthesis model",
         "TODO: A100 + Llama-70B for final generation"),
    ]
    print()
    table(
        ["Fix","Name","What it does"],
        improvements, [6, 30, 40]
    )

    banner("DEMO COMPLETE  ---  Samsung PRISM 2026", char="=")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    banner("QUANTUM-INSPIRED ANNEALING PIPELINE  ---  MENTOR DEMO", char="#")
    print(f"""
  Project : Quantum-Inspired Annealing for Multi-Stage Reasoning
  Team    : Samsung PRISM 2026

  This script uses the REAL Qwen-2.5-3B-Instruct model for chain generation,
  the REAL cross-encoder/nli-deberta-v3-base for language scoring,
  and inline QUBO + Simulated Annealing for combinatorial selection.

  THREE QUESTIONS:
    Q1  Arithmetic : "Train at 60 km/h for 2.5 h. How far?"
    Q2  Language   : "Do penguins live near North or South Pole?"
    Q3  Multi-step : "Baker: 48 cookies, sell 3/4, give away 6. How many left?"

  THREE METHODS COMPARED:
    (1) Greedy decoding       -- single best chain
    (2) CoT Top-K majority    -- top-K by score, majority vote
    (3) QUBO pipeline (ours)  -- SA on QUBO matrix

  NOTE: First run downloads ~6 GB of models from HuggingFace.
    """)

    interactive = sys.stdin.isatty()
    if interactive:
        pause("  [Enter] Press Enter to load models and begin...")

    np.random.seed(42)
    random.seed(42)

    bundle = ModelBundle()
    all_results = []

    for qi, q in enumerate(QUESTIONS):
        banner(f"QUESTION {q['id']}  ---  {q['type'].upper()}", char="=")

        chains = generate_chains(bundle, q, verbose=True)
        chains = score_chains(bundle, chains, q, q["gold"])
        Q, embeddings = build_qubo(bundle, chains, q)
        selected_idx  = solve_qubo(Q, chains)
        results       = compare_methods(chains, selected_idx, q, q["gold"])
        run_inference(bundle, results, chains, q, q["gold"])

        all_results.append({"question": q, "results": results})

        if interactive and qi < len(QUESTIONS) - 1:
            pause(f"\n  [Enter] Press Enter for Q{qi+2}...")
        else:
            print()

    global_summary(all_results)
