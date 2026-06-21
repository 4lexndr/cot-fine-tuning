#!/usr/bin/env python3
"""Fine-tune Qwen2-Math-1.5B-Instruct with QLoRA on the AMC reasoning dataset.

After training the adapter is saved to OUTPUT_DIR.  If the remote Ollama host
is reachable the script prints exact steps to deploy and benchmark.  You can
also call run_benchmark() directly once the fine-tuned model is loaded into
Ollama.

Usage:
  python training.py
"""

import json
import re
import urllib.request
import urllib.error

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

# ─── Remote Ollama host (same as benchmark.py) ─────────────────────────────────
HOST = "http://10.0.4.34:11434"

# ─── Paths ─────────────────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2-Math-1.5B-Instruct"
TRAIN_FILE = "train.jsonl"
EVAL_FILE = "eval.jsonl"
TEST_FILE = "test.jsonl"
OUTPUT_DIR = "./training-output"

# ─── Training hyper-parameters ─────────────────────────────────────────────────
# Reduce MAX_SEQ_LENGTH (e.g. to 512) or BATCH_SIZE (to 1) if you hit OOM.
# Gradient checkpointing is already on; that's the next lever after those two.
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 2
GRAD_ACCUM = 4          # effective batch = BATCH_SIZE × GRAD_ACCUM = 8
NUM_EPOCHS = 5          # EarlyStoppingCallback will stop before this if needed
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01     # AdamW L2 regularisation; increase if model overfits
WARMUP_RATIO = 0.05
EARLY_STOPPING_PATIENCE = 2

# ─── LoRA ──────────────────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32         # α = 2r; rsLoRA rescales internally so this is safe
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",    # attention projections
    "gate_proj", "up_proj", "down_proj",         # MLP projections
]

# ─── Prompt helpers ────────────────────────────────────────────────────────────

SYSTEM_MSG = "You are a math competition expert. Solve the following AMC problem step by step."



def _build_messages(problem: dict) -> list[dict]:
    reasoning = problem.get("reasoning") or problem.get("solution", "")
    return [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": f"Problem:\n{problem['problem']}"},
        {"role": "assistant", "content": f"{reasoning}\n\nFinal answer: {problem['answer']}"},
    ]


def _build_inference_prompt(problem_text: str) -> str:
    return (
        "Solve this AMC multiple choice problem. "
        "Give your final answer as a single letter (A, B, C, D, or E).\n\n"
        f"{problem_text}\n\nFinal answer:"
    )


# ─── Dataset ───────────────────────────────────────────────────────────────────

def _load_split(path: str, tokenizer) -> Dataset:
    with open(path) as f:
        raw = [json.loads(line) for line in f if line.strip()]
    usable = [
        r for r in raw
        if r.get("problem") and r.get("answer")
        and (r.get("reasoning") or r.get("solution"))
    ]
    texts = [
        {
            "text": tokenizer.apply_chat_template(
                _build_messages(item), tokenize=False, add_generation_prompt=False
            )
        }
        for item in usable
    ]
    print(f"  {path}: {len(texts)} examples")
    return Dataset.from_list(texts)


def load_train_eval(train_path: str, eval_path: str, tokenizer) -> tuple[Dataset, Dataset]:
    print("Loading datasets...")
    train_ds = _load_split(train_path, tokenizer)
    eval_ds = _load_split(eval_path, tokenizer)
    return train_ds, eval_ds


# ─── Ollama connection (mirrors benchmark.py) ──────────────────────────────────

def check_connection() -> bool:
    try:
        req = urllib.request.Request(f"{HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        print(f"Connected to Ollama at {HOST}")
        print(f"Available models: {models}")
        return True
    except urllib.error.URLError as e:
        print(f"WARNING: Cannot reach Ollama at {HOST}: {e}")
        return False


def _ollama_chat(model: str, messages: list[dict]) -> str:
    url = f"{HOST}/api/chat"
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 1024},
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read())
    return body["message"]["content"]


def _extract_answer(text: str) -> str | None:
    for pattern in [
        r"(?:final answer|answer)[:\s]*\(?([A-E])\)?",
        r"\\boxed\{([A-E])\}",
        r"\b([A-E])\b\s*$",
        r"\(([A-E])\)",
        r"textbf\{?\(?([A-E])\)?",
        r"\b([A-E])\b",
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def run_benchmark(model: str, test_file: str = TEST_FILE) -> dict:
    """Benchmark a model on the test set via remote Ollama — mirrors benchmark.py."""
    with open(test_file) as f:
        problems = [json.loads(line) for line in f if line.strip()]

    CLASSES = ["1-10", "11-20", "21-25"]
    results = {c: {"correct": 0, "total": 0} for c in CLASSES}

    for i, problem in enumerate(problems):
        cls = problem["class"]
        correct = problem["answer"]
        tag = f"{problem['year']} AMC 10{problem.get('contest', 'A')} P{problem['problem_num']}"
        print(f"[{i+1}/{len(problems)}] {tag}...", end=" ", flush=True)
        try:
            messages = [
                {"role": "system", "content": "You are a math competition expert."},
                {"role": "user", "content": _build_inference_prompt(problem["problem"])},
            ]
            response = _ollama_chat(model, messages)
            predicted = _extract_answer(response)
            is_correct = predicted == correct
            results[cls]["total"] += 1
            if is_correct:
                results[cls]["correct"] += 1
            print(f"pred={predicted} correct={correct} {'✓' if is_correct else '✗'}")
        except Exception as e:
            print(f"ERROR: {e}")
            results[cls]["total"] += 1

    print("\n=== Post-training Benchmark ===")
    summary = {}
    for cls in CLASSES:
        t = results[cls]["total"]
        c = results[cls]["correct"]
        pct = round(c / t * 100, 1) if t > 0 else 0.0
        summary[cls] = pct
        print(f"Class {cls}: {c}/{t} = {pct}%")
    total = sum(v["total"] for v in results.values())
    correct_total = sum(v["correct"] for v in results.values())
    overall = round(correct_total / total * 100, 1) if total > 0 else 0.0
    print(f"Overall: {correct_total}/{total} = {overall}%")
    return summary


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        raise SystemExit("ERROR: No CUDA GPU found.")
    try:
        import bitsandbytes  
    except ImportError:
        raise SystemExit("ERROR: bitsandbytes not installed. Run: pip install bitsandbytes")

    remote_available = check_connection()

    print(f"\nLoading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, eval_ds = load_train_eval(TRAIN_FILE, EVAL_FILE, tokenizer)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    print(f"\nLoading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="cuda",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
        use_rslora=True,
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_steps=10,
        report_to="none",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        completion_only_loss=True,  # compute loss on assistant responses only
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    print("\nStarting fine-tuning...")
    trainer.train()

    print(f"\nSaving adapter to {OUTPUT_DIR}/")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    if remote_available:
        print(
            f"\nOllama at {HOST} is reachable. Once the fine-tuned adapter is "
            "loaded into Ollama, benchmark it with:\n"
            "  python benchmark.py <your-model-name>\n"
            "Or call run_benchmark('<your-model-name>') from this script."
        )
    else:
        print(f"\nOllama at {HOST} was unreachable. Start it and run benchmark.py when ready.")


if __name__ == "__main__":
    main()
