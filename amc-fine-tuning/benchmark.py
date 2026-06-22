#!/usr/bin/env python3
"""Benchmark a HuggingFace model locally (CUDA) on all problems in test.jsonl.

Usage:
  python benchmark.py <model_id_or_path> [output_dir]
  python benchmark.py Qwen/Qwen2-Math-1.5B-Instruct
  python benchmark.py Qwen/Qwen2.5-1.5B-Instruct ./initial_benchmarks/qwen2.5-1.5b
  python benchmark.py ./training-output ./results
"""

import json
import os
import re
import sys
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

PROBLEMS_FILE = "test.jsonl"
CLASSES = ["1-10", "11-20", "21-25"]
SYSTEM_MSG = "You are a math competition expert. Solve the following problem step by step."
MAX_NEW_TOKENS = 1024


def _load_problems(path):
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            problems.append(json.loads(line))
    return problems


def _build_prompt(problem_text):
    return (
        f"Solve this AMC multiple choice problem. "
        f"Give your final answer as a single letter (A, B, C, D, or E).\n\n"
        f"{problem_text}\n\n"
        f"Final answer:"
    )


def _extract_answer(response_text):
    text = response_text.strip()

    m = re.search(r"(?:final answer|answer)[:\s]*\(?([A-E])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\\boxed\{([A-E])\}", text)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-E])\b\s*$", text)
    if m:
        return m.group(1).upper()

    m = re.search(r"\(([A-E])\)", text)
    if m:
        return m.group(1).upper()

    m = re.search(r"textbf\{?\(?([A-E])\)?", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-E])\b", text)
    if m:
        return m.group(1).upper()

    return None


def _load_model(model_id):
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def _generate(tokenizer, model, messages):
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    prompt_len = input_ids.shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python benchmark.py <model_id_or_path> [output_dir]")
        sys.exit(1)

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU found.")
        sys.exit(1)

    model_id = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) == 3 else "."
    os.makedirs(output_dir, exist_ok=True)
    tokenizer, model = _load_model(model_id)

    problems = _load_problems(PROBLEMS_FILE)
    print(f"Loaded {len(problems)} problems from {PROBLEMS_FILE}\n")

    results_by_class = {c: {"correct": 0, "total": 0} for c in CLASSES}
    detail_log = []

    for i, problem in enumerate(problems):
        year = problem["year"]
        num = problem["problem_num"]
        cls = problem["class"]
        contest = problem.get("contest", "A")
        correct_answer = problem["answer"]

        print(f"[{i+1}/{len(problems)}] {year} AMC 10{contest} P{num} (class {cls})...", end=" ", flush=True)

        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": _build_prompt(problem["problem"])},
        ]

        try:
            generated = _generate(tokenizer, model, messages)
            predicted = _extract_answer(generated)
            is_correct = predicted == correct_answer

            results_by_class[cls]["total"] += 1
            if is_correct:
                results_by_class[cls]["correct"] += 1

            print(f"pred={predicted} correct={correct_answer} {'OK' if is_correct else 'WRONG'}")
            detail_log.append({
                "year": year,
                "contest": contest,
                "problem_num": num,
                "class": cls,
                "predicted": predicted,
                "correct": correct_answer,
                "is_correct": is_correct,
                "response": generated[:1024],
            })

        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            results_by_class[cls]["total"] += 1
            detail_log.append({
                "year": year,
                "contest": contest,
                "problem_num": num,
                "class": cls,
                "predicted": None,
                "correct": correct_answer,
                "is_correct": False,
                "error": str(e),
            })

    print("\n=== Results ===")
    summary = {}
    for cls in CLASSES:
        total = results_by_class[cls]["total"]
        correct = results_by_class[cls]["correct"]
        pct = round(correct / total * 100, 1) if total > 0 else 0.0
        summary[cls] = pct
        print(f"Class {cls}: {correct}/{total} = {pct}%")

    total_all = sum(v["total"] for v in results_by_class.values())
    correct_all = sum(v["correct"] for v in results_by_class.values())
    overall = round(correct_all / total_all * 100, 1) if total_all > 0 else 0.0
    print(f"Overall: {correct_all}/{total_all} = {overall}%")

    model_slug = model_id.replace(":", "-").replace("/", "-").replace("\\", "-").replace(".", "-")
    results_file = os.path.join(output_dir, f"results_{model_slug}.json")
    detail_file = os.path.join(output_dir, f"results_{model_slug}_detail.json")

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {results_file}: {summary}")

    with open(detail_file, "w") as f:
        json.dump(detail_log, f, indent=2)
    print(f"Saved detail log to {detail_file}")
