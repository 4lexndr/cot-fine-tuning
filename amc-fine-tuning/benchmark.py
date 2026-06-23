#!/usr/bin/env python3
"""Benchmark a HuggingFace model locally (CUDA) on all problems in test.jsonl.

Usage:
  python benchmark.py <model_id_or_path> [output_dir]
  python benchmark.py Qwen/Qwen2.5-1.5B-Instruct ./results/base
  python benchmark.py ./training-output ./results/finetuned
"""

import argparse
import json
import os
import random
import sys
import traceback

import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

PROBLEMS_FILE = "test.jsonl"
CLASSES = ["1-10", "11-20", "21-25"]
SYSTEM_MSG = "You are a math competition expert. Solve the following problem step by step."
MAX_NEW_TOKENS = 3048
DEFAULT_BATCH_SIZE = 4

_openai_client = OpenAI()

JUDGE_SYSTEM = (
    "You are a grader for AMC math competition answers. "
    "Given a problem and a model's response, identify the single letter (A, B, C, D, or E) "
    "that the model chose as its final answer. "
    "Reply with ONLY that single uppercase letter. "
    "Reply with NONE if: the model did not commit to any answer, OR the model's stated answer "
    "(whether expressed as a letter or a value) does not match any of the answer choices "
    "listed in the problem — i.e., the model hallucinated a value not present in the problem."
)


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
    )


def _judge_answer(problem_text, response_text):
    """Use OpenAI to extract the final answer letter from the model's response."""
    user_msg = (
        f"PROBLEM:\n{problem_text}\n\n"
        f"MODEL RESPONSE:\n{response_text}\n\n"
        "What single letter (A–E) did the model choose as its final answer?"
    )
    completion = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=5,
        temperature=0,
    )
    raw = completion.choices[0].message.content.strip().upper()
    if raw in ("A", "B", "C", "D", "E"):
        return raw
    return None


def _load_model(model_id):
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {model_id}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        print("Using Flash Attention 2.")
    except Exception:
        print("Flash Attention 2 unavailable, falling back to default attention.")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
    model.eval()
    print("Compiling model (first inference will be slow)...")
    model = torch.compile(model, mode="reduce-overhead")
    tokenizer.padding_side = "left"
    return tokenizer, model


def _generate_batch(tokenizer, model, messages_list):
    """Generate responses for a batch of conversations simultaneously."""
    text_inputs = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_list
    ]
    encoded = tokenizer(
        text_inputs,
        return_tensors="pt",
        padding=True,
        truncation=False,
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

    return [
        tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
        for out in output_ids
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a HuggingFace model on AMC problems.")
    parser.add_argument("model_id", help="HuggingFace model ID or local path")
    parser.add_argument("output_dir", nargs="?", default=".", help="Directory to write result files")
    parser.add_argument(
        "--manual-problems", type=int, default=None, metavar="N",
        help="Randomly sample N problems from the test set instead of using all",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, metavar="N",
        help=f"Number of problems to generate in parallel (default: {DEFAULT_BATCH_SIZE})",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU found.")
        sys.exit(1)

    model_id = args.model_id
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    tokenizer, model = _load_model(model_id)

    problems = _load_problems(PROBLEMS_FILE)
    if args.manual_problems is not None:
        if args.manual_problems >= len(problems):
            print(f"Warning: --manual-problems {args.manual_problems} >= dataset size {len(problems)}, using all problems.")
        else:
            problems = random.sample(problems, args.manual_problems)
            print(f"Sampled {len(problems)} problems randomly (--manual-problems={args.manual_problems})")
    print(f"Running on {len(problems)} problems from {PROBLEMS_FILE}\n")

    results_by_class = {c: {"correct": 0, "total": 0, "none": 0} for c in CLASSES}
    detail_log = []

    def _print_progress(label):
        print(f"\n--- {label} ---")
        for c in CLASSES:
            t = results_by_class[c]["total"]
            if t == 0:
                continue
            cor = results_by_class[c]["correct"]
            non = results_by_class[c]["none"]
            print(f"  Class {c}: {cor}/{t} = {round(cor/t*100,1)}%  (none: {non}/{t} = {round(non/t*100,1)}%)")
        total_so_far = sum(v["total"] for v in results_by_class.values())
        correct_so_far = sum(v["correct"] for v in results_by_class.values())
        if total_so_far:
            print(f"  Overall: {correct_so_far}/{total_so_far} = {round(correct_so_far/total_so_far*100,1)}%")
        print()

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
            predicted = _judge_answer(problem["problem"], generated)
            is_correct = predicted == correct_answer

            results_by_class[cls]["total"] += 1
            if is_correct:
                results_by_class[cls]["correct"] += 1
            if predicted is None:
                results_by_class[cls]["none"] += 1

            print(f"pred={predicted} correct={correct_answer} {'OK' if is_correct else 'WRONG'}")
            detail_log.append({
                "year": year,
                "contest": contest,
                "problem_num": num,
                "class": cls,
                "predicted": predicted,
                "correct": correct_answer,
                "is_correct": is_correct,
                "response": generated,
            })

        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            results_by_class[cls]["total"] += 1
            results_by_class[cls]["none"] += 1
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

        if (i + 1) % 5 == 0:
            _print_progress(f"Progress [{i+1}/{len(problems)}]")

    print("\n=== Results ===")
    summary = {}
    for cls in CLASSES:
        total = results_by_class[cls]["total"]
        correct = results_by_class[cls]["correct"]
        none = results_by_class[cls]["none"]
        accuracy = round(correct / total * 100, 1) if total > 0 else 0.0
        none_pct = round(none / total * 100, 1) if total > 0 else 0.0
        summary[cls] = {"accuracy": accuracy, "none_pct": none_pct}
        print(f"Class {cls}: {correct}/{total} = {accuracy}%  (none: {none}/{total} = {none_pct}%)")

    total_all = sum(v["total"] for v in results_by_class.values())
    correct_all = sum(v["correct"] for v in results_by_class.values())
    none_all = sum(v["none"] for v in results_by_class.values())
    overall = round(correct_all / total_all * 100, 1) if total_all > 0 else 0.0
    overall_none = round(none_all / total_all * 100, 1) if total_all > 0 else 0.0
    summary["overall"] = {"accuracy": overall, "none_pct": overall_none}
    print(f"Overall: {correct_all}/{total_all} = {overall}%  (none: {none_all}/{total_all} = {overall_none}%)")

    model_slug = model_id.replace(":", "-").replace("/", "-").replace("\\", "-").replace(".", "-")
    results_file = os.path.join(output_dir, f"results_{model_slug}.json")
    detail_file = os.path.join(output_dir, f"results_{model_slug}_detail.json")

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {results_file}: {summary}")

    with open(detail_file, "w") as f:
        json.dump(detail_log, f, indent=2)
    print(f"Saved detail log to {detail_file}")
