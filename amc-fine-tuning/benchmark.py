#!/usr/bin/env python3
"""Benchmark a model via Ollama on all problems in test.jsonl.

Usage:
  python benchmark.py <model>
  python benchmark.py qwen2-math:1.5b
"""

import json
import re
import sys
import urllib.request
import urllib.error

HOST = "http://10.0.4.34:11434"
PROBLEMS_FILE = "test.jsonl"
CLASSES = ["1-10", "11-20", "21-25"]


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


def _ollama_chat(model, messages):
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


def _check_connection(model):
    try:
        req = urllib.request.Request(f"{HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        available = [m["name"] for m in data.get("models", [])]
        print(f"Connected to Ollama at {HOST}")
        print(f"Available models: {available}")
        if not any(model in m for m in available):
            print(f"WARNING: '{model}' not found. Run: ollama pull {model}")
        return True
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot reach Ollama at {HOST}: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python benchmark.py <model>")
        sys.exit(1)

    model = sys.argv[1]

    if not _check_connection(model):
        sys.exit(1)

    problems = _load_problems(PROBLEMS_FILE)
    print(f"Loaded {len(problems)} problems from {PROBLEMS_FILE}")

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
            {"role": "system", "content": "You are a math competition expert."},
            {"role": "user", "content": _build_prompt(problem["problem"])},
        ]

        try:
            generated = _ollama_chat(model, messages)
            predicted = _extract_answer(generated)
            is_correct = predicted == correct_answer

            results_by_class[cls]["total"] += 1
            if is_correct:
                results_by_class[cls]["correct"] += 1

            print(f"pred={predicted} correct={correct_answer} {'✓' if is_correct else '✗'}")
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

    model_slug = model.replace(":", "-").replace("/", "-")
    results_file = f"results_{model_slug}.json"
    detail_file = f"results_{model_slug}_detail.json"

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {results_file}: {summary}")

    with open(detail_file, "w") as f:
        json.dump(detail_log, f, indent=2)
    print(f"Saved detail log to {detail_file}")
