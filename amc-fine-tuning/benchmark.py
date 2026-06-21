#!/usr/bin/env python3
"""Benchmark a model via Ollama on AMC 10 problems loaded from a JSONL dataset.

Usage:
  python benchmark.py
  python benchmark.py --contest A --years 2015-2025 --problems 1-25
  python benchmark.py --contest both --host http://10.0.4.34:11434 --model qwen2-math:7b
"""

import argparse
import json
import re
import sys
import urllib.request
import urllib.error

DEFAULT_PROBLEMS_FILE = "problems.jsonl"
DEFAULT_HOST = "http://10.0.4.34:11434"
DEFAULT_MODEL = "qwen2.5:1.5B"

CLASSES = ["1-10", "11-20", "21-25"]


def _parse_range(s, label):
    """Parse '2015-2025' → [2015..2025] or '3,5,7' → [3, 5, 7]."""
    s = s.strip()
    if "," not in s and "-" in s:
        lo, _, hi = s.partition("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in s.split(",")]


def _load_problems(path, years, problem_nums, contests):
    """Load and filter problems from a JSONL file."""
    year_set = set(years)
    num_set = set(problem_nums)
    contest_set = set(contests)

    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Records without a 'contest' field are legacy AMC 10A entries
            contest = rec.get("contest", "A")
            if rec["year"] in year_set and rec["problem_num"] in num_set and contest in contest_set:
                problems.append(rec)
    return problems


def _build_prompt(problem_text):
    return (
        f"Solve this AMC multiple choice problem. "
        f"Give your final answer as a single letter (A, B, C, D, or E).\n\n"
        f"{problem_text}\n\n"
        f"Final answer:"
    )


def _extract_answer(response_text):
    """Extract A/B/C/D/E from model output, trying several common formats."""
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


def _ollama_chat(host, model, messages):
    url = f"{host}/api/chat"
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


def _check_connection(host, model):
    try:
        req = urllib.request.Request(f"{host}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        available = [m["name"] for m in data.get("models", [])]
        print(f"Connected to Ollama at {host}")
        print(f"Available models: {available}")
        if not any(model in m for m in available):
            print(f"WARNING: '{model}' not found. Run: ollama pull {model}")
        return True
    except urllib.error.URLError as e:
        print(f"ERROR: Cannot reach Ollama at {host}: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark a model on AMC 10 problems via Ollama.")
    parser.add_argument("--years", default="2015-2025",
                        help="Year range or comma-list (default: 2015-2025)")
    parser.add_argument("--problems", default="1-25",
                        help="Problem range or comma-list (default: 1-25)")
    parser.add_argument("--contest", choices=["A", "B", "both"], default="A",
                        help="Contest filter: A, B, or both (default: A)")
    parser.add_argument("--host", default=DEFAULT_HOST,
                        help=f"Ollama host URL (default: {DEFAULT_HOST})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--input", default=DEFAULT_PROBLEMS_FILE,
                        help=f"Problems JSONL file (default: {DEFAULT_PROBLEMS_FILE})")
    parser.add_argument("--output", default="results_ollama",
                        help="Output filename prefix (default: results_ollama)")
    args = parser.parse_args()

    years = _parse_range(args.years, "years")
    problem_nums = _parse_range(args.problems, "problems")
    contests = ["A", "B"] if args.contest == "both" else [args.contest]

    if not _check_connection(args.host, args.model):
        sys.exit(1)

    all_problems = _load_problems(args.input, years, problem_nums, contests)
    print(f"Loaded {len(all_problems)} problems from {args.input}")

    scoreable = [p for p in all_problems if p.get("answer")]
    print(f"Problems with answer keys: {len(scoreable)}")

    results_by_class = {c: {"correct": 0, "total": 0} for c in CLASSES}
    detail_log = []

    for i, problem in enumerate(scoreable):
        year = problem["year"]
        num = problem["problem_num"]
        cls = problem["class"]
        contest = problem.get("contest", "A")
        correct_answer = problem["answer"]

        print(f"[{i+1}/{len(scoreable)}] {year} AMC 10{contest} P{num} (class {cls})...", end=" ", flush=True)

        messages = [
            {"role": "system", "content": "You are a math competition expert."},
            {"role": "user", "content": _build_prompt(problem["problem"])},
        ]

        try:
            generated = _ollama_chat(args.host, args.model, messages)
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
                "response": generated[:200],
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

    results_file = f"{args.output}.json"
    detail_file = f"{args.output}_detail.json"

    with open(results_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {results_file}: {summary}")

    with open(detail_file, "w") as f:
        json.dump(detail_log, f, indent=2)
    print(f"Saved detail log to {detail_file}")
