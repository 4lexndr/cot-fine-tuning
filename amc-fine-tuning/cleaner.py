#!/usr/bin/env python3
"""Clean and validate problem text in JSONL dataset files via OpenAI API.

For each problem, the model either:
  - Returns cleaned problem text (replaces the existing field), or
  - Returns the sentinel DELETE if the problem is incomplete (missing its
    core mathematical expression), in which case the row is dropped.

Usage:
  python cleaner.py                              # process test.jsonl and train.jsonl
  python cleaner.py --db test.jsonl              # process one file
  python cleaner.py --db test.jsonl train.jsonl  # process specific files
"""

import argparse
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

DEFAULT_FILES = ["test.jsonl", "train.jsonl"]
CHECKPOINT_EVERY = 30
MAX_WORKERS = 30
DELETE = "DELETE"
MAX_RETRIES = 6

client = OpenAI()

PROBLEM_SYSTEM = """You are a dataset validator and cleaner for AMC math competition problems.
You will receive the raw problem text for a single AMC problem.

FIRST, check whether the problem is COMPLETE. A complete problem must have:
- A self-contained question whose mathematical content is fully present in the text.
- All expressions, variables, and context needed to solve it.

A problem is INCOMPLETE if the stem is clearly missing its core content, for example:
- "What is the value of" followed immediately by answer choices with nothing in between.
- "Which of the following is equivalent to" with no expression given.
- "What is the area of the region defined by" with no equation or description.
- "Let" followed immediately by a question with no definition of the variable or function.
- Any other case where the question cannot be understood or solved from the text alone.

If the problem is INCOMPLETE, respond with exactly the single word: DELETE

Otherwise, clean the problem text so it contains ONLY the problem statement:
- Remove trailing headings such as "Solution", "Solution 1", "See Also", "Video Solution", etc.
- Remove stray HTML tags, wiki markup, or navigation artifacts.
- Remove category labels or any text unrelated to the problem.
- Keep all answer choices (A) through (E).
- Keep all LaTeX math notation exactly as-is (dollar signs, backslashes, etc.).
- Do not rephrase or alter the math content in any way.
- Output ONLY the cleaned problem text — no commentary, no labels, no preamble."""


def clean_or_delete(text: str) -> str:
    delay = 1.0
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": PROBLEM_SYSTEM},
                    {"role": "user", "content": text},
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise


def process_file(path: str) -> None:
    with open(path) as f:
        problems = [json.loads(line) for line in f if line.strip()]

    print(f"[{path}] Loaded {len(problems)} problems")

    to_process = [(i, p) for i, p in enumerate(problems) if p.get("problem")]

    lock = threading.Lock()
    completed = 0
    delete_indices = set()

    def worker(i, problem):
        result = clean_or_delete(problem["problem"])
        return i, result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker, i, p): i for i, p in to_process}

        for future in as_completed(futures):
            try:
                i, result = future.result()
                p = problems[i]
                label = f"{p['year']} AMC 10{p.get('contest','')} P{p['problem_num']}"

                if result == DELETE:
                    delete_indices.add(i)
                    print(f"[{path}] {label} — DELETED (incomplete)")
                else:
                    problems[i]["problem"] = result
                    print(f"[{path}] {label} — cleaned")

            except Exception as e:
                i = futures[future]
                print(f"[{path}] index {i} — ERROR: {e}")

            with lock:
                completed += 1
                if completed % CHECKPOINT_EVERY == 0:
                    _write(path, problems, delete_indices)
                    print(f"[{path}] >> Checkpoint at {completed}/{len(to_process)}")

    _write(path, problems, delete_indices)
    kept = len(problems) - len(delete_indices)
    print(f"[{path}] Done. {len(delete_indices)} deleted, {kept} kept.")


def _write(path: str, problems: list, delete_indices: set) -> None:
    with open(path, "w") as f:
        for i, p in enumerate(problems):
            if i not in delete_indices:
                f.write(json.dumps(p) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db", nargs="+", default=DEFAULT_FILES, metavar="FILE",
        help="JSONL file(s) to process (default: test.jsonl train.jsonl)",
    )
    args = parser.parse_args()

    for path in args.db:
        process_file(path)
