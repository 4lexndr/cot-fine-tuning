#!/usr/bin/env python3
"""Clean problem and solution columns in all_problems.jsonl via OpenAI API.

Strips garbage HTML, headings, author signatures, and truncation artifacts
from the 'problem' and 'solution' fields of each row.
"""

import json
import os
from openai import OpenAI

DATABASE = "all_problems.jsonl"
CHECKPOINT_EVERY = 20

client = OpenAI()

PROBLEM_SYSTEM = """You are a dataset cleaner for AMC math competition problems.

You will receive raw text scraped from AoPS wiki for a single AMC problem.
Clean it so it contains ONLY the problem statement itself.

Rules:
- Remove any trailing headings such as "Solution", "Solution 1", "See Also", "Video Solution", etc.
- Remove any stray HTML tags, wiki markup, or navigation artifacts.
- Remove any lines that are headings, category labels, or unrelated to the problem.
- Keep the full problem statement text, including all answer choices (A) through (E).
- Keep all LaTeX math notation exactly as-is (dollar signs, backslashes, etc.).
- Do not rephrase or alter the math content in any way.
- Output ONLY the cleaned problem text — no commentary, no labels, no preamble.
"""

SOLUTION_SYSTEM = """You are a dataset cleaner for AMC math competition solutions.

You will receive raw text scraped from AoPS wiki for a single solution.
Clean it so it contains ONLY the solution content itself.

Rules:
- Remove author signatures such as "~username", "- username", "By username", or similar.
- Remove headings like "Solution 1", "Solution 2", "Video Solution", "See Also", etc.
- If there are multiple numbered solutions, keep only Solution 1 (the first one).
- Remove any stray HTML tags, wiki markup, or navigation artifacts.
- Keep all LaTeX math notation exactly as-is (dollar signs, backslashes, etc.).
- Do not rephrase or alter the math content in any way.
- Output ONLY the cleaned solution text — no commentary, no labels, no preamble.
"""


def clean_field(text: str, system_prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def main():
    with open(DATABASE) as f:
        problems = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(problems)} problems from {DATABASE}")

    for i, prob in enumerate(problems):
        label = f"[{i+1}/{len(problems)}] {prob.get('year')} AMC 10{prob.get('contest')} P{prob.get('problem_num')}"

        problem_text = prob.get("problem", "")
        solution_text = prob.get("solution", "")

        changed = False

        if problem_text:
            try:
                cleaned_problem = clean_field(problem_text, PROBLEM_SYSTEM)
                if cleaned_problem != problem_text:
                    prob["problem"] = cleaned_problem
                    changed = True
                print(f"{label} — problem cleaned")
            except Exception as e:
                print(f"{label} — problem ERROR: {e}")

        if solution_text:
            try:
                cleaned_solution = clean_field(solution_text, SOLUTION_SYSTEM)
                if cleaned_solution != solution_text:
                    prob["solution"] = cleaned_solution
                    changed = True
                print(f"{label} — solution cleaned")
            except Exception as e:
                print(f"{label} — solution ERROR: {e}")

        if not problem_text and not solution_text:
            print(f"{label} — skipped (no content)")

        if (i + 1) % CHECKPOINT_EVERY == 0:
            with open(DATABASE, "w") as f:
                for p in problems:
                    f.write(json.dumps(p) + "\n")
            print(f"  >> Checkpoint saved at {i+1}")

    with open(DATABASE, "w") as f:
        for prob in problems:
            f.write(json.dumps(prob) + "\n")

    print(f"\nDone. Wrote cleaned data to {DATABASE}")


if __name__ == "__main__":
    main()
