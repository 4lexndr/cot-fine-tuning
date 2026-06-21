#!/usr/bin/env python3
"""Clean only the solution field for 2011-2020 rows (newly scraped solutions)."""
import json
from openai import OpenAI

DATABASE = "all_problems.jsonl"
CHECKPOINT_EVERY = 20
client = OpenAI()

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
- Output ONLY the cleaned solution text — no commentary, no labels, no preamble."""

def clean_solution(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SOLUTION_SYSTEM},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

with open(DATABASE) as f:
    problems = [json.loads(l) for l in f if l.strip()]

print(f"Loaded {len(problems)} problems")

target_years = set(range(2011, 2021))
targets = [(i, p) for i, p in enumerate(problems) if p.get("year") in target_years and p.get("solution")]
print(f"Cleaning solutions for {len(targets)} rows in 2011-2020")

for count, (i, prob) in enumerate(targets):
    label = f"[{count+1}/{len(targets)}] {prob['year']} 10{prob['contest']} P{prob['problem_num']}"
    sol = prob.get("solution", "")
    if not sol:
        print(f"{label} — skipped (no solution)")
        continue
    try:
        cleaned = clean_solution(sol)
        problems[i]["solution"] = cleaned
        print(f"{label} — cleaned")
    except Exception as e:
        print(f"{label} — ERROR: {e}")

    if (count + 1) % CHECKPOINT_EVERY == 0:
        with open(DATABASE, "w") as f:
            for p in problems:
                f.write(json.dumps(p) + "\n")
        print(f"  >> Checkpoint at {count+1}")

with open(DATABASE, "w") as f:
    for p in problems:
        f.write(json.dumps(p) + "\n")
print(f"\nDone. Wrote cleaned data to {DATABASE}")
