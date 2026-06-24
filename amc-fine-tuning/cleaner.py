import sys
import json
import time
from openai import OpenAI

DATA = "./problems.jsonl"
MAX_RETRIES = 6
DELETE = "DELETE" # sentinel the model returns for incomplete rows

client = OpenAI()
API_MODEL = "gpt-4o-mini"

# Evaluates only the problem field. Never sees the solution.
PROBLEM_SYSTEM_MESSAGE = \
    "You are a dataset cleaner for AMC math competition problems.\n\n" \
    "You will receive only the problem text. Evaluate it purely on structure — do NOT\n" \
    "attempt to solve it, evaluate expressions, or check answer-choice correctness.\n\n" \
    "--- DELETE if ANY of these structural defects is present ---\n\n" \
    "1. NO answer choices: Every valid AMC problem ends with answer choices (A) through (E),\n" \
    "   formatted with \\textbf{(A)}, \\mathrm{(A)}, or similar. If the problem has NO such\n" \
    "   answer-choice line at all, DELETE — it is not a complete AMC problem.\n\n" \
    "2. Cut-off: The problem text ends mid-sentence with no question and no answer choices\n" \
    "   (e.g. 'How many integers satisfy the condition:' followed by nothing).\n\n" \
    "3. HTML / wiki artifacts: Raw HTML tags (<ref>, <br/>, <math>, etc.) or unclosed wiki\n" \
    "   markup ([[, ==, {{) appear anywhere in the problem text.\n\n" \
    "4. Post-choice leak: Any prose or derivation text appears AFTER the last answer-choice\n" \
    "   line — a valid problem ends right after choice (E).\n\n" \
    "--- NEVER DELETE for these reasons ---\n\n" \
    "- The problem says 'in the figure', 'as shown', or references a diagram — keep it.\n" \
    "- Answer choices use \\textbf{}, \\text{}, \\mathrm{}, or any LaTeX text command.\n" \
    "- The problem defines an operation, gives a condition, or states equations — these\n" \
    "  are part of the question, not a defect.\n" \
    "- When in doubt: KEEP.\n\n" \
    "If deleting: reply with exactly DELETE and nothing else.\n" \
    "Otherwise: output only the cleaned problem text. Strip any wiki/navigation headings\n" \
    "(Solution, See Also, Video Solution, etc.) that trail after the answer choices.\n" \
    "Keep all LaTeX exactly as-is. Do not rephrase, rewrite, or add anything."

# Evaluates only the solution field. Never sees the problem.
SOLUTION_SYSTEM_MESSAGE = \
    "You are a dataset validator for AMC math competition solutions.\n\n" \
    "You will receive only the solution text.\n\n" \
    "Rule 1 — explicit answer: if the solution contains a \\boxed{} expression, or explicitly\n" \
    "names an answer choice (A), (B), (C), (D), or (E), or ends with a clear mathematical\n" \
    "conclusion (e.g. 'Thus x = 5', 'Therefore the answer is 3', 'the solutions are x=y or y=0'),\n" \
    "it is complete. Do NOT delete it.\n\n" \
    "Rule 2 — HTML artifacts: DELETE if the solution contains raw HTML tags\n" \
    "(<ref>, <br/>, <math>, etc.) or unclosed wiki markup ([[, ==, {{).\n\n" \
    "Rule 3 — truly cut off: DELETE only if the solution ends mid-sentence or mid-calculation\n" \
    "with NO conclusion at all — not a stated value, not an answer choice, not a boxed answer.\n" \
    "Example cut-off endings: 'Add those two together to get', 'Substitute back into s.',\n" \
    "'Since both expressions represent the same length, you can set them equal to each other.'\n\n" \
    "When in doubt: reply OK.\n\n" \
    "If deleting: reply with exactly DELETE and nothing else.\n" \
    "Otherwise: reply with exactly OK and nothing else."

# helpers --------------
def _call(system: str, user: str) -> str | None:
    delay = 1
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=API_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
            else:
                print(f"API error, leaving row unchanged: {e}")
                return None
    return None

def clean_row(num: int, row: dict):
    problem_result = _call(PROBLEM_SYSTEM_MESSAGE, row["problem"])
    if problem_result == DELETE:
        return num, DELETE

    solution_result = _call(SOLUTION_SYSTEM_MESSAGE, row["solution"])
    if solution_result == DELETE:
        return num, DELETE

    return num, problem_result

# main code ------------
# load and validate problems
with open(DATA, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(rows)} problems from {DATA}")

for num, row in enumerate(rows):
    if not (row.get("problem") and row.get("solution")):
        sys.exit(f"Row {num} is missing a problem or solution")

# clean rows sequentially
results = {}
for num, row in enumerate(rows):
    num, result = clean_row(num, row)
    results[num] = result
    if result is None:
        print(f"Row {num} could not be cleaned; keeping original")
    elif result == DELETE:
        print(f"Row {num} is incomplete; deleting")
    else:
        print(f"Row {num} cleaned")

# write kept rows back to file
kept = []
for num, row in enumerate(rows):
    result = results[num]
    if result == DELETE:
        continue
    if result is not None:
        row["problem"] = result
    kept.append(row)

with open(DATA, "w", encoding="utf-8") as f:
    for row in kept:
        f.write(json.dumps(row) + "\n")

print(f"Done! {len(kept)}/{len(rows)} rows kept.")
