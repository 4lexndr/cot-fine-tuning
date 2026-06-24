import json
from openai import OpenAI

DATA = "./problems.jsonl"
CHECKPOINT_EVERY = 10

API_MODEL = "gpt-4o"
SYSTEM_PROMPT = """
You are a math reasoning assistant. Your job is to produce a clear, step-by-step reasoning trace for an AMC competition problem.

--- YOUR INPUT ---
You will receive two clearly separated fields:

  PROBLEM: <the problem statement, including answer choices (A)–(E)>
  SOLUTION: <a worked solution that arrives at the correct answer>

These are TWO DISTINCT INPUTS. Do not treat them as one continuous block of text.
The SOLUTION is your primary source. The PROBLEM is context only.

--- YOUR TASK ---
Rewrite the solution as a first-person reasoning trace — the internal monologue of a student
working through the problem step by step. Your output will be used as training data for a small
language model (Qwen2.5-1.5B-Instruct), so clarity, consistency, and correctness are critical.

Follow these rules strictly:

1. PRIMARY SOURCE IS THE SOLUTION.
   Base every reasoning step on what the solution actually does. Do not invent steps, shortcuts,
   or alternative methods that are not present in the solution.

2. USE THE PROBLEM FOR CONTEXT ONLY.
   Read the problem to understand what is being asked, what variables are defined, and what the
   answer choices are. Reference it when you need to name a variable, quote a condition, or
   identify which answer choice is correct. Do not re-derive the answer from the problem alone.

3. WRITE IN EXPLICIT STEPS.
   Each logical step should be its own sentence or short paragraph. Do not skip arithmetic or
   algebraic manipulations — write them out. A student should be able to follow every transition.

4. NO FABRICATION.
   If the solution is ambiguous or skips a step, bridge the gap with the minimal logical inference
   needed. Never introduce facts, formulas, or sub-problems that are not grounded in the solution.

5. END WITH THE ANSWER.
   The final sentence must state the answer clearly, e.g.:
   "The answer is (C) 14." or "Therefore the answer is \\boxed{14}, which corresponds to choice (C)."

6. FORMAT.
   - Plain prose with inline LaTeX for math (e.g. $x^2 + 1$, \\boxed{42}).
   - No headers, bullet points, or section labels.
   - No meta-commentary ("The solution tells us...", "As given above...").
   - Length should match the complexity of the solution — typically 100–400 words.
"""

# retrieve data from database
with open(DATA) as f:
    problems = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(problems)} problems from {DATA}")
already_done = sum(1 for p in problems if p.get("reasoning"))
print(f"Already have reasoning: {already_done}")

to_process = [(i, p) for i, p in enumerate(problems) if p.get("solution") and not p.get("reasoning")]
print(f"To process: {len(to_process)}")

client = OpenAI()

# helpers --------------
def structure_message(problem: str, solution: str):
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": f"PROBLEM: {problem}\nSOLUTION: {solution}"
        }
    ]

def get_reasoning(content: tuple[str, str]):
    response = client.chat.completions.create(
        model=API_MODEL,
        messages=structure_message(content[0], content[1]),
    )
    return response.choices[0].message.content

# main loop ------------
for completed, (i, line) in enumerate(to_process, 1):
    print(f"Processing {completed}/{len(to_process)} (index {i})", flush=True)

    try:
        content = (line["problem"], line["solution"])
        reasoning = get_reasoning(content)
        problems[i]["reasoning"] = reasoning
    except Exception as e:
        print(f"  [index {i}] ERROR: {e}")
        continue

    if completed % CHECKPOINT_EVERY == 0:
        with open(DATA, "w") as f:
            f.writelines(json.dumps(p) + "\n" for p in problems)
        print(f"  >> Checkpoint saved at {completed}")

with open(DATA, "w") as f:
    f.writelines(json.dumps(p) + "\n" for p in problems)

print(f"\nDone! Wrote updated data to {DATA}")
print(f"Total with reasoning: {sum(1 for p in problems if p.get('reasoning'))}/{len(problems)}")
