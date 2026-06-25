import json
from openai import OpenAI

# constants ------------
DATA = "./problems.jsonl"
CHECKPOINT_EVERY = 50
API_MODEL = "o3-mini"

# system message -------
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

# helpers --------------
def structure_message(problem: str, solution: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"PROBLEM: {problem}\nSOLUTION: {solution}"},
    ]

def get_reasoning(client: OpenAI, problem: str, solution: str) -> str:
    response = client.chat.completions.create(
        model=API_MODEL,
        messages=structure_message(problem, solution),
    )
    return response.choices[0].message.content

def save(path: str, problems: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p) + "\n" for p in problems)

# main code ------------
if __name__ == "__main__":
    with open(DATA, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]

    client = OpenAI()
    indices = [i for i, p in enumerate(problems) if "reasoning" not in p]

    print(f"Found {len(indices)} problems without reasoning ({len(problems)} total).")

    last_checkpoint = 0
    for completed, i in enumerate(indices, 1):
        print(f"\rProcessing {completed}/{len(indices)}", flush=True, end="")
        try:
            problems[i]["reasoning"] = get_reasoning(client, problems[i]["problem"], problems[i]["solution"])
        except Exception as e:
            print(f"\n  [index {i}] ERROR: {e}")
            continue

        if completed - last_checkpoint >= CHECKPOINT_EVERY:
            save(DATA, problems)
            last_checkpoint = completed

    save(DATA, problems)
    print(f"\nDone! Wrote updated data to {DATA}")
