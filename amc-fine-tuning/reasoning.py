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
You will receive three clearly separated fields:

  PROBLEM: <the problem statement, including answer choices (A)–(E)>
  SOLUTION: <a worked solution that arrives at the correct answer>
  CORRECT ANSWER: <the answer choice letter (A)–(E) that is known to be correct>

These are DISTINCT INPUTS. Do not treat them as one continuous block of text.
The SOLUTION is your primary source. The PROBLEM is context. The CORRECT ANSWER is the ground
truth your trace MUST arrive at through honest reasoning.

--- YOUR TASK ---
Rewrite the solution as a first-person reasoning trace — the internal monologue of a careful
student working through the problem step by step. Your output is training data for a small
language model (Qwen2.5-1.5B-Instruct). The single most important property is CORRECTNESS: every
arithmetic and algebraic step must be right, and the trace must genuinely DERIVE the correct
answer rather than assert it. A fluent trace that reaches the answer by hand-waving is worse than
useless — it teaches the model to fake confidence.

Follow these rules strictly:

1. PRIMARY SOURCE IS THE SOLUTION.
   Base every reasoning step on what the solution actually does. Do not invent steps, shortcuts,
   or alternative methods that are not present in the solution.

2. USE THE PROBLEM FOR CONTEXT ONLY.
   Read the problem to understand what is being asked, what variables are defined, and what the
   answer choices are. Do not re-derive the answer from the problem alone.

3. WRITE IN EXPLICIT, VERIFIABLE STEPS.
   Each logical step is its own sentence or short paragraph. Do not skip arithmetic or algebraic
   manipulations — write them out so each transition can be checked. Carry exact values; never
   round mid-way and never paper over a computation with "it can be shown" or "clearly".

4. EVERY COMPUTATION MUST BE CORRECT.
   Before you commit to a number, make sure it is actually right. If a step in the source solution
   looks wrong or unclear, recompute it yourself and write the correct version. Do not reproduce an
   error. Do not invent facts, formulas, or sub-problems that are not grounded in the solution.

5. VERIFY BEFORE CONCLUDING.
   The second-to-last step must be an explicit self-check that confirms the derived result: e.g.
   substitute the value back into the original equation/condition, re-add the terms, recount the
   cases, or sanity-check units/magnitude. State the result of this check. The reasoning you reach
   MUST match the CORRECT ANSWER. If your honest derivation does not land on the correct answer,
   you have made an error — find and fix it; never bridge the gap by simply declaring the answer.

6. END WITH THE ANSWER.
   The final sentence states the answer clearly and consistently with the work above, e.g.:
   "The answer is (C) 14." or "Therefore the answer is \\boxed{14}, which corresponds to choice (C)."

7. FORMAT.
   - Plain prose with inline LaTeX for math (e.g. $x^2 + 1$, \\boxed{42}).
   - No headers, bullet points, or section labels.
   - No meta-commentary ("The solution tells us...", "The correct answer is given as...").
   - Length should match the complexity of the solution — typically 100–400 words.
"""

# helpers --------------
def structure_message(problem: str, solution: str, answer: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"PROBLEM: {problem}\nSOLUTION: {solution}\nCORRECT ANSWER: {answer}"},
    ]

def get_reasoning(client: OpenAI, problem: str, solution: str, answer: str) -> str:
    response = client.chat.completions.create(
        model=API_MODEL,
        messages=structure_message(problem, solution, answer),
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
            problems[i]["reasoning"] = get_reasoning(client, problems[i]["problem"], problems[i]["solution"], problems[i]["answer"])
        except Exception as e:
            print(f"\n  [index {i}] ERROR: {e}")
            continue

        if completed - last_checkpoint >= CHECKPOINT_EVERY:
            save(DATA, problems)
            last_checkpoint = completed

    save(DATA, problems)
    print(f"\nDone! Wrote updated data to {DATA}")
