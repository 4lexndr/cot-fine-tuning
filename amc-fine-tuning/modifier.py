from api import request
import json

DATABASE = "all_problems.jsonl"
CHECKPOINT_EVERY = 10

with open(DATABASE) as f:
    problems = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(problems)} problems from {DATABASE}")
already_done = sum(1 for p in problems if p.get("reasoning"))
print(f"Already have reasoning: {already_done}")

for i, problem in enumerate(problems):
    solution = problem.get("solution")

    if not solution:
        print(f"[{i+1}/{len(problems)}] Skipping — no solution")
        continue

    if problem.get("reasoning"):
        print(f"[{i+1}/{len(problems)}] Skipping — reasoning already present")
        continue

    try:
        problem["reasoning"] = request(problem["problem"], solution)
        print(f"[{i+1}/{len(problems)}] {problem['year']} AMC 10{problem['contest']} P{problem['problem_num']} — done")
    except Exception as e:
        print(f"[{i+1}/{len(problems)}] ERROR: {e}")

    if (i + 1) % CHECKPOINT_EVERY == 0:
        with open(DATABASE, "w") as f:
            for p in problems:
                f.write(json.dumps(p) + "\n")
        print(f"  >> Checkpoint saved at {i+1}")

with open(DATABASE, "w") as f:
    for problem in problems:
        f.write(json.dumps(problem) + "\n")

print(f"\nDone. Wrote updated data to {DATABASE}")
print(f"Total with reasoning: {sum(1 for p in problems if p.get('reasoning'))}/{len(problems)}")
