from api import request
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

DATABASE = "all_problems.jsonl"
CHECKPOINT_EVERY = 10
MAX_WORKERS = 20

with open(DATABASE) as f:
    problems = [json.loads(line) for line in f if line.strip()]

print(f"Loaded {len(problems)} problems from {DATABASE}")
already_done = sum(1 for p in problems if p.get("reasoning"))
print(f"Already have reasoning: {already_done}")

to_process = [
    (i, p) for i, p in enumerate(problems)
    if p.get("solution") and not p.get("reasoning")
]
print(f"To process: {len(to_process)}")

lock = threading.Lock()
completed = 0

def process(i, problem):
    reasoning = request(problem["problem"], problem["solution"])
    return i, reasoning


with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {executor.submit(process, i, p): i for i, p in to_process}

    for future in as_completed(futures):
        try:
            i, reasoning = future.result()
            problems[i]["reasoning"] = reasoning
            p = problems[i]
            print(f"{p['year']} AMC 10{p['contest']} P{p['problem_num']} — done")
        except Exception as e:
            i = futures[future]
            print(f"[index {i}] ERROR: {e}")

        with lock:
            completed += 1
            if completed % CHECKPOINT_EVERY == 0:
                with open(DATABASE, "w") as f:
                    for p in problems:
                        f.write(json.dumps(p) + "\n")
                print(f"  >> Checkpoint saved at {completed}")

with open(DATABASE, "w") as f:
    for problem in problems:
        f.write(json.dumps(problem) + "\n")

print(f"\nDone. Wrote updated data to {DATABASE}")
print(f"Total with reasoning: {sum(1 for p in problems if p.get('reasoning'))}/{len(problems)}")