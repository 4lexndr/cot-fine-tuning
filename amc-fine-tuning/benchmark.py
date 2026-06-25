import sys
import json
import os
import re
import torch
from datetime import datetime

from openai import OpenAI
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# file constants -------
TEST_DATA = "./test.jsonl"
RESULTS_DIR = "./benchmark-results"
CLASSES = ["1-10", "11-20", "21-25"]
TRUNCATION_LENGTH = 2250 # allow slightly longer reasoning chains
JUDGE_MODEL = "o3-mini"

# system messages ------
MODEL_SYSTEM_MESSAGE = "You are a math competition expert. Read the given problem closely, then solve it step by step."
JUDGE_SYSTEM_MESSAGE = (
    "You are a math competition grader. You are given a problem and a student's response. "
    "Identify the letter or mathematical expression the student chose as their final answer. "
    "First, write one sentence describing what the student committed to (or why you can't identify one). "
    "Then on the next line write only: the single uppercase letter A–E, "
    "NONE (if the student didn't commit to any answer), "
    "or CORRUPT (if the response is truncated with no clear answer). "
    "If the student used a mathematical expression, map it to the corresponding letter."
)

openai = OpenAI() # OpenAI client

# helper functions ------
def load_problems(path: str):
    with open(path) as f:
        problems = [json.loads(line) for line in f]
    return problems

def build_message(problem: str):
    return [
        {"role": "system", "content": MODEL_SYSTEM_MESSAGE},
        {"role": "user", "content": problem},
    ]

def judge_answer(problem: str, model_response: str):
    completion = openai.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_MESSAGE},
            {"role": "user", "content": f"PROBLEM:\n{problem}\n\nSTUDENT RESPONSE:\n{model_response}"},
        ],
        # o3-mini is a reasoning model: hidden reasoning tokens count against max_completion_tokens
        reasoning_effort="low",
        max_completion_tokens=2000,
    )
    raw = completion.choices[0].message.content.strip()
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    predicted = lines[-1].upper() if lines else "NONE"
    thought = lines[0] if len(lines) > 1 else ""
    return predicted, thought

def make_run_dir(model_id: str) -> str:
    slug = re.sub(r"[^\w\-.]", "_", model_id).strip("._")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, f"{slug}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def generate_response(problem: str, tokenizer, model):
    prompt = tokenizer.apply_chat_template(
        build_message(problem),
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=TRUNCATION_LENGTH,
            do_sample=False, # greedy decoding for reproducible benchmarks
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(output[0][encoded["input_ids"].shape[-1]:], skip_special_tokens=True)

# main code ------------
if not torch.cuda.is_available():
    sys.exit("No CUDA GPU found")

# add model-specifying parameter
if len(sys.argv) < 2:
    sys.exit("Usage: python new_benchmark.py <model_id_or_path>")
MODEL = sys.argv[1]

# load tokenizer, model, and test data
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
)

# put model into evaluation mode instead of training mode
model.eval()
problems = load_problems(TEST_DATA)
run_dir = make_run_dir(MODEL)

# tally results per difficulty class (none = no answer, corrupt = truncated)
results = {class_: {"correct": 0, "total": 0, "none": 0, "corrupt": 0} for class_ in CLASSES}
detailed_log = []

# grade the model's performance on each problem
for num, problem in enumerate(problems):
    class_ = problem["class"]
    correct_answer = problem["answer"]

    print(f"[{num + 1}/{len(problems)}] {class_}", flush=True)
    response = generate_response(problem["problem"], tokenizer, model)

    chars = len(response)
    start = response[:60].replace("\n", " ")
    end = response[-60:].replace("\n", " ") if chars > 60 else ""
    print(f"  {chars} chars | \"{start}\" ... \"{end}\"", flush=True)

    predicted, thought = judge_answer(problem["problem"], response)
    marker = "✓" if predicted == correct_answer else "✗"
    print(f"  {predicted} / {correct_answer} {marker}", flush=True)
    if thought:
        print(f"  judge: {thought}", flush=True)

    results[class_]["total"] += 1
    if predicted == correct_answer:
        results[class_]["correct"] += 1
    elif predicted == "NONE":
        results[class_]["none"] += 1
    elif predicted == "CORRUPT":
        results[class_]["corrupt"] += 1

    detailed_log.append({
        "num": num + 1,
        "class": class_,
        "problem": problem["problem"],
        "correct_answer": correct_answer,
        "response": response,
        "predicted_answer": predicted,
        "judge_thought": thought,
        "is_correct": predicted == correct_answer,
    })

# print accuracy per class and overall
print("\n=== Results ===")
total_correct = 0
total_count = 0
total_none = 0
total_corrupt = 0
per_class_summary = {}
for class_ in CLASSES:
    correct = results[class_]["correct"]
    total = results[class_]["total"]
    none = results[class_]["none"]
    corrupt = results[class_]["corrupt"]
    total_correct += correct
    total_count += total
    total_none += none
    total_corrupt += corrupt
    accuracy = round(correct / total * 100, 1) if total else 0.0
    none_pct = round(none / total * 100, 1) if total else 0.0
    corrupt_pct = round(corrupt / total * 100, 1) if total else 0.0
    print(f"Class {class_}: {correct}/{total} = {accuracy}%  (none: {none} = {none_pct}%, corrupt: {corrupt} = {corrupt_pct}%)")
    per_class_summary[class_] = {
        "correct": correct, "total": total,
        "accuracy_pct": accuracy,
        "none": none, "none_pct": none_pct,
        "corrupt": corrupt, "corrupt_pct": corrupt_pct,
    }

overall = round(total_correct / total_count * 100, 1) if total_count else 0.0
overall_none = round(total_none / total_count * 100, 1) if total_count else 0.0
overall_corrupt = round(total_corrupt / total_count * 100, 1) if total_count else 0.0
print(f"Overall: {total_correct}/{total_count} = {overall}%  (none: {total_none} = {overall_none}%, corrupt: {total_corrupt} = {overall_corrupt}%)")

# save results
with open(os.path.join(run_dir, "detailed_log.json"), "w") as f:
    json.dump(detailed_log, f, indent=2)

performance = {
    "model": MODEL,
    "timestamp": datetime.now().isoformat(),
    "overall": {
        "correct": total_correct, "total": total_count,
        "accuracy_pct": overall,
        "none": total_none, "none_pct": overall_none,
        "corrupt": total_corrupt, "corrupt_pct": overall_corrupt,
    },
    "per_class": per_class_summary,
}
with open(os.path.join(run_dir, "performance.json"), "w") as f:
    json.dump(performance, f, indent=2)

print(f"\nResults saved to {run_dir}")
