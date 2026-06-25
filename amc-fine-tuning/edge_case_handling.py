import json
from transformers import AutoTokenizer

# import some constants from training.py
TRUNCATION_LENGTH = 1500
SYSTEM_MESSAGE = "You are a math competition expert. Read the given problem closely, then solve it step by step."

DATA = "problems.jsonl"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
problems = [json.loads(l) for l in open(DATA, encoding="utf-8")]

cleaned = []
removed = []
for i, line in enumerate(problems):
    if len(tokenizer.encode(line["problem"] + line["reasoning"] + SYSTEM_MESSAGE)) > TRUNCATION_LENGTH:
        removed.append(i)
        continue
    cleaned.append(line)


print(f"Removed {len(problems) - len(cleaned)} lines: {removed}")
with open(DATA, "w", encoding="utf-8") as f:
    for p in cleaned:
        f.write(json.dumps(p) + "\n")

print(f"Done! Wrote data to {DATA}")