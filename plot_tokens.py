import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# same system message from trainer.py
SYSTEM_MESSAGE = "You are a math competition expert. Read the given problem closely, then solve it step by step."

DATA = "./problems.jsonl"
with open(DATA) as f:
    problems = [json.loads(line) for line in f if line.strip()]

# tokenize using native AutoTokenizer to get accurate token counts
enc = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
token_counts = [
    len(enc.encode(p["reasoning"] + SYSTEM_MESSAGE + p["problem"]))
    for p in problems
]

x = 1500 # testing possible TRUNCATION_LENGTH values
avg = sum(token_counts) / len(token_counts)
print(f"Entries with reasoning: {len(token_counts)}")
print(f"Avg tokens:  {avg:.1f}")
print(f"Min tokens:  {min(token_counts)}")
print(f"Max tokens:  {max(token_counts)}")

entries_above_x = sum(c > x for c in token_counts)
print(f"\nNumber of entries above {x}: {entries_above_x}; will lose {(entries_above_x/len(problems))}% of data")

plt.figure(figsize=(10, 5))
plt.hist(token_counts, bins=40, color="steelblue", edgecolor="white")
plt.axvline(avg, color="red", linestyle="--", label=f"Mean: {avg:.0f}")
plt.xlabel("Token count")
plt.ylabel("Number of problems")
plt.title("Token distribution of reasoning field")
plt.legend()
plt.tight_layout()
plt.savefig("token_distribution.png", dpi=150)
plt.show()
print("Saved to token_distribution.png")
