import json
import random

DATABASE = "problems.jsonl"
TRAIN_FILE = "train.jsonl"
EVAL_FILE = "eval.jsonl"
TEST_FILE = "test.jsonl"

TEST_FRACTION = 0.20
EVAL_FRACTION = 0.10
SEED = 42

def main():
    with open(DATABASE, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(problems)} problems from {DATABASE}")

    by_class = {}
    for p in problems:
        cls = p.get("class", "unknown")
        by_class.setdefault(cls, []).append(p)

    rng = random.Random(SEED)
    train, eval_, test = [], [], []

    for cls, group in sorted(by_class.items()):
        rng.shuffle(group)
        n_test = max(1, round(len(group) * TEST_FRACTION))
        n_eval = max(1, round(len(group) * EVAL_FRACTION))
        test.extend(group[:n_test])
        eval_.extend(group[n_test:n_test + n_eval])
        train.extend(group[n_test + n_eval:])

    rng.shuffle(train)
    rng.shuffle(eval_)
    rng.shuffle(test)

    for path, split in [(TRAIN_FILE, train), (EVAL_FILE, eval_), (TEST_FILE, test)]:
        with open(path, "w", encoding="utf-8") as f:
            for p in split:
                f.write(json.dumps(p) + "\n")

    total = len(problems)
    print(f"Train: {len(train):4d} ({len(train)/total*100:.1f}%) -> {TRAIN_FILE}")
    print(f"Eval:  {len(eval_):4d} ({len(eval_)/total*100:.1f}%) -> {EVAL_FILE}")
    print(f"Test:  {len(test):4d} ({len(test)/total*100:.1f}%) -> {TEST_FILE}")
    print()

    for cls in sorted(by_class):
        n = len(by_class[cls])
        n_tr = sum(1 for p in train if p.get("class") == cls)
        n_ev = sum(1 for p in eval_ if p.get("class") == cls)
        n_te = sum(1 for p in test if p.get("class") == cls)
        print(f"  Class {cls}: {n} total -> {n_tr} train / {n_ev} eval / {n_te} test")

if __name__ == "__main__":
    main()
