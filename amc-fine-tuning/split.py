#!/usr/bin/env python3
"""Split all_problems.jsonl into train.jsonl (85%) and test.jsonl (15%).

Stratified by problem class so difficulty distribution is preserved.
"""

import json
import random

DATABASE = "all_problems.jsonl"
TRAIN_FILE = "train.jsonl"
TEST_FILE = "test.jsonl"
TEST_FRACTION = 0.15
SEED = 42


def main():
    with open(DATABASE) as f:
        problems = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(problems)} problems from {DATABASE}")

    # Group by class for stratified split
    by_class = {}
    for p in problems:
        cls = p.get("class", "unknown")
        by_class.setdefault(cls, []).append(p)

    rng = random.Random(SEED)
    train, test = [], []

    for cls, group in sorted(by_class.items()):
        rng.shuffle(group)
        n_test = max(1, round(len(group) * TEST_FRACTION))
        test.extend(group[:n_test])
        train.extend(group[n_test:])

    # Shuffle final lists so they aren't grouped by class
    rng.shuffle(train)
    rng.shuffle(test)

    with open(TRAIN_FILE, "w") as f:
        for p in train:
            f.write(json.dumps(p) + "\n")

    with open(TEST_FILE, "w") as f:
        for p in test:
            f.write(json.dumps(p) + "\n")

    print(f"Train: {len(train)} problems → {TRAIN_FILE}")
    print(f"Test:  {len(test)} problems  → {TEST_FILE}")
    print(f"Test fraction: {len(test) / len(problems) * 100:.1f}%")

    for cls in sorted(by_class):
        n = len(by_class[cls])
        n_test = sum(1 for p in test if p.get("class") == cls)
        print(f"  Class {cls}: {n} total → {n_test} test ({n_test/n*100:.1f}%)")


if __name__ == "__main__":
    main()
