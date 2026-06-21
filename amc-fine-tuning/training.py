#!/usr/bin/env python3
"""Fine-tune BASE_MODEL with LoRA and AdamW on the AMC reasoning dataset.

Requirements (install before running):
  pip install peft datasets

Usage:
  python training.py
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

# ─── Configuration ──────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = "train.jsonl"
OUTPUT_DIR = "lora-finetuned"

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

MAX_LENGTH = 2048
BATCH_SIZE = 4
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05

# ─── Dataset ────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = (
    "You are a math competition expert. Solve the following AMC problem step by step.\n\n"
    "Problem:\n{problem}\n\n"
    "Reasoning:\n{reasoning}\n\n"
    "Final answer: {answer}"
)


def format_example(problem: dict) -> str:
    return PROMPT_TEMPLATE.format(
        problem=problem.get("problem", ""),
        reasoning=problem.get("reasoning", problem.get("solution", "")),
        answer=problem.get("answer", ""),
    )


class AMCDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(path) as f:
            raw = [json.loads(line) for line in f if line.strip()]
        # Keep only entries that have a problem and at least a solution/reasoning
        self.examples = [
            r for r in raw
            if r.get("problem") and (r.get("reasoning") or r.get("solution"))
        ]
        print(f"Dataset: {len(self.examples)} usable examples from {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = format_example(self.examples[idx])
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze()
        attention_mask = encoded["attention_mask"].squeeze()
        # Labels are the same as input_ids for causal LM; mask padding tokens
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading tokenizer and model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = AMCDataset(TRAIN_FILE, tokenizer, MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        optim="adamw_torch",
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="none",
        dataloader_drop_last=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    print("Starting fine-tuning...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
