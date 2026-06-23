import sys
import json
import torch
import bitsandbytes
from datasets import Dataset
from transformers import (
    AutoTokenizer, # tokenization
    AutoModelForCausalLM, # loading model
    BitsAndBytesConfig, # quantization
    EarlyStoppingCallback, # overfitting prevention
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

# constants ------------
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_DATA = "./train.jsonl"
EVAL_DATA = "./eval.jsonl"
OUTPUT_DIR = "./finetuned"

# hyperparams ----------
TRUNCATION_LENGTH = 4096 # reasoning chains are very long and truncation is NOT GOOD
BATCH_SIZE = 2
GRADIENT_ACCUM = 4 # effective batch size: 2 * 4 = 8
EPOCHS = 4 # controlled by EarlyStoppingCallback
ALPHA = 2e-4
WEIGHT_DECAY = 0.02 # fights overfitting
WARMUP_RATIO = 0.05 # gives weights time to adjust
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.005 # minimum improvement to reset patience counter

LORA_R = 14
LORA_ALPHA = 28
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# system message -------
SYSTEM_MESSAGE = "You are a math competition expert. Read the problem closely, then solve it step by step."

# helpers --------------
def build_message(problem: str, reasoning: str):
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": reasoning} 
    ]

def load_dataset(path: str, tokenizer):
    with open (path) as f:
        rows = [json.loads(line) for line in f]
    
    for num, row in enumerate(rows):
        if not (row["problem"] and row["answer"] and row["reasoning"]):
            sys.exit(f"Row {num} in {path} has a bad format")
    
    cleaned = [{
        "text": tokenizer.apply_chat_template(
            build_message(row), 
            tokenize=False,
            add_generation_prompt=False,
        )
    } for row in rows]

    return Dataset.from_list(cleaned)

# main code ------------
if not torch.cuda.is_available():
    sys.exit("No CUDA GPU found")

# load tokenizer and train/eval data
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
train_data = load_dataset(TRAIN_DATA, tokenizer)
eval_data = load_dataset(EVAL_DATA, tokenizer)

# load model, bnb_config, lora_config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    quantization_config=bnb_config,
    device_map="cuda",
    trust_remote_code=True,
)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=LORA_TARGET_MODULES,
    task_type="CAUSAL_LM",
    use_rslora=True,
)

# training hyperparameters
training_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM,
    learning_rate=ALPHA,
    weight_decay=WEIGHT_DECAY,
    lr_scheduler_type="cosine",
    warmup_ratio=WARMUP_RATIO,
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    save_strategy="steps",
    eval_strategy="steps",
    save_steps=25,
    eval_steps=25,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=4,
    logging_steps=10,
    report_to="none",
    max_length=TRUNCATION_LENGTH,
    dataset_text_field="text",
    completion_only_loss=True,  # compute loss on assistant RESPONSES only
)

# early stopping callback
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
)

# define trainer
trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=train_data,
    eval_dataset=eval_data,
    processing_class=tokenizer,
    peft_config=lora_config,
    callbacks=[early_stopping]
)

print("Starting training! Good luck!")
trainer.train()

# save both parameters and tokenizer config in output directory
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

