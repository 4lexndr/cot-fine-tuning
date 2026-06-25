import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import json

import matplotlib
matplotlib.use("Agg")  # non-GUI backend; avoids errors when there is no display during training

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, # tokenization
    AutoModelForCausalLM, # loading model
    BitsAndBytesConfig, # quantization
    EarlyStoppingCallback, # overfitting prevention
    TrainerCallback, # logging loss 
)
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

# constants ------------
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_DATA = "./train.jsonl"
EVAL_DATA = "./eval.jsonl"
OUTPUT_DIR = "./finetuning-results"

# hyperparams ----------
TRUNCATION_LENGTH = 1500
BATCH_SIZE = 1
GRADIENT_ACCUM = 8 # effective batch size: 2 * 4 = 8
EPOCHS = 4 # controlled by EarlyStoppingCallback
ALPHA = 1e-4
WEIGHT_DECAY = 0.02 # fights overfitting
WARMUP_RATIO = 0.05 # gives weights time to adjust
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.0125 # minimum improvement to reset patience counter

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.06
# attention projections AND the MLP block — attention-only LoRA mostly restyles output;
# adding gate/up/down lets the adapter actually shift the model's computation
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# system message -------
SYSTEM_MESSAGE = "You are a math competition expert. Read the given problem closely, then solve it step by step."

# helpers --------------
def build_message(problem: str, reasoning: str):
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": reasoning} 
    ]

def load_dataset(path: str, tokenizer):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    
    for num, row in enumerate(rows):
        if not (row["problem"] and row["answer"] and row["reasoning"]):
            sys.exit(f"Row {num} in {path} has a bad format")
    
    cleaned = [{
        "text": tokenizer.apply_chat_template(
            build_message(row["problem"], row["reasoning"]), 
            tokenize=False,
            add_generation_prompt=False,
        )
    } for row in rows]

    return Dataset.from_list(cleaned)

# handle updated train and eval loss and plot them for visibility
class LossPlotCallback(TrainerCallback):
    def __init__(self):
        self.train_steps = []
        self.train_losses = []
        self.eval_steps = []
        self.eval_losses = []

    # called on a log event; check if it's new train loss -> add to graph
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.train_steps.append(state.global_step)
            self.train_losses.append(logs["loss"])
            self.save_plot()

    # called on an evaluation metric; check if it's new eval loss -> add to graph
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            self.eval_steps.append(state.global_step)
            self.eval_losses.append(metrics["eval_loss"])
            self.save_plot()

    def save_plot(self):
        fig, ax = plt.subplots()

        if self.train_steps:
            ax.plot(self.train_steps, self.train_losses, color="steelblue", label="Train loss", linewidth=1.5)
        if self.eval_steps:
            ax.plot(self.eval_steps, self.eval_losses, color="darkorange", label="Eval loss", linewidth=1.5)

        all_steps = self.train_steps + self.eval_steps
        if all_steps:
            x_min = min(all_steps)
            x_max = max(all_steps)
            span = x_max - x_min if x_max > x_min else 1
            ax.set_xlim(left=x_min - span * 0.05)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Progress")
        ax.legend()
        fig.tight_layout()
        fig.savefig("model_loss.png", dpi=120)
        plt.close(fig)

# main code ------------
if __name__ == "__main__":
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
        attn_implementation="sdpa",
        quantization_config=bnb_config,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.config.use_cache = False # disable KV cache for training

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
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
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
        save_steps=50,
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

    loss_plot = LossPlotCallback()

    # define trainer
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        peft_config=lora_config,
        callbacks=[early_stopping, loss_plot]
    )

    print("Starting training! Good luck!")
    trainer.train()

    # save both parameters and tokenizer config in output directory
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
