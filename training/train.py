"""LoRA fine-tune with Unsloth. Runs on a free Colab T4 (16 GB).

    python training/train.py --data data/processed/spongebob.jsonl --out models/spongebob-lora
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from persona import SYSTEM_PROMPT

from unsloth import FastLanguageModel  # import first so Unsloth patches transformers
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="unsloth/Llama-3.2-3B-Instruct")  # or unsloth/mistral-7b-v0.3-bnb-4bit
ap.add_argument("--data", default="data/processed/spongebob.jsonl")
ap.add_argument("--out", default="models/spongebob-lora")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--max-seq", type=int, default=1024)
ap.add_argument("--lr", type=float, default=2e-4)
a = ap.parse_args()

model, tok = FastLanguageModel.from_pretrained(
    model_name=a.base, max_seq_length=a.max_seq, dtype=None, load_in_4bit=True)

model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth", random_state=42)

def to_text(ex):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["instruction"]},
            {"role": "assistant", "content": ex["response"]}]
    return {"text": tok.apply_chat_template(msgs, tokenize=False)}

ds = load_dataset("json", data_files=a.data, split="train").map(to_text)

trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds, dataset_text_field="text",
    max_seq_length=a.max_seq, packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2, gradient_accumulation_steps=4,
        warmup_steps=10, num_train_epochs=a.epochs, learning_rate=a.lr,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="linear", seed=42, output_dir="outputs", report_to="none"))
trainer.train()

model.save_pretrained(a.out)   # LoRA adapter only (small)
tok.save_pretrained(a.out)
print("Saved adapter to", a.out)
