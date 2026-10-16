"""CPU-friendly LoRA fine-tune (no GPU, no Unsloth). Small model, small data, slow but works.

    python training/train_cpu.py --data data/processed/spongebob.jsonl --out models/spongebob-lora
"""
import argparse, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from persona import SYSTEM_PROMPT
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForLanguageModeling)

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="HuggingFaceTB/SmolLM2-360M-Instruct")  # bigger: Qwen/Qwen2.5-0.5B-Instruct
ap.add_argument("--data", default="data/processed/spongebob.jsonl")
ap.add_argument("--out", default="models/spongebob-lora")
ap.add_argument("--max-samples", type=int, default=1500, help="cap data so it finishes in reasonable time")
ap.add_argument("--max-seq", type=int, default=256)
ap.add_argument("--epochs", type=float, default=1)
ap.add_argument("--lr", type=float, default=2e-4)
a = ap.parse_args()

torch.set_num_threads(max(1, torch.get_num_threads()))
tok = AutoTokenizer.from_pretrained(a.base)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float32)
model = get_peft_model(model, LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
model.print_trainable_parameters()

ds = load_dataset("json", data_files=a.data, split="train")
ds = ds.select(range(min(a.max_samples, len(ds))))

def encode(ex):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": ex["instruction"]},
            {"role": "assistant", "content": ex["response"]}]
    text = tok.apply_chat_template(msgs, tokenize=False)
    return tok(text, truncation=True, max_length=a.max_seq)

ds = ds.map(encode, remove_columns=ds.column_names)

Trainer(
    model=model, train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    args=TrainingArguments(
        output_dir="outputs", per_device_train_batch_size=1, gradient_accumulation_steps=8,
        num_train_epochs=a.epochs, learning_rate=a.lr, warmup_steps=5, logging_steps=5,
        save_strategy="no", report_to="none", use_cpu=True)).train()

model.save_pretrained(a.out)
tok.save_pretrained(a.out)
print("Saved adapter to", a.out)
