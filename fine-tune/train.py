# train.py

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import torch
import os

# Make sure you're using a GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration for our model
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
dataset_path = 'D:\SML-chatbot\data\data.jsonl'
output_dir = "./output"
num_train_epochs = 5
learning_rate = 2e-4
per_device_train_batch_size = 2
save_steps = 10
logging_steps = 5

# === LOAD TOKENIZER AND MODEL ===
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# === LOAD DATASET ===
dataset = load_dataset("json", data_files=dataset_path, split="train")

# Format the dataset for training
def format_example(example):
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    return {"text": prompt}

dataset = dataset.map(format_example)

# Tokenize the dataset
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize)

# === LoRA CONFIGURATION ===
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1
)

model = get_peft_model(model, peft_config)

# === TRAINING CONFIG ===
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    num_train_epochs=num_train_epochs,
    learning_rate=learning_rate,
    logging_steps=logging_steps,
    save_steps=save_steps,
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# === DATA COLLATOR ===
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === TRAINER ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# === START TRAINING ===
trainer.train()

# === SAVE FINAL MODEL ===
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"\n✅ Fine-tuning complete! Model saved at: {output_dir}")
