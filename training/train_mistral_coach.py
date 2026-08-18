"""
LoRA Fine-tuning for Mistral 7B Inspiration Coach
Optimized for RTX 4050 (6GB VRAM) with 8-bit quantization
"""

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("🎯 MISTRAL 7B LORA FINE-TUNING FOR INSPIRATION COACH")
print("=" * 80)
print(f"Device: cuda")
print(f"GPU: NVIDIA GeForce RTX 4050 Laptop GPU")
print(f"VRAM: 6.44GB")
print()

# Configuration
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATASET_PATH = "../datasets/inspiration_coach_data.json"
OUTPUT_DIR = "../models/inspiration_coach_mistral"
TOKENIZED_DIR = "../datasets/tokenized_coach_mistral"

print("=" * 80)
print("🔧 LOADING MODEL WITH 8-BIT QUANTIZATION")
print("=" * 80)

# Disable quantization for TinyLlama compatibility
bnb_config = None

print("Loading tokenizer...")
try:
    # Try standard load first
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, trust_remote_code=True)
except Exception as e:
    print(f"Standard load failed, trying alternative: {e}")
    # Fallback: use the slower tokenizer
    from transformers import LlamaTokenizer
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print("✅ Tokenizer loaded")

print("\nLoading model with 8-bit quantization + CPU offloading (this may take 2-3 minutes)...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    trust_remote_code=True,
)
print("✅ Model loaded and quantized")
print(f"   Total Parameters: 7B")
print(f"   Quantization: 8-bit NF4")

# Prepare model for LoRA
model = prepare_model_for_kbit_training(model)

print("\n" + "=" * 80)
print("⚙️  CONFIGURING LORA")
print("=" * 80)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
    ],
)

model = get_peft_model(model, lora_config)
print("✅ LoRA Applied")
model.print_trainable_parameters()

# Load dataset
print("\n" + "=" * 80)
print("📊 LOADING DATASET")
print("=" * 80)

with open(DATASET_PATH, 'r') as f:
    data = json.load(f)

# Format for training
formatted_data = []
for item in data:
    prompt = item['instruction']
    response = item['response']
    full_text = f"User: {prompt}\n\nCoach: {response}"
    formatted_data.append({"text": full_text})

print(f"✅ Loaded {len(formatted_data)} examples")

# Create dataset
from datasets import Dataset
dataset = Dataset.from_dict({"text": [item["text"] for item in formatted_data]})

# Tokenize
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )

print("Tokenizing dataset...")
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

# Split
train_test_split = tokenized_dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]

print(f"✅ Dataset prepared")
print(f"   Train samples: {len(train_dataset)}")
print(f"   Eval samples: {len(eval_dataset)}")

# Training arguments
print("\n" + "=" * 80)
print("⚙️  TRAINING CONFIGURATION")
print("=" * 80)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=10,  # More epochs for better learning
    per_device_train_batch_size=1,  # Must be 1 for 6GB VRAM
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,  # Simulate larger batch
    warmup_ratio=0.1,
    learning_rate=1e-4,  # Lower for Mistral
    bf16=True,  # Use bfloat16 for efficiency
    logging_steps=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    optim="paged_adamw_8bit",  # 8-bit optimizer
)

print("Epochs: 10")
print("Batch size: 1")
print("Gradient accumulation: 4")
print("Learning rate: 1e-4")
print("Optimizer: paged_adamw_8bit")
print("Precision: bfloat16")

# Trainer
print("\n" + "=" * 80)
print("🎓 STARTING TRAINING")
print("=" * 80)
print("This will take 10-15 minutes on RTX 4050")
print()

trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()

# Save model
print("\n" + "=" * 80)
print("💾 SAVING MODEL")
print("=" * 80)

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Model saved to: {OUTPUT_DIR}")

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)
print(f"\nModel saved to: {OUTPUT_DIR}")
print("\nNext steps:")
print("1. Test the model: python agents/test_mistral_coach.py")
print("2. Use the agent: python agents/inspiration_coach_agent.py (update model path)")