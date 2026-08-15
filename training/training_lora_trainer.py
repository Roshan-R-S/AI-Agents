"""
LoRA Fine-tuning Script for RTX 4050 - Using HuggingFace Trainer
Optimized for 16GB RAM + 6GB VRAM
Uses TinyLlama 1.1B model
ALL PROGRESS BARS DISABLED for PowerShell compatibility
"""

import torch
import json
import os
import logging
import sys
import gc

# Disable ALL progress bars and verbose output BEFORE imports
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Suppress transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.utils.hub").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, load_from_disk
from peft import LoraConfig, get_peft_model

# ============================================================
# DEVICE & HARDWARE SETUP
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n{'='*80}")
print("🎯 LORA FINE-TUNING FOR CUSTOM AI AGENTS")
print(f"{'='*80}")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    print(f"CUDA: {torch.version.cuda}")

# ============================================================
# CONFIGURATION
# ============================================================

class TrainingConfig:
    base_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    output_dir = None
    dataset_path = None
    
    num_epochs = 5
    train_batch_size = 1
    eval_batch_size = 1
    gradient_accumulation_steps = 2
    learning_rate = 5e-5
    warmup_steps = 50
    max_seq_length = 256
    
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05
    lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    logging_steps = 5
    eval_steps = 50
    save_steps = 100
    save_total_limit = 2
    
    weight_decay = 0.01
    max_grad_norm = 1.0

config = TrainingConfig()

# ============================================================
# LOAD MODEL & TOKENIZER
# ============================================================

def load_model_and_tokenizer(model_path=None):
    """Load base model and tokenizer"""
    
    if model_path is None:
        model_path = config.base_model
    
    print(f"\n{'='*80}")
    print("📥 LOADING MODEL AND TOKENIZER")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    
    print("Loading tokenizer...")
    sys.stdout.flush()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print("✅ Tokenizer loaded")
    sys.stdout.flush()
    
    print("\nLoading model (this may take 2-3 minutes)...")
    print("(Do not worry about the loading messages below)")
    sys.stdout.flush()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("✅ Model downloaded")
    print("✅ Model ready (distributed across GPU and CPU)")
    sys.stdout.flush()
    
    all_params = sum(p.numel() for p in model.parameters())
    
    print("✅ Model loaded and ready")
    print(f"   Total Parameters: {all_params / 1e9:.2f}B")
    sys.stdout.flush()
    
    return model, tokenizer

# ============================================================
# PREPARE MODEL FOR LORA
# ============================================================

def prepare_model_for_lora(model):
    """Prepare model for LoRA fine-tuning"""
    
    print(f"\n{'='*80}")
    print("🔧 PREPARING MODEL FOR LORA")
    print(f"{'='*80}")
    sys.stdout.flush()
    
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    print("Applying LoRA to model...")
    model = get_peft_model(model, lora_config)
    sys.stdout.flush()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"\n✅ LoRA Applied")
    print(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
    print(f"   Total params: {all_params:,}")
    sys.stdout.flush()
    
    return model

# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset(test_size=0.1):
    """Load pre-tokenized dataset"""
    
    print(f"\n{'='*80}")
    print("📊 LOADING TOKENIZED DATASET")
    print(f"{'='*80}")
    sys.stdout.flush()
    
    try:
        # Load pre-tokenized dataset
        tokenized_dataset = load_from_disk('../datasets/tokenized_coach')
        print(f"✅ Loaded pre-tokenized dataset")
    except:
        print("❌ Pre-tokenized dataset not found. Run fix_labels.py first.")
        sys.exit(1)
    
    # Split
    print(f"Splitting dataset (test_size={test_size})...")
    split_dataset = tokenized_dataset.train_test_split(test_size=test_size)
    sys.stdout.flush()
    
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"✅ Dataset prepared")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Eval samples: {len(eval_dataset)}")
    sys.stdout.flush()
    
    return train_dataset, eval_dataset

# ============================================================
# TRAINING WITH HUGGINGFACE TRAINER
# ============================================================

def train(
    dataset_path,
    output_dir,
    agent_name="agent"
):
    """Fine-tune model with LoRA using HuggingFace Trainer"""
    
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING {agent_name.upper()}")
    print(f"{'='*80}")
    sys.stdout.flush()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Prepare for LoRA
    model = prepare_model_for_lora(model)
    
    # Prepare dataset
    train_dataset, eval_dataset = prepare_dataset()
    
    # Training configuration
    print(f"\n{'='*80}")
    print("⚙️  TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Batch size: {config.train_batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max sequence length: {config.max_seq_length}")
    sys.stdout.flush()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=False,
        optim="adamw_torch",
        seed=42,
        fp16=False,
        bf16=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Training loop
    print(f"\n{'='*80}")
    print("🎓 STARTING TRAINING")
    print(f"{'='*80}\n")
    sys.stdout.flush()
    
    trainer.train()
    
    # Save final model
    print(f"\n{'='*80}")
    print("💾 SAVING MODEL")
    print(f"{'='*80}")
    sys.stdout.flush()
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"✅ Model saved to: {output_dir}")
    print(f"\n🎉 Training complete!")
    sys.stdout.flush()
    
    return model, tokenizer

# ============================================================
# INFERENCE
# ============================================================

def generate_response(model_dir, prompt, max_length=200, temperature=0.7):
    """Generate response from trained model"""
    
    print(f"\nLoading model from {model_dir}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    
    formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:"""
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("LoRA Training Pipeline Initialized!")
    print("\nUsage in your training script:")
    print("  from training_lora_trainer_final import train")
    print("  train(dataset_path='../datasets/inspiration_coach_data.json',")
    print("        output_dir='../models/inspiration_coach_lora',")
    print("        agent_name='inspiration_coach')")