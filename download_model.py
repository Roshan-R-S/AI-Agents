"""
Download Mistral 7B Base Model
Size: ~9GB total
Time: 20-30 minutes depending on internet

This downloads both:
1. GGUF version (for testing/inference)
2. HuggingFace version (for fine-tuning)
"""

import os
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForCausalLM

print("="*80)
print("📥 DOWNLOADING MISTRAL 7B BASE MODEL")
print("="*80)

# Create folders
os.makedirs("models/base", exist_ok=True)
os.makedirs("models/mistral-7b-hf", exist_ok=True)

print(f"\nDevice: {torch.cuda.get_device_name(0)}")
print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")

# ============================================================
# GGUF VERSION (for testing)
# ============================================================
print("\n" + "="*80)
print("1️⃣  DOWNLOADING GGUF VERSION (for inference testing)")
print("="*80)
print("   Size: ~4.5GB")
print("   Time: 10-15 minutes")

try:
    gguf_path = hf_hub_download(
        repo_id="TheBloke/Mistral-7B-v0.1-GGUF",
        filename="mistral-7b-v0.1.Q4_K_M.gguf",
        cache_dir="models/base",
        resume_download=True
    )
    print(f"\n✅ GGUF downloaded: {gguf_path}")
except Exception as e:
    print(f"\n⚠️  GGUF download skipped (optional): {e}")

# ============================================================
# HUGGINGFACE VERSION (for fine-tuning)
# ============================================================
print("\n" + "="*80)
print("2️⃣  DOWNLOADING HUGGINGFACE VERSION (for fine-tuning)")
print("="*80)
print("   Size: ~4.5GB")
print("   Time: 10-15 minutes")

try:
    print("\n   📝 Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained("models/mistral-7b-hf")
    print("   ✅ Tokenizer saved to models/mistral-7b-hf/")

    print("\n   🤖 Downloading model...")
    print("   This is large (~4.5GB), please wait...")
    
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=True
    )
    model.save_pretrained("models/mistral-7b-hf")
    print("   ✅ Model saved to models/mistral-7b-hf/")
    
except Exception as e:
    print(f"\n❌ Error downloading HuggingFace model: {e}")
    print("   Make sure you have ~9GB free disk space")
    print("   Make sure internet connection is stable")
    exit(1)

# ============================================================
# VERIFY
# ============================================================
print("\n" + "="*80)
print("✅ MODELS DOWNLOADED SUCCESSFULLY!")
print("="*80)

# Verify files exist
import os
hf_files = os.listdir("models/mistral-7b-hf") if os.path.exists("models/mistral-7b-hf") else []
print(f"\nFiles in models/mistral-7b-hf/: {len(hf_files)} files")
for f in hf_files[:5]:  # Show first 5
    print(f"   - {f}")

print("\n🎯 NEXT STEPS:")
print("   1. Create inspiration_coach_data.json (training data)")
print("   2. Run training script: python training/train_inspiration_coach.py")
print("   3. Test the trained model")
print("\nReady to build your AI agent! 🚀")
