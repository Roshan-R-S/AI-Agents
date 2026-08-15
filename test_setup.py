"""
Test that your hardware setup is ready for fine-tuning
Run: python test_setup.py
"""

import torch
import sys

print("="*80)
print("🔍 CHECKING YOUR SETUP FOR AI AGENT TRAINING")
print("="*80)

# Check Python
print(f"\n✅ Python: {sys.version.split()[0]}")

# Check PyTorch
print(f"✅ PyTorch: {torch.__version__}")

# Check CUDA
if torch.cuda.is_available():
    print(f"✅ CUDA: Available")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
    print(f"   CUDA Version: {torch.version.cuda}")
    device_check = "✅ PASS"
else:
    print("❌ CUDA: NOT AVAILABLE")
    print("   This will cause training to be VERY slow")
    device_check = "❌ FAIL"

# Check transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✅ Transformers: Installed")
    transformers_check = "✅ PASS"
except Exception as e:
    print(f"❌ Transformers: {e}")
    transformers_check = "❌ FAIL"

# Check PEFT (LoRA)
try:
    from peft import LoraConfig, get_peft_model
    print("✅ PEFT (LoRA): Installed")
    peft_check = "✅ PASS"
except Exception as e:
    print(f"❌ PEFT: {e}")
    peft_check = "❌ FAIL"

# Check bitsandbytes
try:
    import bitsandbytes
    print("✅ Bitsandbytes: Installed")
    bitsandbytes_check = "✅ PASS"
except Exception as e:
    print(f"❌ Bitsandbytes: {e}")
    bitsandbytes_check = "❌ FAIL"

# Check datasets
try:
    from datasets import Dataset
    print("✅ Datasets: Installed")
    datasets_check = "✅ PASS"
except Exception as e:
    print(f"❌ Datasets: {e}")
    datasets_check = "❌ FAIL"

# Check huggingface_hub
try:
    from huggingface_hub import hf_hub_download
    print("✅ HuggingFace Hub: Installed")
    hub_check = "✅ PASS"
except Exception as e:
    print(f"❌ HuggingFace Hub: {e}")
    hub_check = "❌ FAIL"

print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)

all_checks = [device_check, transformers_check, peft_check, bitsandbytes_check, datasets_check, hub_check]

if all("✅" in check for check in all_checks):
    print("✅ ALL SYSTEMS GO! You're ready to train AI agents! 🚀")
else:
    print("⚠️  Some dependencies are missing. Install them:")
    print("   pip install -r requirements_training.txt")

print("\n📚 TRAINING ESTIMATES (per agent):")
print("   1000 examples: ~4-5 hours on RTX 4050")
print("   500 examples: ~2-3 hours")
print("   100 examples: ~30 minutes")

print("\n🎯 Total time for 7 agents: ~30-35 hours (spread over 8 weeks)")
print("\n" + "="*80)
