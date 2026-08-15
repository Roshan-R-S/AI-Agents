"""
Train Inspiration Coach Agent
Based on Claude Opus personality
Run: python training/train_inspiration_coach.py
"""

import os
import sys
from pathlib import Path

# Add training module to path
sys.path.insert(0, str(Path(__file__).parent))

from training_lora_trainer import train, generate_response

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "../datasets/inspiration_coach_data.json"
OUTPUT_DIR = "../models/inspiration_coach_lora"
AGENT_NAME = "inspiration_coach"

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("🎯 INSPIRATION COACH TRAINING")
    print("="*80)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Agent: {AGENT_NAME}")
    print("="*80 + "\n")
    
    # Check dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        print(f"Create it first!")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Train
    try:
        print("Starting training...")
        print("This will take 4-5 hours on RTX 4050")
        print("Please don't close this terminal!\n")
        
        model, tokenizer = train(
            dataset_path=DATASET_PATH,
            output_dir=OUTPUT_DIR,
            agent_name=AGENT_NAME
        )
        
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETE!")
        print("="*80)
        print(f"Model saved to: {OUTPUT_DIR}")
        print("\nNext steps:")
        print("1. Test the model: python agents/test_inspiration_coach.py")
        print("2. Use the agent: python agents/inspiration_coach.py")
        print("\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("You can resume from the last checkpoint")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
