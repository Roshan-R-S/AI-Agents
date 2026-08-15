"""
Test Inspiration Coach Training Quality (IMPROVED)
Run: python agents/test_inspiration_coach.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/inspiration_coach_lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Test prompts to evaluate quality
TEST_PROMPTS = [
    "I'm feeling overwhelmed with life decisions. How do I choose the right path?",
    "How do I stay motivated when facing failures?",
    "I'm scared of taking risks. How do I overcome fear?",
    "What should I do when I feel lost and purposeless?",
    "How can I build confidence in myself?",
]

# ============================================================
# TEST
# ============================================================

def test_model(model_path, test_prompts):
    """Test the trained model"""
    
    print("="*80)
    print("🧪 TESTING INSPIRATION COACH (IMPROVED)")
    print("="*80)
    
    # Check if model exists
    import os
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Make sure training is complete!")
        return False
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    model.eval()
    
    # Test each prompt
    print(f"\n{'-'*80}")
    print("Testing with sample prompts...")
    print(f"{'-'*80}\n")
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}/{len(test_prompts)}")
        print(f"Prompt: {prompt}\n")
        
        # Format prompt
        formatted_prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:"""
        
        try:
            # Tokenize
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
            
            # Generate with IMPROVED parameters
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=700,  # ✅ Increased from 500 - allow longer responses
                    min_length=150,  # ✅ New: ensure minimum length
                    temperature=0.75,  # ✅ Balanced creativity
                    top_p=0.90,  # ✅ Better quality sampling
                    top_k=50,  # ✅ New: restrict to top 50 tokens
                    do_sample=True,
                    num_beams=1,  # ✅ Single beam (fast, good for this model)
                    repetition_penalty=1.3,  # ✅ Reduce repetition STRONGLY
                    length_penalty=0.6,  # ✅ Don't penalize longer responses
                    no_repeat_ngram_size=3,  # ✅ Don't repeat 3-grams
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True,  # ✅ Stop when EOS reached
                )
            
            # Decode
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response
            if "### Response:" in response:
                response = response.split("### Response:")[-1].strip()
            
            print(f"Response:\n{response}\n")
            print(f"{'-'*80}\n")
            
        except Exception as e:
            print(f"❌ Error generating response: {e}\n")
            continue
    
    print("="*80)
    print("✅ TESTING COMPLETE")
    print("="*80)
    print("\nIf responses look good and follow the Inspiration Coach personality:")
    print("✅ Training was successful!")
    print("Use: python agents/inspiration_coach.py")
    print("\nIf responses still have repetition issues:")
    print("⚠️  Dataset may need cleaning (remove repetitive training examples)")
    print("\n")
    
    return True

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    test_model(MODEL_PATH, TEST_PROMPTS)