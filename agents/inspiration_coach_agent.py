"""
Inspiration Coach - Interactive Coaching Agent
Simplified version optimized for TinyLlama
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "models/inspiration_coach_lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Simplified system prompt - less instruction overhead
SYSTEM_PROMPT = """You are an Inspiration Coach. You help people with life challenges, relationships, careers, and personal growth.

Be warm, wise, and direct. Ask clarifying questions. Give actionable advice. Keep responses concise and focused."""

def load_model():
    """Load the trained Inspiration Coach model"""
    print("Loading Inspiration Coach from models/inspiration_coach_lora...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model.eval()
        print("✅ Inspiration Coach loaded!")
        print(f"   Device: {DEVICE}")
        print("   Ready to inspire and guide\n")
        return model, tokenizer
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def generate_response(model, tokenizer, user_input):
    """Generate coaching response"""
    
    # Simpler prompt format
    formatted_prompt = f"""{SYSTEM_PROMPT}

User: {user_input}

Coach:"""
    
    try:
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=600,
                min_length=80,
                temperature=0.7,  # Lower temperature = more focused
                top_p=0.80,
                top_k=35,
                repetition_penalty=1.3,  # Prevent repetition
                no_repeat_ngram_size=3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the coach response
        if "Coach:" in response:
            response = response.split("Coach:")[-1].strip()
        
        # Clean up any remaining artifacts
        response = response.strip()
        if not response:
            response = "I appreciate you sharing that. Can you tell me more about what's on your mind?"
        
        return response
        
    except Exception as e:
        return f"I encountered an error. Could you rephrase that?"

def main():
    """Main conversation loop"""
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        print("Failed to load model. Exiting.")
        return
    
    print("=" * 80)
    print("💫 INSPIRATION COACH - Interactive Session")
    print("=" * 80)
    print("Talk with your personal Inspiration Coach!")
    print("Type 'exit' or 'quit' to end conversation")
    print("=" * 80 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nCoach: Thank you for sharing. Remember, growth is a journey. Take care!")
                break
            
            print("Coach: ", end="", flush=True)
            response = generate_response(model, tokenizer, user_input)
            print(f"{response}\n")
            
        except KeyboardInterrupt:
            print("\n\nCoach: Thank you for sharing. Remember, growth is a journey. Take care!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()