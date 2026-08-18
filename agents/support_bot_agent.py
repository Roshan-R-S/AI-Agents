"""
Support Bot Agent - OpenRouter + NVIDIA Nemotron 3 Ultra
Customer service, troubleshooting, and support escalation
"""

import requests
import json
import os

# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# System prompt for customer support
SYSTEM_PROMPT = """You are a professional Support Bot - an empathetic customer service representative who solves problems and helps customers successfully.

Your approach:
1. **Empathize First** - Acknowledge the customer's frustration or problem genuinely
2. **Understand Deeply** - Ask clarifying questions to fully understand the issue
3. **Take Ownership** - Show that you're here to help and will work until it's resolved
4. **Troubleshoot Systematically** - Provide clear, step-by-step solutions
5. **Know Your Limits** - Recognize when to escalate to human support
6. **Be Professional** - Maintain professionalism while being warm and approachable
7. **Follow Up** - Offer continued support and document the resolution

When helping customers, address:
- **Problem Understanding** - What exactly is the issue?
- **Root Cause** - Why is this happening?
- **Immediate Solutions** - Quick fixes they can try now
- **Permanent Solutions** - How to prevent this in the future
- **Escalation Path** - When and how to escalate
- **Documentation** - Offering to save solutions for reference

Do:
✅ Be patient and kind
✅ Use simple, clear language
✅ Provide step-by-step instructions
✅ Explain technical concepts simply
✅ Validate customer emotions
✅ Offer multiple solutions when possible
✅ Follow up on resolution
✅ Know when to escalate

Don't:
❌ Blame the customer
❌ Use jargon without explanation
❌ Rush the customer
❌ Make promises you can't keep
❌ Dismiss their concerns
❌ Be defensive

You represent the company with professionalism and care."""

def load_api_key():
    """Load OpenRouter API key"""
    if not OPENROUTER_API_KEY:
        print("❌ ERROR: OPENROUTER_API_KEY environment variable not set!")
        print("\nTo set it in PowerShell:")
        print('  $env:OPENROUTER_API_KEY = "your-api-key-here"')
        return None
    return OPENROUTER_API_KEY

def generate_response(messages):
    """Call OpenRouter API for support response"""
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.6,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 1500,
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code != 200:
            return f"API Error: {response.status_code} - {response.text}"
        
        data = response.json()
        return data["choices"][0]["message"]["content"]
        
    except Exception as e:
        return f"Error calling API: {str(e)}"

def main():
    """Main conversation loop"""
    
    api_key = load_api_key()
    if not api_key:
        return
    
    print("=" * 80)
    print("🎧 SUPPORT BOT - Customer Service & Support")
    print("=" * 80)
    print("Powered by NVIDIA Nemotron 3 Ultra")
    print("Welcome! How can we help you today?")
    print("Type 'exit' or 'quit' to end the chat")
    print("=" * 80 + "\n")
    
    # Conversation history
    conversation = []
    
    print("Support: Hello! Thank you for contacting us. How can I help you today?\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nSupport: Thank you for contacting us. We hope your issue is resolved!")
                print("Have a great day! 😊")
                break
            
            # Add user message to conversation
            conversation.append({
                "role": "user",
                "content": user_input
            })
            
            # Prepare messages with system prompt
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT}
            ] + conversation
            
            print("\nSupport: ", end="", flush=True)
            
            # Get response from API
            response = generate_response(messages)
            print(f"{response}\n")
            
            # Add support response to conversation history
            conversation.append({
                "role": "assistant",
                "content": response
            })
            
        except KeyboardInterrupt:
            print("\n\nSupport: Thank you for contacting us. We hope your issue is resolved!")
            print("Have a great day! 😊")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()