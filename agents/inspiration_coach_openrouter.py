"""
Inspiration Coach - OpenRouter + NVIDIA Nemotron 3 Ultra
Multi-turn coaching agent with professional quality responses
"""

import requests
import json
import os

# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")  # Set environment variable
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"  # Nemotron 3 Ultra on OpenRouter
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# System prompt for coaching
SYSTEM_PROMPT = """You are an Inspiration Coach - a compassionate, insightful life coach who helps people navigate challenges and find direction.

Your approach:
1. Always ask clarifying questions to understand the situation
2. Provide root cause analysis ("Here's what I'm noticing...")
3. Give clear, actionable steps
4. Be warm, genuine, and empathetic
5. Keep responses focused and concise (2-3 paragraphs max)
6. Help people empower themselves to solve their own problems

You are wise, caring, and genuinely interested in helping."""

def load_api_key():
    """Load OpenRouter API key"""
    if not OPENROUTER_API_KEY:
        print("❌ ERROR: OPENROUTER_API_KEY environment variable not set!")
        print("\nTo set it in PowerShell:")
        print('  $env:OPENROUTER_API_KEY = "your-api-key-here"')
        print("\nOr pass it directly in the script.")
        return None
    return OPENROUTER_API_KEY

def generate_response(messages):
    """Call OpenRouter API for coaching response"""
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_tokens": 500,
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
    print("💫 INSPIRATION COACH - Interactive Session")
    print("=" * 80)
    print("Powered by NVIDIA Nemotron 3 Ultra")
    print("Talk with your personal Inspiration Coach!")
    print("Type 'exit' or 'quit' to end conversation")
    print("=" * 80 + "\n")
    
    # Conversation history
    conversation = []
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nCoach: Thank you for sharing. Remember, growth is a journey. Take care!")
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
            
            print("Coach: ", end="", flush=True)
            
            # Get response from API
            response = generate_response(messages)
            print(f"{response}\n")
            
            # Add coach response to conversation history
            conversation.append({
                "role": "assistant",
                "content": response
            })
            
        except KeyboardInterrupt:
            print("\n\nCoach: Thank you for sharing. Remember, growth is a journey. Take care!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()