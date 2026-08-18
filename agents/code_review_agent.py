"""
Code Review Agent - OpenRouter + Cohere North Mini Code
Technical code analysis, feedback, and improvement suggestions
"""

import requests
import json
import os

# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"  # Nemotron 3 Ultra on OpenRouter
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# System prompt for code review
SYSTEM_PROMPT = """You are an expert Code Review Agent - a senior software engineer who reviews code with precision and helps developers improve their craft.

Your approach:
1. **Analyze the code thoroughly** - Look for bugs, performance issues, security vulnerabilities, readability problems
2. **Identify patterns** - Recognize anti-patterns, code smells, and architectural issues
3. **Provide specific feedback** - Don't just say "this is bad," explain why and show better approaches
4. **Suggest improvements** - Offer concrete refactoring suggestions with examples
5. **Explain best practices** - Share why certain approaches are better (SOLID principles, DRY, KISS, etc.)
6. **Be constructive** - Criticize the code, not the developer. Be encouraging and educational
7. **Prioritize issues** - Critical bugs first, then performance, then style

When reviewing code, address:
- **Correctness** - Does it work? Will it fail in edge cases?
- **Performance** - Is it efficient? Any bottlenecks?
- **Security** - Any vulnerabilities or unsafe practices?
- **Readability** - Is it clear? Will others understand it?
- **Maintainability** - Will it be easy to modify later?
- **Best Practices** - Does it follow language/framework conventions?

Format your reviews clearly with sections and specific line references when possible."""

def load_api_key():
    """Load OpenRouter API key"""
    if not OPENROUTER_API_KEY:
        print("❌ ERROR: OPENROUTER_API_KEY environment variable not set!")
        print("\nTo set it in PowerShell:")
        print('  $env:OPENROUTER_API_KEY = "your-api-key-here"')
        return None
    return OPENROUTER_API_KEY

def generate_response(messages):
    """Call OpenRouter API for code review"""
    
    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "model": MODEL,
            "messages": messages,
            "temperature": 0.5,  # Lower temperature for technical accuracy
            "top_p": 0.9,
            "max_tokens": 2000,  # More tokens for detailed reviews
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
    print("💻 CODE REVIEW AGENT - Expert Code Analysis")
    print("=" * 80)
    print("Powered by Cohere North Mini Code")
    print("Paste your code and get professional technical feedback!")
    print("Type 'exit' or 'quit' to end session")
    print("=" * 80 + "\n")
    
    # Conversation history
    conversation = []
    
    print("Tip: You can paste:")
    print("  • Full code snippets")
    print("  • Functions or methods")
    print("  • File paths with code context")
    print("  • Questions about your code")
    print("=" * 80 + "\n")
    
    while True:
        try:
            print("You (paste code or question):")
            user_input = input().strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nReviewer: Thanks for sharing your code. Keep improving! 🚀")
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
            
            print("\nReviewer: ", end="", flush=True)
            
            # Get response from API
            response = generate_response(messages)
            print(f"{response}\n")
            
            # Add reviewer response to conversation history
            conversation.append({
                "role": "assistant",
                "content": response
            })
            
            # Continue prompt
            print("-" * 80)
            print("Follow-up (ask about specific lines, request refactoring, etc.):")
            
        except KeyboardInterrupt:
            print("\n\nReviewer: Thanks for sharing your code. Keep improving! 🚀")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()