from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

llm = LLM(model="ollama/neural-chat", base_url="http://localhost:11434")

support_agent = Agent(
    role="Customer Support Specialist",
    goal="Help customers with questions and issues",
    backstory="You are friendly, helpful, and patient. You solve problems quickly.",
    llm=llm
)

# Different customer questions to handle
questions = [
    "How do I return a product?",
    "What's your refund policy?",
    "Can I change my order after placing it?"
]

# Run for each question
for question in questions:
    print(f"\n{'='*80}")
    print(f"CUSTOMER: {question}")
    print('='*80)
    
    task = Task(
        description=f"A customer asks: {question}. Provide a helpful, friendly answer.",
        expected_output="Clear, helpful response",
        agent=support_agent
    )
    
    crew = Crew(agents=[support_agent], tasks=[task])
    result = crew.kickoff()
    
    # Save each response
    filename = f"support_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w") as f:
        f.write(f"Question: {question}\n\nAnswer:\n{result}")
    
    print(f"\n{result}")
    print(f"✅ Saved to: {filename}")