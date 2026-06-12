from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv

load_dotenv()

llm = LLM(model="ollama/mistral", base_url="http://localhost:11434")

# Agent WITHOUT memory (no OpenAI needed)
support_agent = Agent(
    role="Customer Support Specialist",
    goal="Help customers with issues",
    backstory="You are a helpful support agent",
    llm=llm
)

# Task 1: First conversation
task1 = Task(
    description="Customer says: 'Hi, I have a problem with my order #12345. It hasn't arrived yet.'",
    expected_output="Helpful response addressing the issue",
    agent=support_agent
)

# Task 2: Follow-up
task2 = Task(
    description="Customer says: 'Can you check the status of my order?' (They mentioned order #12345 before)",
    expected_output="Response addressing the follow-up",
    agent=support_agent
)

crew = Crew(
    agents=[support_agent],
    tasks=[task1, task2]
)

if __name__ == "__main__":
    print("\n" + "="*80)
    print("CUSTOMER SUPPORT CHATBOT")
    print("="*80 + "\n")
    
    result = crew.kickoff()
    
    print("\n" + "="*80)
    print(result)
    print("="*80 + "\n")