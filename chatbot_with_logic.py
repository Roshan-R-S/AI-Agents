from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv

load_dotenv()

llm = LLM(model="ollama/mistral", base_url="http://localhost:11434")

support_agent = Agent(
    role="Intelligent Support Agent",
    goal="Handle customer issues with smart routing and priority assessment",
    backstory="You analyze issues, determine priority levels, and route to appropriate teams",
    llm=llm
)

task1 = Task(
    description="Customer says: 'My order is CRITICAL and hasn't shipped yet!'. Assess priority (HIGH/MEDIUM/LOW) and respond appropriately.",
    expected_output="Priority assessment and escalation decision",
    agent=support_agent
)

task2 = Task(
    description="Customer has a BILLING DISPUTE. Determine if this needs escalation to billing team. Provide solution steps.",
    expected_output="Assessment and routing to appropriate team",
    agent=support_agent
)

task3 = Task(
    description="Customer wants to RETURN an item. Provide step-by-step return instructions.",
    expected_output="Clear return process instructions",
    agent=support_agent
)

crew = Crew(agents=[support_agent], tasks=[task1, task2, task3])

if __name__ == "__main__":
    print("\n" + "="*80)
    print("INTELLIGENT SUPPORT CHATBOT WITH CONDITIONAL LOGIC")
    print("="*80 + "\n")
    
    result = crew.kickoff()
    
    print("\n" + "="*80)
    print(result)
    print("="*80 + "\n")