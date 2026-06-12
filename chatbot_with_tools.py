from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv

load_dotenv()

llm = LLM(model="ollama/mistral", base_url="http://localhost:11434")

# ============================================================
# SUPPORT AGENT (No tools - just pure reasoning)
# ============================================================
support_agent = Agent(
    role="Customer Support Specialist",
    goal="Help customers with orders and issues",
    backstory="You are an expert support agent with access to order database and refund system",
    llm=llm
)

# ============================================================
# TASKS
# ============================================================
task1 = Task(
    description="Customer 'John' asks: 'Where is my order 12345?'. Check order status from database and respond.",
    expected_output="Response with order status (Order 12345 is in transit, arriving tomorrow)",
    agent=support_agent
)

task2 = Task(
    description="Customer 'Sarah' is upset and says: 'I want a refund for order 12346. It arrived damaged.' Process the refund and be empathetic.",
    expected_output="Empathetic response confirming refund for $45.50",
    agent=support_agent
)

task3 = Task(
    description="Check the purchase history of customer 'Mike': Order 12347 ($55). Summarize his orders.",
    expected_output="Summary of Mike's purchase history",
    agent=support_agent
)

crew = Crew(agents=[support_agent], tasks=[task1, task2, task3])

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SUPPORT CHATBOT WITH REASONING")
    print("="*80 + "\n")
    
    result = crew.kickoff()
    
    print("\n" + "="*80)
    print(result)
    print("="*80 + "\n")