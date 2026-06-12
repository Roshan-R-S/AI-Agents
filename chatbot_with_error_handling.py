from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv

load_dotenv()

llm = LLM(model="ollama/mistral", base_url="http://localhost:11434")

support_agent = Agent(
    role="Support Agent",
    goal="Help customers handle errors gracefully",
    backstory="You handle errors gracefully and guide customers to solutions",
    llm=llm
)

task1 = Task(
    description="Try to check order status for order '99999' (doesn't exist in system). Handle the error gracefully and suggest valid order numbers.",
    expected_output="Helpful error message with guidance",
    agent=support_agent
)

task2 = Task(
    description="Validate this customer email: 'john@example.com'. Check if it's valid format.",
    expected_output="Email validation result",
    agent=support_agent
)

task3 = Task(
    description="A payment operation failed temporarily. Retry strategy needed.",
    expected_output="Retry instructions and success message",
    agent=support_agent
)

crew = Crew(agents=[support_agent], tasks=[task1, task2, task3])

if __name__ == "__main__":
    print("\n" + "="*80)
    print("SUPPORT CHATBOT WITH ERROR HANDLING")
    print("="*80 + "\n")
    
    result = crew.kickoff()
    
    print("\n" + "="*80)
    print(result)
    print("="*80 + "\n")