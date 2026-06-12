from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv

load_dotenv()

llm = LLM(model="ollama/neural-chat", base_url="http://localhost:11434")

researcher = Agent(
    role="Research Expert",
    goal="Research information thoroughly",
    backstory="You are a detailed researcher",
    llm=llm
)

writer = Agent(
    role="Content Writer",
    goal="Write clear, well-organized content",
    backstory="You write excellent summaries",
    llm=llm
)

research_task = Task(
    description="Research top 3 machine learning trends in 2024",
    expected_output="3 detailed ML trends",
    agent=researcher
)

write_task = Task(
    description="Based on the research, write a professional summary",
    expected_output="A well-written summary",
    agent=writer
)

crew = Crew(agents=[researcher, writer], tasks=[research_task, write_task])

if __name__ == "__main__":
    print("\n" + "="*80)
    print("RUNNING AGENT CREW WITH FILE SAVING")
    print("="*80 + "\n")
    
    result = crew.kickoff()
    
    # Save result to file (Python built-in)
    filename = "ml_trends_summary.txt"
    with open(filename, "w") as f:
        f.write(str(result))
    
    print("\n" + "="*80)
    print(f"✅ Results saved to: {filename}")
    print("="*80)
    print(result)