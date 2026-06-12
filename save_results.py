from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

llm = LLM(model="ollama/neural-chat", base_url="http://localhost:11434")

researcher = Agent(
    role="Research Expert",
    goal="Find accurate information",
    backstory="You are a thorough researcher",
    llm=llm
)

analyzer = Agent(
    role="Data Analyst",
    goal="Analyze and provide insights",
    backstory="You find patterns and conclusions",
    llm=llm
)

research_task = Task(
    description="Research the top 5 AI applications in 2024",
    expected_output="5 AI applications with brief descriptions",
    agent=researcher
)

analysis_task = Task(
    description="Analyze these AI applications and their impact",
    expected_output="Analysis of impact on industries",
    agent=analyzer
)

crew = Crew(agents=[researcher, analyzer], tasks=[research_task, analysis_task])

if __name__ == "__main__":
    result = crew.kickoff()
    
    # Save to file with timestamp
    filename = f"ai_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w") as f:
        f.write(str(result))
    
    print(f"\n✅ Results saved to: {filename}")
    print("\n" + "="*80)
    print(result)