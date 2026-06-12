from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv

load_dotenv()

llm = LLM(model="ollama/neural-chat", base_url="http://localhost:11434")

# Agent 1: Researcher
researcher = Agent(
    role="Research Expert",
    goal="Find accurate information",
    backstory="You are thorough researcher",
    llm=llm
)

# Agent 2: Analyst
analyst = Agent(
    role="Data Analyst",
    goal="Analyze and provide insights",
    backstory="You find patterns and draw conclusions",
    llm=llm
)

# Tasks
research_task = Task(
    description="Research the benefits of AI in healthcare",
    expected_output="3 key benefits of AI in healthcare",
    agent=researcher
)

analysis_task = Task(
    description="Analyze the research findings and explain their impact",
    expected_output="Analysis of impact on healthcare industry",
    agent=analyst
)

# Crew
crew = Crew(
    agents=[researcher, analyst],
    tasks=[research_task, analysis_task]
)

if __name__ == "__main__":
    result = crew.kickoff()
    print(result)