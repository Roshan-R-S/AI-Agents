from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv

load_dotenv()

# Use Ollama locally (FREE - no API key needed!)
llm = LLM(
    model="ollama/llama2",
    base_url="http://localhost:11434"
)

# Agent using local Ollama
researcher = Agent(
    role="Research Expert",
    goal="Find and explain information",
    backstory="You are a helpful researcher",
    llm=llm
)

# Task
task = Task(
    description="What are the top 3 AI trends in 2024?",
    expected_output="A list of 3 AI trends with brief explanations",
    agent=researcher
)

# Crew
crew = Crew(
    agents=[researcher],
    tasks=[task]
)

# Run
if __name__ == "__main__":
    result = crew.kickoff()
    print(result)