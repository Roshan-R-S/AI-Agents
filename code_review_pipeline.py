from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv

load_dotenv()

llm = LLM(model="ollama/neural-chat", base_url="http://localhost:11434")

# Code reviewer
reviewer = Agent(
    role="Senior Code Reviewer",
    goal="Review code and suggest improvements",
    backstory="You are an expert programmer",
    llm=llm
)

# Security expert
security = Agent(
    role="Security Expert",
    goal="Identify security issues",
    backstory="You specialize in secure coding",
    llm=llm
)

# Documentation specialist
docs = Agent(
    role="Documentation Specialist",
    goal="Improve code documentation",
    backstory="You ensure code is well documented",
    llm=llm
)

# Sample Python code to review
code_sample = """
def process_user_data(user_input):
    data = eval(user_input)
    result = data * 2
    return result
"""

review_task = Task(
    description=f"Review this Python code for quality: {code_sample}. Suggest improvements.",
    expected_output="Code review with suggestions",
    agent=reviewer
)

security_task = Task(
    description=f"Check this code for security issues: {code_sample}. Identify vulnerabilities.",
    expected_output="Security analysis",
    agent=security
)

docs_task = Task(
    description="Suggest documentation improvements for the reviewed code",
    expected_output="Documentation suggestions",
    agent=docs
)

crew = Crew(agents=[reviewer, security, docs], tasks=[review_task, security_task, docs_task])

if __name__ == "__main__":
    result = crew.kickoff()
    print(result)