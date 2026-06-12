from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

llm = LLM(model="ollama/neural-chat", base_url="http://localhost:11434")

# Agent 1: Data Analyst
analyst = Agent(
    role="Data Analyst",
    goal="Analyze business data and identify trends",
    backstory="You are expert at analyzing data and finding insights",
    llm=llm
)

# Agent 2: Report Writer
reporter = Agent(
    role="Report Writer",
    goal="Write clear business reports",
    backstory="You turn data insights into compelling reports",
    llm=llm
)

# Agent 3: Executive Summarizer
executive = Agent(
    role="Executive Summary Writer",
    goal="Summarize reports for executives",
    backstory="You extract key findings for decision makers",
    llm=llm
)

# Task 1: Analyze
analyze_task = Task(
    description="Analyze sales data: Q1 sales were $500K, Q2 were $650K, Q3 were $800K, Q4 were $950K. Identify trends and patterns.",
    expected_output="Data analysis with trends identified",
    agent=analyst
)

# Task 2: Write Report
report_task = Task(
    description="Write a detailed business report based on the sales analysis. Include growth rate, trends, and recommendations.",
    expected_output="Complete business report",
    agent=reporter
)

# Task 3: Executive Summary
summary_task = Task(
    description="Create a 1-page executive summary with key findings and top 3 recommendations.",
    expected_output="Executive summary (1 page max)",
    agent=executive
)

crew = Crew(agents=[analyst, reporter, executive], tasks=[analyze_task, report_task, summary_task])

if __name__ == "__main__":
    print("\n" + "="*80)
    print("DATA ANALYSIS PIPELINE")
    print("="*80 + "\n")
    
    result = crew.kickoff()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"report_{timestamp}.txt"
    
    with open(filename, "w") as f:
        f.write(str(result))
    
    print("="*80)
    print(result)
    print("\n✅ Report saved to:", filename)
    print("="*80 + "\n")