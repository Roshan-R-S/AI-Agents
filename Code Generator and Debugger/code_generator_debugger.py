from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

llm = LLM(model="ollama/mistral", base_url="http://localhost:11434")

# ============================================================
# AGENT 1: Architect (Design Solution)
# ============================================================
architect = Agent(
    role="Software Architect",
    goal="Design optimal solutions for coding problems",
    backstory="You are an experienced architect who designs clean, scalable solutions. You think about edge cases and best practices.",
    llm=llm
)

# ============================================================
# AGENT 2: Code Generator (Write Code)
# ============================================================
code_generator = Agent(
    role="Code Generator",
    goal="Write clean, production-ready code",
    backstory="You are an expert programmer who writes efficient, readable code following best practices and design patterns.",
    llm=llm
)

# ============================================================
# AGENT 3: Code Reviewer (Review Quality)
# ============================================================
code_reviewer = Agent(
    role="Code Reviewer",
    goal="Review code for quality, security, and best practices",
    backstory="You are a meticulous code reviewer. You catch bugs, security issues, and suggest improvements.",
    llm=llm
)

# ============================================================
# AGENT 4: Debugger (Find & Fix Bugs)
# ============================================================
debugger = Agent(
    role="Debugger",
    goal="Find and fix bugs in code",
    backstory="You are a master debugger. You identify issues, root causes, and create fixes.",
    llm=llm
)

# ============================================================
# AGENT 5: Optimizer (Improve Performance)
# ============================================================
optimizer = Agent(
    role="Performance Optimizer",
    goal="Optimize code for speed and efficiency",
    backstory="You specialize in performance. You identify bottlenecks and optimize code.",
    llm=llm
)

# ============================================================
# TASK 1: Architect Design
# ============================================================
task_design = Task(
    description="Design a solution for: Create a Python function that finds the longest word in a sentence, handles edge cases, and returns word length too.",
    expected_output="Solution design with approach, data structures, and algorithm explanation",
    agent=architect
)

# ============================================================
# TASK 2: Code Generation
# ============================================================
task_generate = Task(
    description="Write Python code for: A function that finds the longest word in a sentence. Include docstring, type hints, and error handling.",
    expected_output="Complete, production-ready Python code with docstrings and type hints",
    agent=code_generator
)

# ============================================================
# TASK 3: Code Review
# ============================================================
task_review = Task(
    description="Review this code for quality: 1) Is it readable? 2) Does it follow best practices? 3) Are there security issues? 4) Is error handling adequate?",
    expected_output="Code review with suggestions for improvement",
    agent=code_reviewer
)

# ============================================================
# TASK 4: Debug Check
# ============================================================
task_debug = Task(
    description="Test this code with edge cases: empty string, single word, multiple spaces, special characters. Find potential bugs and suggest fixes.",
    expected_output="Bug report with test cases and fixes",
    agent=debugger
)

# ============================================================
# TASK 5: Optimization
# ============================================================
task_optimize = Task(
    description="Optimize the code for performance. Check: time complexity, space complexity, unnecessary loops, and suggest optimizations.",
    expected_output="Optimized code with performance explanation",
    agent=optimizer
)

# ============================================================
# CREW - Code Generation Pipeline
# ============================================================
crew = Crew(
    agents=[architect, code_generator, code_reviewer, debugger, optimizer],
    tasks=[task_design, task_generate, task_review, task_debug, task_optimize]
)

# ============================================================
# RUN SYSTEM
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("CODE GENERATOR & DEBUGGER SYSTEM")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    print("🏗️  Architect designing solution...")
    print("💻 Code generator writing code...")
    print("👀 Reviewer checking quality...")
    print("🐛 Debugger testing for bugs...")
    print("⚡ Optimizer improving performance...")
    print("\n" + "="*80 + "\n")
    
    result = crew.kickoff()
    
    # Save to file
    filename = f"code_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w") as f:
        f.write("="*80 + "\n")
        f.write("CODE GENERATION & DEBUGGING OUTPUT\n")
        f.write("="*80 + "\n\n")
        f.write(str(result))
        f.write("\n\n" + "="*80)
    
    print("="*80)
    print(result)
    print("\n" + "="*80)
    print(f"✅ Output saved to: {filename}")
    print("="*80 + "\n")