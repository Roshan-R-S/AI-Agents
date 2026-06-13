from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

llm = LLM(model="ollama/mistral", base_url="http://localhost:11434")

architect = Agent(
    role="Software Architect",
    goal="Design optimal solutions",
    backstory="Expert architect designing clean solutions",
    llm=llm
)

code_generator = Agent(
    role="Code Generator",
    goal="Write clean, production-ready code",
    backstory="Expert programmer writing efficient code",
    llm=llm
)

code_reviewer = Agent(
    role="Code Reviewer",
    goal="Review code quality",
    backstory="Meticulous reviewer catching bugs",
    llm=llm
)

debugger = Agent(
    role="Debugger",
    goal="Find and fix bugs",
    backstory="Master debugger identifying issues",
    llm=llm
)

optimizer = Agent(
    role="Performance Optimizer",
    goal="Optimize for speed",
    backstory="Performance specialist",
    llm=llm
)

def generate_code(requirement: str):
    """Generate code based on user requirement"""
    
    task_design = Task(
        description=f"Design a solution for: {requirement}",
        expected_output="Solution design with approach",
        agent=architect
    )

    task_generate = Task(
        description=f"Write Python code for: {requirement}. Include docstrings, type hints, and error handling.",
        expected_output="Complete, production-ready code",
        agent=code_generator
    )

    task_review = Task(
        description="Review the generated code for quality, best practices, and security issues.",
        expected_output="Code review with suggestions",
        agent=code_reviewer
    )

    task_debug = Task(
        description="Test code with edge cases and find potential bugs.",
        expected_output="Bug report with fixes",
        agent=debugger
    )

    task_optimize = Task(
        description="Optimize code for performance.",
        expected_output="Optimized code with explanation",
        agent=optimizer
    )

    crew = Crew(
        agents=[architect, code_generator, code_reviewer, debugger, optimizer],
        tasks=[task_design, task_generate, task_review, task_debug, task_optimize]
    )
    
    return crew.kickoff()

if __name__ == "__main__":
    # Example requirements
    requirements = [
        "Create a function that calculates factorial with memoization",
        "Build a function that reverses a string without using built-in reverse",
        "Create a function that finds all prime numbers up to N",
    ]
    
    for i, req in enumerate(requirements, 1):
        print("\n" + "="*80)
        print(f"REQUIREMENT {i}: {req}")
        print("="*80 + "\n")
        
        result = generate_code(req)
        
        # Save
        filename = f"code_gen_{i}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w") as f:
            f.write(f"Requirement: {req}\n\n")
            f.write(str(result))
        
        print(result)
        print(f"\n✅ Saved to: {filename}\n")