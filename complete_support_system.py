from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

llm = LLM(model="ollama/mistral", base_url="http://localhost:11434")

# ============================================================
# AGENT 1: First Line Support
# ============================================================
first_line = Agent(
    role="First Line Support Agent",
    goal="Verify customers and handle initial inquiries",
    backstory="You are the first contact. You verify customers from database and check order status.",
    llm=llm
)

# ============================================================
# AGENT 2: Resolution Specialist
# ============================================================
resolver = Agent(
    role="Resolution Specialist",
    goal="Resolve customer issues and process requests",
    backstory="You handle refunds, replacements, and complex issues. You're empathetic and solution-focused.",
    llm=llm
)

# ============================================================
# AGENT 3: Escalation Manager
# ============================================================
manager = Agent(
    role="Escalation Manager",
    goal="Handle VIP and complex escalations",
    backstory="You manage critical issues for VIP customers. You ensure satisfaction.",
    llm=llm
)

# ============================================================
# WORKFLOW TASKS
# ============================================================
task1 = Task(
    description="Customer sarah@email.com asks: 'Where is my order 12347?'. She's a VIP customer. Verify her account and check order 12347 status (it's in Processing, amount $150).",
    expected_output="Friendly greeting, customer verification, and order status",
    agent=first_line
)

task2 = Task(
    description="Sarah (VIP customer) is unhappy about the delay. Process a $150 refund for order 12347 due to delay. Send notification confirming refund.",
    expected_output="Empathetic response and refund confirmation (5-7 business days processing)",
    agent=resolver
)

task3 = Task(
    description="Sarah is a VIP customer who had a shipping delay issue. Escalate to manager for follow-up call to ensure satisfaction.",
    expected_output="Escalation confirmation with manager follow-up plan",
    agent=manager
)

# ============================================================
# CREW - COMPLETE SUPPORT SYSTEM
# ============================================================
crew = Crew(
    agents=[first_line, resolver, manager],
    tasks=[task1, task2, task3]
)

# ============================================================
# RUN SYSTEM
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPLETE CUSTOMER SUPPORT SYSTEM")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    result = crew.kickoff()
    
    # Save to file
    filename = f"support_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(filename, "w") as f:
        f.write("="*80 + "\n")
        f.write("CUSTOMER SUPPORT SYSTEM LOG\n")
        f.write("="*80 + "\n\n")
        f.write(str(result))
        f.write("\n\n" + "="*80)
    
    print("="*80)
    print(result)
    print("\n" + "="*80)
    print(f"✅ Support log saved to: {filename}")
    print("="*80 + "\n")