from crewai import Agent, Task, Crew
from crewai.llm import LLM
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

llm = LLM(model="ollama/neural-chat", base_url="http://localhost:11434")

# ============================================================
# AGENT 1: The Writer
# ============================================================
writer = Agent(
    role="Content Writer",
    goal="Write engaging, informative blog posts",
    backstory="You are a creative writer who produces high-quality content",
    llm=llm
)

# ============================================================
# AGENT 2: The Editor
# ============================================================
editor = Agent(
    role="Content Editor",
    goal="Improve and refine written content",
    backstory="You are a meticulous editor who enhances clarity and flow",
    llm=llm
)

# ============================================================
# AGENT 3: The Publisher
# ============================================================
publisher = Agent(
    role="Publishing Specialist",
    goal="Prepare content for publication",
    backstory="You format and optimize content for maximum impact",
    llm=llm
)

# ============================================================
# TASK 1: Writer creates the draft
# ============================================================
write_task = Task(
    description="Write a blog post about 'The Future of Remote Work in 2025'. Make it engaging and informative. Include 3-4 key points.",
    expected_output="A complete blog post draft",
    agent=writer
)

# ============================================================
# TASK 2: Editor improves the draft
# ============================================================
edit_task = Task(
    description="Review and improve the blog post. Fix grammar, enhance clarity, improve flow, and make it more compelling. Keep all key points.",
    expected_output="An improved, polished version of the blog post",
    agent=editor
)

# ============================================================
# TASK 3: Publisher prepares final version
# ============================================================
publish_task = Task(
    description="Prepare the blog post for publication. Add a catchy title, format it nicely, add SEO tips, and prepare metadata (summary for social media).",
    expected_output="Final publication-ready blog post with title, content, and social media summary",
    agent=publisher
)

# ============================================================
# CREATE THE CREW (Team of agents)
# ============================================================
crew = Crew(
    agents=[writer, editor, publisher],
    tasks=[write_task, edit_task, publish_task]
)

# ============================================================
# RUN THE PIPELINE
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("CONTENT CREATION PIPELINE")
    print("="*80)
    print("\nPhase 1: Writer creates draft...")
    print("Phase 2: Editor refines...")
    print("Phase 3: Publisher prepares final version...")
    print("\n" + "="*80 + "\n")
    
    # Run the crew
    result = crew.kickoff()
    
    # Save the final result
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"blog_post_{timestamp}.txt"
    
    with open(filename, "w") as f:
        f.write("="*80 + "\n")
        f.write("FINAL BLOG POST (READY FOR PUBLICATION)\n")
        f.write("="*80 + "\n\n")
        f.write(str(result))
        f.write("\n\n" + "="*80)
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80)
    
    # Display results
    print("="*80)
    print("FINAL OUTPUT:")
    print("="*80)
    print(result)
    print("\n" + "="*80)
    print(f"✅ Blog post saved to: {filename}")
    print("="*80 + "\n")