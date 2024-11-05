import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from Tools.search_tools import SearchTools  # Importing the SearchTools class
from langchain_openai import ChatOpenAI  # Importing the ChatOpenAI class

# Load environment variables from .env
load_dotenv()

# Get the Serper API key
serper_api_key = os.getenv('SERPER_API_KEY')

# Check if the Serper API key was loaded correctly
if not serper_api_key:
    raise ValueError(
        "The Serper API key was not found. Check the .env file")

# Instantiate the search tool using the SearchTools class
search_tool = SearchTools()

# 1. Research Agent - using Serper for searches
researcher = Agent(
    role="Researcher",
    goal="""Research step-by-step guidance on a typical data science project involving preprocessing,
    exploratory data analysis, and how to select, implement, and optimize a machine learning model.
    """,
    backstory="""Research best practices for data preparation and machine learning implementation to
    create a robust and suitable model for a typical data science project.""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool.search_internet]  # Using the class's search method
)

# 2. Writing Agent
writer = Agent(
    role="Writer",
    goal="Develop a complete tutorial explaining how to execute a data science project using machine learning.",
    backstory="""Structure the explanation in an educational way, covering preparation, EDA, model selection, creation, optimization,
    and implementation examples.""",
    verbose=True,
    allow_delegation=False
)

# 3. Developer Agent
developer = Agent(
    role="Developer",
    goal="Provide practical code examples for a data science project.",
    backstory="""Create code for a Jupyter notebook containing the steps to execute a data science project""",
    verbose=True,
    allow_delegation=False
)

# 4. Review Agent
reviewer = Agent(
    role="Reviewer",
    goal="Review the created tutorial to ensure it is clear, accurate, and easy to understand.",
    backstory="""Verify if the tutorial covers all essential aspects, if the code is correct, and if the reader can
    follow the instructions without difficulties.""",
    verbose=True,
    allow_delegation=False
)

# Instantiate the GPT-4 Mini model
OpenAIGPT4Mini = ChatOpenAI(
    model="gpt-4-mini"  # Specifies the model as GPT-4 Mini
)

# Define tasks for each agent
task1 = Task(
    description="""Research how to implement a data science project and the best practices""",
    agent=researcher,
    expected_output="Step-by-step of a data science project"
)

task2 = Task(
    description="""Create a tutorial structure explaining how to execute a data science project, covering preprocessing, EDA, model selection, implementation, and optimization of a machine learning model""",
    agent=writer,
    expected_output="Complete structure of the tutorial on a data science project."
)

task3 = Task(
    description="""Implement a practical example of a data science project using pandas, numpy, scikit-learn, and other libraries as needed. Provide commented code.""",
    agent=developer,
    expected_output="Source code of a Jupyter notebook for a data science project, with comments and explanations."
)

task4 = Task(
    description="""Review the tutorial and code, ensuring the content is understandable and examples are correct.""",
    agent=reviewer,
    expected_output="Complete review of the tutorial and code, ensuring clarity and accuracy."
)

# Instantiate the crew and add tasks
crew = Crew(
    agents=[researcher, writer, developer, reviewer],
    tasks=[task1, task2, task3, task4],
    verbose=True,
    process=Process.sequential,  # Tasks will be executed one after another
    manager_llm=OpenAIGPT4Mini  # Adds the GPT-4 Mini model as manager
)

# Start the agent execution process
result = crew.kickoff()

# Display results
print("######################")
print(result)

# Create report

# Step 1: Define the variable with the content
markdown_content = result

# Check the type of the variable
if not isinstance(markdown_content, str):
    markdown_content = str(markdown_content)  # Convert to string if necessary

# Step 2: Open a file in write mode
with open('report.md', 'w') as file:
    # Step 3: Write the content to the file
    file.write(markdown_content)

# The file is automatically closed after the with block
