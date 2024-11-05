# Data Science Project Assistant with Serper Search

## Table of Contents
- [Data Science Project Assistant with Serper Search](#data-science-project-assistant-with-serper-search)
  - [Table of Contents](#table-of-contents)
  - [Project Objective](#project-objective)
  - [Project Structure and Steps](#project-structure-and-steps)
  - [Tools and Techniques Utilized](#tools-and-techniques-utilized)
  - [Specific Results and Outcomes](#specific-results-and-outcomes)
  - [What I Have Learned from This Project](#what-i-have-learned-from-this-project)
  - [How to Use This Repository](#how-to-use-this-repository)

---

## Project Objective

This project aims to automate and streamline the process of researching, creating, and reviewing a comprehensive data science tutorial. It leverages the Serper API to fetch data from internet searches, integrates with an OpenAI model, and coordinates multiple "agents" with specific roles (Researcher, Writer, Developer, Reviewer). The system produces a well-structured data science tutorial, code examples, and a final review to ensure clarity and accuracy.

[back to top](#)

## Project Structure and Steps

1. **Environment Setup**:
   - Loading environment variables from a `.env` file, which includes the Serper API key.

2. **Agent Definition**:
   - **Researcher**: Searches for best practices and guides for data science projects.
   - **Writer**: Compiles the information into a structured tutorial.
   - **Developer**: Creates example code for a Jupyter notebook, demonstrating steps in a data science project.
   - **Reviewer**: Reviews the entire tutorial and code to ensure clarity, accuracy, and ease of use.

3. **Search Tools**:
   - A custom class `SearchTools` is created to perform internet searches via the Serper API. This tool fetches results on given topics, allowing agents to source relevant information for tutorial creation.

4. **Execution Process**:
   - Agents are coordinated in a sequence where each performs their task one after the other.
   - The `Crew` class manages task allocation and uses the `Process` module to handle task execution.

5. **Report Generation**:
   - The final output is saved in a markdown file `report.md`, consolidating all research, structured tutorial, code samples, and the review summary.

[back to top](#)

## Tools and Techniques Utilized

- **Python Libraries**:
  - `os`, `json`, `requests`: For handling environment variables, JSON data, and HTTP requests.
  - `dotenv`: To manage environment variables securely.
  - `langchain`: Utilized to manage and create various tools for the agents, including internet search capabilities.
  - `crewAI`: Utilized to create the agents, tools, tasks and organize everything into a process

- **API Integrations**:
  - **Serper API**: Provides search functionality that enables agents to perform internet research.

- **Multi-Agent Coordination**:
  - Use of distinct agent roles (Researcher, Writer, Developer, Reviewer) to replicate a structured workflow similar to a data science project lifecycle.

- **Language Model (LLM)**:
  - Integration of the GPT-4 Mini model as an oversight manager, coordinating agents and their tasks.

[back to top](#)

## Specific Results and Outcomes

- **Generated Data Science Tutorial**:
  - The project outputs a tutorial covering the full lifecycle of a data science project, from data preparation and exploratory data analysis to model selection, training, and optimization.

- **Practical Code Examples**:
  - A Markdown file is generated, demonstrating practical steps in data science using Python libraries such as `pandas`, `numpy`, and `scikit-learn`.

- **Enhanced Research Capabilities**:
  - The project automates web search for relevant topics, pulling information dynamically from the internet to keep the tutorial up-to-date with best practices.

- **Streamlined Workflow for Documentation**:
  - A clear workflow is established with task delegation among specialized agents, creating a structured, easy-to-follow documentation process.

[back to top](#)

## What I Have Learned from This Project

Through this project, I developed a range of technical and project management skills:

- **Enhanced Knowledge of Data Science Lifecycle**: Learned about each phase in a data science project, including best practices for data preparation, model training, and evaluation.
  
- **Multi-Agent Systems in Python**: Gained experience in setting up and managing multiple agents with specific roles, enabling a modular and efficient approach to complex workflows.

- **API Integration and Data Handling**: Improved skills in integrating APIs like Serper, handling JSON data, and managing API keys securely.

- **Process Automation**: Strengthened my ability to create automated workflows for task execution and documentation, significantly improving efficiency.

- **Using LLMs as Workflow Managers**: Developed competencies in integrating large language models as task managers, which added coordination and intelligence to the workflow.

This project has helped me refine my approach to project structuring, collaborative task management, and practical skills in data science, making it a valuable learning experience and a robust framework for similar future projects.

[back to top](#)

## How to Use This Repository

Follow these steps to set up and run the project:

1. **Clone the Repository**:
   - Clone this repository to your local machine:
     ```bash
     git clone https://github.com/yourusername/yourrepositoryname.git
     cd yourrepositoryname
     ```

2. **Set Up Environment Variables**:
   - Create a `.env` file in the root directory of the project.
   - Add your API keys for OpenAI and Serper in the `.env` file:
     ```
     OPENAI_API_KEY=your_openai_api_key
     SERPER_API_KEY=your_serper_api_key
     ```
   - These keys are required for accessing the OpenAI model and Serper search functionality.

3. **Install Required Packages**:
   - Ensure you have Python 3.6 or higher installed. Install dependencies using:
     ```bash
     pip install -r requirements.txt
     ```

4. **Run the Project**:
   - Execute the main script to initiate the workflow:
     ```bash
     python main.py
     ```
   - This will start the agents’ task execution and produce the final report.

5. **View the Results**:
   - After running the project, a file named `report.md` will be generated, containing the tutorial, code examples, and the review summary.

This project can be re-used for any other purpose, simply change agent intructions and tasks and it should execute as expected. Additional agents and tasks can also be created, further refining the application.

[back to top](#)