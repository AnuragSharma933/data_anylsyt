import os
import shutil
from typing import Optional, Dict, Any
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from tools import inspect_dataset_schema, execute_python_analytics, set_current_data_path

def clean_output_directory():
    """Ensure clean outputs folder structure before running new analysis."""
    os.makedirs("outputs/charts", exist_ok=True)
    for filename in os.listdir("outputs/charts"):
        file_path = os.path.join("outputs/charts", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    final_report = "outputs/final_report.md"
    if os.path.exists(final_report):
        os.remove(final_report)

def get_groq_llm(api_key: str, model_name: str) -> ChatGroq:
    """Initialize Groq LLM instance for CrewAI agents."""
    model_id = model_name
    if not model_id.startswith("groq/") and not "/" in model_id:
        model_id = model_id  # langchain_groq expects model name like llama-3.3-70b-versatile
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_id,
        temperature=0.2
    )

def run_analysis_workflow(
    file_path: str,
    client_instructions: str,
    api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
    status_callback=None
) -> str:
    """
    Executes the 3-agent CrewAI workflow:
    1. Senior Data Analyst
    2. QA Reviewer
    3. Executive Deliverable Writer
    """
    # 1. Prepare environment & path
    set_current_data_path(file_path)
    clean_output_directory()

    if status_callback:
        status_callback("Initializing LLM and Agents...")

    llm = get_groq_llm(api_key=api_key, model_name=model_name)

    # Agent 1: Senior Data Analyst
    analyst_agent = Agent(
        role="Senior Data Analyst",
        goal="Inspect data schemas, calculate summary statistics, write and execute Python code for analytical insights and chart generation.",
        backstory=(
            "You are an expert Data Analyst with deep mastery over Pandas, Matplotlib, and Seaborn. "
            "You rigorously inspect data schemas, detect patterns, generate informative data visualization charts, "
            "and compute crucial descriptive metrics."
        ),
        tools=[inspect_dataset_schema, execute_python_analytics],
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Agent 2: QA Reviewer
    qa_agent = Agent(
        role="QA Reviewer",
        goal="Double-check all numbers, statistical statements, chart references, and Python logic to guarantee 0% hallucination.",
        backstory=(
            "You are a meticulous Quality Assurance Reviewer. You verify all statistics, calculations, "
            "and insights provided by data analysts. You cross-reference findings against the dataset "
            "and ensure zero hallucinations and top accuracy."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Agent 3: Executive Deliverable Writer
    writer_agent = Agent(
        role="Executive Deliverable Writer",
        goal="Package insights, key metric tables, chart references, and executive summary into a polished client-ready Markdown report.",
        backstory=(
            "You are a elite Management Consultant and Deliverable Writer. You excel at turning complex data "
            "and technical outputs into compelling, beautifully formatted, client-ready business reports."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

    # Task 1: Data Analysis & Visualization
    task1 = Task(
        description=(
            f"Analyze the dataset located at '{file_path}'.\n"
            f"Client Instructions / Focus Areas: {client_instructions if client_instructions else 'Provide a complete comprehensive analysis.'}\n"
            "Step 1: Inspect the schema and summary statistics of the dataset.\n"
            "Step 2: Write Python code using the `execute_python_analytics` tool to perform deeper analysis.\n"
            "Step 3: Generate at least 2 key visualization charts (e.g., distribution, trend, correlation, bar charts) "
            "and save them as `.png` files into `outputs/charts/` (e.g. `outputs/charts/distribution.png`).\n"
            "Provide detailed statistical findings, key trends, and list exact file paths of generated charts."
        ),
        expected_output=(
            "A comprehensive technical analysis detailing findings, exact statistics, code execution results, "
            "and the list of generated chart image file paths."
        ),
        agent=analyst_agent
    )

    # Task 2: QA Verification
    task2 = Task(
        description=(
            "Review the technical analysis provided by the Senior Data Analyst.\n"
            "1. Verify that all numbers and mathematical conclusions are supported by the dataset tools.\n"
            "2. Confirm that all referenced charts exist and were successfully generated in `outputs/charts/`.\n"
            "3. Identify any missing perspectives or logical errors, and specify verified findings ready for client delivery."
        ),
        expected_output=(
            "A QA review report validating the mathematical accuracy, statistical consistency, and chart references."
        ),
        agent=qa_agent
    )

    # Task 3: Executive Report Deliverable
    task3 = Task(
        description=(
            "Synthesize the verified analysis and QA review into a high-grade executive report.\n"
            "Requirements:\n"
            "1. Format as clean, professional Markdown.\n"
            "2. Include an Executive Summary, Key Performance Metrics table, Detailed Findings, "
            "and Business Recommendations.\n"
            "3. Include Markdown image tags referencing every generated chart in `outputs/charts/` "
            "(e.g., `![Chart Name](outputs/charts/chart_name.png)`).\n"
            "4. Write the final complete Markdown content to `outputs/final_report.md`."
        ),
        expected_output=(
            "A client-ready executive deliverable formatted in Markdown, saved to outputs/final_report.md."
        ),
        agent=writer_agent,
        output_file="outputs/final_report.md"
    )

    if status_callback:
        status_callback("Running Autonomous Agent Crew...")

    crew = Crew(
        agents=[analyst_agent, qa_agent, writer_agent],
        tasks=[task1, task2, task3],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()

    # Guarantee file output saved if crew returns text
    report_path = "outputs/final_report.md"
    if not os.path.exists(report_path) or os.path.getsize(report_path) == 0:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(str(result))

    return str(result)

if __name__ == "__main__":
    import sys
    print("CrewAI Autonomous Data Analyst Engine ready.")
