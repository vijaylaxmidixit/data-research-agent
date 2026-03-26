"""
main.py — Entry point for data-research-agent
========================================
A local AI agent that discovers, retrieves, and analyzes public datasets
from 5 open data platforms using Qwen3 8B via Ollama.

Usage:
    python main.py --query "ETL pipeline datasets"
    python main.py --query "time series sensor data" --max-results 10
"""

#  Standard library imports 
import os
import time

#  Load environment variables FIRST 
# IMPORTANT: load_dotenv() must be called before any other imports. 
# so that Kaggle, HuggingFace, and Ollama credentials are available
# when the tool modules are loaded.
from dotenv import load_dotenv
load_dotenv()

# Explicitly set Kaggle credentials in the environment.
# The Kaggle SDK reads from os.environ, not directly from .env,
# so we must bridge them here.
os.environ["KAGGLE_USERNAME"] = os.getenv("KAGGLE_USERNAME", "")
os.environ["KAGGLE_KEY"] = os.getenv("KAGGLE_KEY", "")

#  Third-party imports 
import typer                          # CLI framework — builds --query, --max-results flags
from rich.console import Console      # Pretty terminal output with colors

#  Internal imports 
# Imported AFTER load_dotenv() so all env vars are available
from src.agent.research_agent import build_agent

#  CLI App setup 
app = typer.Typer()       # Typer app — registers CLI commands
console = Console()       # Rich console — used for colored output


@app.command()
def main(
    query: str = typer.Option(
        ...,            # ... means this argument is REQUIRED
        "--query", "-q",
        help="Research query for dataset discovery (e.g. 'ETL pipeline datasets')"
    ),
    max_results: int = typer.Option(
        5,              # Default value — 5 results per platform
        "--max-results", "-n",
        help="Maximum number of datasets to retrieve per platform"
    ),
):
    """
    dataset-scout — Local AI Dataset Discovery

    Searches Kaggle, HuggingFace, UCI, Data.gov, and OpenDataSoft
    for datasets relevant to your query using a local Qwen3 8B LLM.
    Generates a Markdown report saved to the reports/ folder.
    """

    # Print the query to terminal in bold blue
    console.print(f"\n[bold blue]Researching:[/] {query}\n")

    # Record start time to measure total agent execution time
    start = time.time()

    # Build the ReAct agent with all 7 tools loaded
    # This initializes the LLM connection to Ollama and all API tools
    agent = build_agent()

    # Invoke the agent — this starts the ReAct reasoning loop:
    # 1. LLM receives the query
    # 2. LLM decides which tool to call
    # 3. Tool returns results
    # 4. LLM analyzes results and decides next action
    # 5. Loop continues until LLM reaches "Final Answer"
    result = agent.invoke({
        "input": f"Research query: {query}. Max {max_results} results per platform."
    })

    # Calculate total elapsed time
    elapsed = round(time.time() - start, 1)

    # Print the agent's final answer
    console.print(result["output"])

    # Print elapsed time in muted gray
    console.print(f"\n[dim]Completed in {elapsed}s[/dim]")


#  Script entry point 
if __name__ == "__main__":
    # Run the Typer CLI app when script is executed directly
    # e.g. python main.py --query "retail datasets"
    app()