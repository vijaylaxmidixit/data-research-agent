"""
src/agent/research_agent.py — Core ReAct Agent
================================================
Builds and configures the LangChain ReAct agent that powers dataset-scout.

The ReAct (Reason + Act) pattern works like this:
    1. LLM receives a question
    2. LLM thinks (Thought) about what to do
    3. LLM picks a tool (Action) and provides input (Action Input)
    4. Tool runs and returns a result (Observation)
    5. LLM thinks again based on the observation
    6. Loop repeats until LLM writes "Final Answer"

This file:
    - Defines the system prompt that guides the LLM behavior
    - Loads all 7 tools (5 platform searches + analyzer + reporter)
    - Connects to Qwen3 8B running locally via Ollama
    - Returns a configured AgentExecutor ready to invoke
"""

#  LangChain imports 
from langchain.agents import AgentExecutor          # Runs the ReAct loop
from langchain.agents import create_react_agent     # Creates the ReAct agent
from langchain_ollama import OllamaLLM              # Local LLM via Ollama
from langchain.prompts import PromptTemplate        # Structures the system prompt

#  Standard library 
from dotenv import load_dotenv
import os

# ── Tool imports — each tool connects to one data platform ───────────────────
from src.tools.kaggle_tool import KaggleDatasetTool        # Kaggle REST API
from src.tools.huggingface_tool import HFDatasetTool       # HuggingFace Hub API
from src.tools.uci_tool import UCIDatasetTool              # UCI ML Repository (curated)
from src.tools.datagov_tool import DataGovTool             # Data.gov CKAN API
from src.tools.ods_tool import OpenDataSoftTool            # OpenDataSoft API
from src.tools.analyzer_tool import DatasetAnalyzerTool    # Ranks datasets by relevance
from src.tools.report_tool import MarkdownReportTool       # Saves Markdown report to disk

load_dotenv()


#  ReAct Prompt Template 
# This is the system prompt that tells the LLM:
#   - What its role is
#   - What tools are available
#   - The exact format to use for Thought/Action/Observation
#   - Important rules for each tool
#
# Template variables filled at runtime:
#   {tools}           — auto-filled with tool names + descriptions
#   {tool_names}      — auto-filled with comma-separated tool names
#   {input}           — the user's research query
#   {agent_scratchpad} — the LLM's running chain of thought
REACT_PROMPT = PromptTemplate.from_template("""You are a Data Engineering Research Assistant.
Your goal is to discover, retrieve, and analyze public datasets useful for data engineering practice.

You have access to the following tools:
{tools}

Use this format STRICTLY:

Question: the input question you must answer
Thought: think about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original question

IMPORTANT TOOL USAGE:
- kaggle_search: input must be {{"query": "search term", "max_results": 5}}
- huggingface_search: input must be {{"query": "search term", "max_results": 5}}
- uci_search: input must be {{"query": "search term", "max_results": 5}}
- datagov_search: input must be {{"query": "search term", "max_results": 5}}
- opendatasoft_search: input must be {{"query": "search term", "max_results": 5}}
- dataset_analyzer: input must be a valid JSON array string like [{{"name": "x", "url": "y"}}]
- generate_report: input must be a JSON string like {{"query": "my query", "datasets": [...]}}

Begin!

Question: {input}
Thought: {agent_scratchpad}""")


def build_agent() -> AgentExecutor:
    """
    Builds and returns a configured ReAct AgentExecutor.

    Steps:
        1. Connect to Qwen3 8B running locally via Ollama
        2. Initialize all 7 tools
        3. Create the ReAct agent with the prompt
        4. Wrap in AgentExecutor with safety limits

    Returns:
        AgentExecutor: Ready-to-invoke agent with all tools loaded
    """

    #  LLM Configuration 
    # Qwen3 8B runs locally via Ollama — no data leaves your machine
    # temperature=0.1 → near-deterministic responses (better for tool use)
    # num_ctx=8192    → context window size (handles large dataset metadata)
    llm = OllamaLLM(
        model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.1,
        num_ctx=8192,
    )

    #  Tool Registration 
    # Each tool is a LangChain BaseTool subclass with:
    #   - name: how the LLM refers to the tool
    #   - description: tells the LLM when and how to use this tool
    #   - _run(): the actual logic executed when tool is called
    tools = [
        KaggleDatasetTool(),      # Searches kaggle.com via REST API
        HFDatasetTool(),          # Searches huggingface.co/datasets API
        UCIDatasetTool(),         # Returns curated UCI ML Repository datasets
        DataGovTool(),            # Searches catalog.data.gov CKAN API
        OpenDataSoftTool(),       # Searches data.opendatasoft.com API
        DatasetAnalyzerTool(),    # Scores and ranks collected datasets
        MarkdownReportTool(),     # Saves final report to reports/ folder
    ]

    #  Agent Creation 
    # create_react_agent combines: LLM + tools + prompt template
    # into a runnable agent that follows the ReAct reasoning pattern
    agent = create_react_agent(llm, tools, REACT_PROMPT)

    #  AgentExecutor Configuration 
    # AgentExecutor manages the ReAct loop:
    #   max_iterations=10      → stops after 10 tool calls (prevents infinite loops)
    #   handle_parsing_errors  → recovers gracefully if LLM format is wrong
    #   verbose=True           → prints each Thought/Action/Observation to terminal
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=10,
        handle_parsing_errors=True,
        verbose=True,
    )