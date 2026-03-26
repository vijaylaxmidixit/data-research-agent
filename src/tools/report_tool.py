"""
src/tools/report_tool.py — Markdown Report Generator Tool
==========================================================
Generates and saves a structured Markdown report of discovered datasets.

This is the final tool called by the agent after:
    1. Collecting datasets from multiple platforms
    2. Analyzing and ranking them with DatasetAnalyzerTool

The report is saved to the reports/ directory with a timestamp filename.

Output format:
    reports/2026-03-17-1745-ETL-pipeline-datasets.md

Key design decision:
    - Accepts flexible input (JSON string, Python dict, or plain text)
      because the LLM's final output format varies between runs
    - Uses ast.literal_eval as fallback for Python-style dicts
    - Creates reports/ directory if it doesn't exist
"""

from langchain.tools import BaseTool
from pydantic import BaseModel
from datetime import datetime
import json
import ast
import os


class ReportInput(BaseModel):
    """
    Single field input schema.
    Using one field instead of two (query + datasets) because the LLM
    reliably passes everything in one JSON string, which is easier to parse.
    """
    report_data: str  # JSON string containing query and datasets list


class MarkdownReportTool(BaseTool):
    """
    LangChain tool that generates and saves Markdown research reports.

    Called as the last step in the agent's workflow to persist
    the research findings to disk.
    """

    name: str = "generate_report"
    description: str = """Generate a Markdown report from analyzed datasets.
    Input: a JSON string with 'query' and 'datasets' keys, OR just a plain string description.
    Output: confirmation message with path to saved report file."""
    args_schema: type = ReportInput

    def _run(self, report_data: str, **kwargs) -> str:
        """
        Parses input and generates a Markdown report file.

        Input parsing (in order):
            1. Try JSON.loads() — valid JSON string
            2. Try ast.literal_eval() — Python dict format
            3. Use as plain text — save whatever the LLM provided

        Report sections:
            - Header: query, timestamp, dataset count
            - Top Datasets: numbered list with URL, size, license, score
              (max 10 datasets shown)

        Args:
            report_data: JSON string with 'query' and 'datasets' keys

        Returns:
            str: Confirmation message with path to saved file
        """
        try:
            # ── Parse Input ───────────────────────────────────────────────────
            try:
                # Attempt 1: Standard JSON
                data = json.loads(report_data)
            except (json.JSONDecodeError, ValueError):
                try:
                    # Attempt 2: Python dict format
                    data = ast.literal_eval(report_data)
                except Exception:
                    # Attempt 3: Plain text — wrap it in a basic structure
                    data = {"query": "research", "datasets": [], "summary": report_data}

            # ── Extract Fields ────────────────────────────────────────────────
            # Flexible extraction — handles different key names the LLM might use
            query = data.get("query", "dataset-research")
            datasets = data.get("datasets", data.get("datasets_json", []))

            # If datasets is a string (LLM sometimes nests JSON in JSON), parse it
            if isinstance(datasets, str):
                try:
                    datasets = json.loads(datasets)
                except Exception:
                    try:
                        datasets = ast.literal_eval(datasets)
                    except Exception:
                        datasets = []   # Empty list as last resort

            # ── File Setup ────────────────────────────────────────────────────
            # Create reports/ directory if it doesn't exist
            os.makedirs("reports", exist_ok=True)

            # Generate timestamped filename
            # Format: reports/2026-03-17-1745-ETL-pipeline-datasets.md
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M")
            safe_query = query[:30].replace(" ", "-").replace("/", "-")
            filename = f"reports/{timestamp}-{safe_query}.md"

            # ── Build Report Content ──────────────────────────────────────────
            lines = [
                "# Dataset Research Report",
                "",
                f"**Query:** {query}",
                f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                f"**Total datasets found:** {len(datasets) if isinstance(datasets, list) else 'N/A'}",
                "",
                "---",
                "",
                "## Top Datasets",
                "",
            ]

            # Add one section per dataset (max 10)
            if isinstance(datasets, list):
                for i, ds in enumerate(datasets[:10], 1):
                    if isinstance(ds, dict):
                        lines.append(f"### {i}. {ds.get('name', 'Unknown')}")
                        lines.append(f"- **URL:** {ds.get('url', 'N/A')}")
                        # size_mb for Kaggle, num_instances for UCI
                        lines.append(f"- **Size:** {ds.get('size_mb', ds.get('num_instances', 'N/A'))}")
                        lines.append(f"- **License:** {ds.get('license', 'N/A')}")
                        lines.append(f"- **Relevance Score:** {ds.get('relevance_score', 'N/A')}")
                        if ds.get("description"):
                            lines.append(f"- **Description:** {ds.get('description', '')}")
                        lines.append("")   # Blank line between datasets
            else:
                # Fallback: just write whatever data we have
                lines.append(str(data))

            # ── Save to Disk ──────────────────────────────────────────────────
            with open(filename, "w") as f:
                f.write("\n".join(lines))

            dataset_count = len(datasets) if isinstance(datasets, list) else 0
            return f"Report saved to {filename} with {dataset_count} datasets."

        except Exception as e:
            return f"Report generation error: {str(e)}"

    async def _arun(self, *args, **kwargs):
        """Async version — not implemented."""
        raise NotImplementedError("Async not supported")