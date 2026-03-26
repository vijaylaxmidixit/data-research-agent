"""
src/tools/kaggle_tool.py — Kaggle Dataset Search Tool
======================================================
Searches Kaggle's public dataset repository via REST API.

Platform: kaggle.com
Auth: Username + API key (from .env / ~/.kaggle/kaggle.json)
API: https://www.kaggle.com/api/v1/datasets/list
Docs: https://github.com/Kaggle/kaggle-api

Key design decisions:
    - Uses REST API directly instead of Kaggle SDK
      (SDK had breaking changes in newer versions)
    - Query simplification: Kaggle rejects multi-word queries with 400 errors
      so we map DE terms to single keywords via DE_KEYWORDS dict
    - Handles LLM passing full JSON as query string (common LLM behavior)
"""

#  LangChain tool base class 
from langchain.tools import BaseTool
from pydantic import BaseModel

#  Standard library 
from dotenv import load_dotenv
import os
import json
import requests

# Load .env so KAGGLE_USERNAME and KAGGLE_KEY are available
load_dotenv()


#  Query keyword mapping 
# Kaggle's search API rejects multi-word queries (returns 400 Bad Request).
# This map converts common data engineering terms into single Kaggle-friendly
# search keywords that return relevant results.
#
# Example: "ETL pipeline datasets" → splits into ["ETL", "pipeline", "datasets"]
#          → "ETL" matches "etl" key → returns "transactions"
#          → Kaggle search for "transactions" returns relevant DE datasets
DE_KEYWORDS = {
    "etl": "transactions",        # ETL → transaction datasets
    "pipeline": "transactions",   # pipeline → transaction datasets
    "warehouse": "warehouse",     # data warehouse → warehouse datasets
    "streaming": "streaming",     # streaming → streaming datasets
    "kafka": "streaming",         # Kafka → streaming datasets
    "batch": "transactions",      # batch processing → transaction datasets
    "retail": "retail",           # retail → retail datasets
    "transactions": "transactions",
    "timeseries": "timeseries",   # time series → timeseries datasets
    "sensor": "sensor",           # IoT sensor → sensor datasets
    "engineering": "transactions", # data engineering → transaction datasets
    "data": "transactions",       # generic "data" → transactions
    "datasets": "transactions",   # generic "datasets" → transactions
}


def simplify_query(query: str) -> str:
    """
    Converts a multi-word query into a single Kaggle-compatible keyword.

    Strategy:
        1. Lowercase and strip the query
        2. Check each word against DE_KEYWORDS map
        3. Return the mapped keyword if found
        4. Fall back to the first word of the query
        5. Fall back to "data" if query is empty

    Args:
        query: Raw search query (e.g. "ETL pipeline datasets for practice")

    Returns:
        str: Single keyword safe for Kaggle API (e.g. "transactions")
    """
    q = query.lower().strip()
    # Check each word against the keyword map
    for word in q.split():
        if word in DE_KEYWORDS:
            return DE_KEYWORDS[word]
    # Fallback: use first word of query
    return q.split()[0] if q.split() else "data"


#  Input schema 
class KaggleSearchInput(BaseModel):
    """Pydantic model that validates the LLM's tool input before _run() is called."""
    query: str          # Search term — will be simplified internally
    max_results: int = 10  # Number of datasets to return (default: 10)


#  Tool class 
class KaggleDatasetTool(BaseTool):
    """
    LangChain tool for searching Kaggle datasets.

    The agent calls this tool by:
        Action: kaggle_search
        Action Input: {"query": "retail sales", "max_results": 5}
    """

    # Tool identity — used by the LLM to identify and call this tool
    name: str = "kaggle_search"
    description: str = """Search Kaggle for public datasets.
    Input: JSON with 'query' (search term string) and optional 'max_results' (integer).
    Output: JSON list with name, size_mb, votes, url, description."""
    args_schema: type = KaggleSearchInput

    def _run(self, query: str, max_results: int = 10, **kwargs) -> str:
        """
        Executes the Kaggle dataset search.

        Steps:
            1. Load credentials from environment
            2. Handle LLM passing JSON string as query
            3. Simplify query to single keyword
            4. Call Kaggle REST API with basic auth
            5. Parse and return results as JSON string

        Args:
            query: Search term from the LLM
            max_results: Number of datasets to return

        Returns:
            str: JSON array of dataset objects, or JSON error object
        """
        # Reload credentials in case .env wasn't loaded at import time
        load_dotenv()
        username = os.getenv("KAGGLE_USERNAME", "")
        key = os.getenv("KAGGLE_KEY", "")

        # Handle edge case: LLM sometimes passes the full JSON object as query
        # e.g. query = '{"query": "ETL pipeline", "max_results": 5}'
        if query.strip().startswith("{"):
            try:
                parsed = json.loads(query)
                query = parsed.get("query", query)
            except Exception:
                pass  # Not valid JSON — use as-is

        # Convert multi-word DE query to single Kaggle-compatible keyword
        simple_query = simplify_query(query)

        try:
            # Call Kaggle REST API directly
            # auth=(username, key) uses HTTP Basic Authentication
            resp = requests.get(
                "https://www.kaggle.com/api/v1/datasets/list",
                params={
                    "search": simple_query,   # Single keyword search
                    "sortBy": "votes",        # Sort by community votes (quality signal)
                    "pageSize": max_results,  # Limit results
                },
                auth=(username, key),
                timeout=15,   # 15 second timeout
            )

            # Handle non-200 responses
            if resp.status_code != 200:
                return json.dumps({"error": f"Kaggle API error: {resp.status_code}"})

            results = resp.json()

            # Validate response is a list
            if not isinstance(results, list):
                return json.dumps({"error": "Unexpected Kaggle response"})

            # Normalize fields — Kaggle API uses inconsistent field names
            # (some fields have "Nullable" suffix in newer API versions)
            datasets = []
            for ds in results[:max_results]:
                datasets.append({
                    "name": ds.get("title", ds.get("titleNullable", "")),
                    "ref": ds.get("ref", ""),   # Unique dataset identifier
                    "size_mb": round(ds.get("totalBytes", 0) / 1e6, 2),  # Convert bytes to MB
                    "votes": ds.get("voteCount", ds.get("voteCountNullable", 0)),
                    "url": f"https://www.kaggle.com/datasets/{ds.get('ref', '')}",
                    "license": ds.get("licenseName", ds.get("licenseNameNullable", "")),
                    "description": ds.get("subtitleNullable", ds.get("subtitle", "")),
                })
            return json.dumps(datasets, indent=2)

        except requests.exceptions.Timeout:
            return json.dumps({"error": "Kaggle API timed out after 15s"})
        except Exception as e:
            return json.dumps({"error": f"Kaggle tool error: {str(e)}"})

    async def _arun(self, *args, **kwargs):
        """Async version — not implemented (agent runs synchronously)."""
        raise NotImplementedError("Async not supported")