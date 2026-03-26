"""
src/tools/huggingface_tool.py — HuggingFace Dataset Search Tool
================================================================
Searches HuggingFace Hub for public datasets via REST API.

Platform: huggingface.co/datasets
Auth: Optional (HF token in .env) — public datasets work without auth
API: https://huggingface.co/api/datasets
Docs: https://huggingface.co/docs/hub/api

Key design decisions:
    - No sort/direction params — they cause empty results on HF API
    - Query simplification: maps DE terms to HF-friendly single keywords
    - Strips tags with ":" (internal HF metadata tags like "task_categories:tabular")
    - Truncates long descriptions to 200 chars for LLM context efficiency
"""

#  LangChain tool base class 
from langchain.tools import BaseTool
from pydantic import BaseModel

#  Standard library 
import requests
import json


#  Query keyword mapping 
# HuggingFace search works best with single descriptive keywords.
# This map converts data engineering terminology to HF-friendly search terms
# that return relevant tabular/structured datasets.
DE_KEYWORDS = {
    "etl": "tabular",            # ETL → tabular data
    "pipeline": "tabular",       # data pipeline → tabular
    "warehouse": "tabular",      # data warehouse → tabular
    "streaming": "timeseries",   # streaming → time series data
    "kafka": "timeseries",       # Kafka → time series
    "batch": "tabular",          # batch processing → tabular
    "retail": "retail",          # retail → retail datasets
    "transactions": "transactions",
    "timeseries": "timeseries",  # time series → timeseries
    "sensor": "sensor",          # IoT sensors → sensor data
    "engineering": "tabular",    # data engineering → tabular
    "data": "tabular",           # generic → tabular
}


def simplify_query(query: str) -> str:
    """
    Converts a multi-word query into a single HuggingFace-compatible keyword.

    Args:
        query: Raw query string from the LLM

    Returns:
        str: Single keyword for HF API search
    """
    q = query.lower().strip()
    for word in q.split():
        if word in DE_KEYWORDS:
            return DE_KEYWORDS[word]
    # Fallback: first word, or "tabular" if empty
    return q.split()[0] if q.split() else "tabular"


# ── Input schema ──────────────────────────────────────────────────────────────
class HFSearchInput(BaseModel):
    """Validates the LLM's input before _run() executes."""
    query: str
    max_results: int = 10


# ── Tool class ────────────────────────────────────────────────────────────────
class HFDatasetTool(BaseTool):
    """
    LangChain tool for searching HuggingFace Hub datasets.

    Returns public datasets with download counts, tags, and direct URLs.
    Requires no authentication for public datasets.
    """

    name: str = "huggingface_search"
    description: str = """Search HuggingFace Hub for public datasets.
    Input: JSON with 'query' (search term string) and optional 'max_results' (integer).
    Output: JSON list with name, downloads, likes, url, tags."""
    args_schema: type = HFSearchInput

    def _run(self, query: str, max_results: int = 10, **kwargs) -> str:
        """
        Searches HuggingFace Hub datasets API.

        Steps:
            1. Handle LLM passing full JSON as query string
            2. Simplify query to single keyword
            3. Call HF API (no sort params — they break results)
            4. Clean up tags (remove internal HF metadata tags)
            5. Truncate long descriptions
            6. Return normalized JSON array

        Returns:
            str: JSON array of dataset objects, or JSON error object
        """
        try:
            # Handle edge case: LLM sometimes passes full JSON as query
            if query.strip().startswith("{"):
                try:
                    parsed = json.loads(query)
                    query = parsed.get("query", query)
                except Exception:
                    pass

            # Convert to single HF-friendly search keyword
            search_term = simplify_query(query)

            # Call HuggingFace Hub API
            # Note: Do NOT add sort/direction params — they return empty results
            resp = requests.get(
                "https://huggingface.co/api/datasets",
                params={
                    "search": search_term,    # Single keyword search
                    "limit": max_results,     # Max results to return
                },
                timeout=15,
                headers={"Accept": "application/json"},
            )

            # Handle API errors
            if resp.status_code != 200:
                return json.dumps({"error": f"HuggingFace API error: {resp.status_code}"})

            results = resp.json()

            # Handle empty results
            if not isinstance(results, list) or len(results) == 0:
                return json.dumps({"message": "No datasets found", "query": search_term})

            datasets = []
            for ds in results[:max_results]:
                # Truncate long descriptions to save LLM context tokens
                desc = ds.get("description", "")
                if desc and len(desc) > 200:
                    desc = desc[:200] + "..."

                datasets.append({
                    "name": ds.get("id", ""),            # Format: "author/dataset-name"
                    "downloads": ds.get("downloads", 0), # Download count (quality signal)
                    "likes": ds.get("likes", 0),         # Community likes
                    "url": f"https://huggingface.co/datasets/{ds.get('id', '')}",
                    # Filter out internal HF tags (they contain ":" like "task_categories:tabular")
                    # Keep only human-readable tags like "retail", "tabular", "english"
                    "tags": [t for t in ds.get("tags", []) if ":" not in t][:5],
                    "description": desc,
                })
            return json.dumps(datasets, indent=2)

        except requests.exceptions.Timeout:
            return json.dumps({"error": "HuggingFace API timed out"})
        except Exception as e:
            return json.dumps({"error": f"HuggingFace tool error: {str(e)}"})

    async def _arun(self, *args, **kwargs):
        """Async version — not implemented."""
        raise NotImplementedError("Async not supported")