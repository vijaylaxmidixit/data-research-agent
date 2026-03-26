"""
src/tools/datagov_tool.py — Data.gov Dataset Search Tool
=========================================================
Searches the US government's open data catalog via CKAN REST API.

Platform: catalog.data.gov
Auth: None required — fully public API
API: https://catalog.data.gov/api/3/action/package_search
Docs: https://docs.ckan.org/en/latest/api/

Data.gov hosts 300K+ government datasets across health, transport,
finance, environment, and infrastructure — excellent for large-scale
real-world data engineering pipeline practice.

Key design decision:
    - Query simplification to single keywords (multi-word queries
      often return 0 results from the CKAN search engine)
"""

from langchain.tools import BaseTool
from pydantic import BaseModel
import requests
import json


class DataGovSearchInput(BaseModel):
    """Validates LLM input before _run() executes."""
    query: str
    max_results: int = 10


class DataGovTool(BaseTool):
    """
    LangChain tool for searching Data.gov (US Open Government Data).

    Returns government datasets with organization, file formats, and URLs.
    No authentication required.
    """

    name: str = "datagov_search"
    description: str = """Search Data.gov for US government public datasets.
    Input: JSON with 'query' and optional 'max_results'.
    Output: JSON list with name, organization, formats, url."""
    args_schema: type = DataGovSearchInput

    def _run(self, query: str, max_results: int = 10, **kwargs) -> str:
        """
        Searches Data.gov CKAN API for government datasets.

        Steps:
            1. Simplify query to single keyword (CKAN works better)
            2. Call CKAN package_search endpoint
            3. Extract title, organization, file formats, and URL
            4. Return normalized JSON array

        Returns:
            str: JSON array of dataset objects, or JSON error object
        """
        # Simplify multi-word query to first meaningful word
        # CKAN search returns more results with single keywords
        simple_query = query.split()[0] if len(query.split()) > 2 else query
        query = simple_query

        try:
            resp = requests.get(
                "https://catalog.data.gov/api/3/action/package_search",
                params={
                    "q": query,          # Search term
                    "rows": max_results  # Number of results
                },
                timeout=10,
            )

            # Handle API errors
            if resp.status_code != 200 or not resp.text.strip():
                return json.dumps({"error": f"DataGov API error: {resp.status_code}"})

            # CKAN response structure: {"result": {"results": [...]}}
            results = resp.json().get("result", {}).get("results", [])

            if not results:
                return json.dumps({"message": "No DataGov datasets found", "query": query})

            datasets = []
            for ds in results[:max_results]:
                # Extract unique file formats from resources list
                # (each dataset can have multiple file downloads)
                formats = list(set([
                    r.get("format", "") for r in ds.get("resources", [])
                    if r.get("format", "")  # Skip empty format strings
                ]))[:5]

                datasets.append({
                    "name": ds.get("title", ""),
                    "organization": ds.get("organization", {}).get("title", ""),
                    "formats": formats,   # e.g. ["CSV", "JSON", "XML"]
                    "url": f"https://catalog.data.gov/dataset/{ds.get('name', '')}",
                    "notes": ds.get("notes", "")[:200],  # First 200 chars of description
                })
            return json.dumps(datasets, indent=2)

        except Exception as e:
            return json.dumps({"error": f"DataGov tool error: {str(e)}"})

    async def _arun(self, *args, **kwargs):
        """Async version — not implemented."""
        raise NotImplementedError("Async not supported")