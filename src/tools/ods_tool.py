"""
src/tools/ods_tool.py — OpenDataSoft Dataset Search Tool
=========================================================
Searches the OpenDataSoft federated open data network via their Explore API.

Platform: data.opendatasoft.com
Auth: None required for public datasets
API: https://data.opendatasoft.com/api/explore/v2.1/catalog/datasets
Docs: https://help.opendatasoft.com/apis/ods-explore-v2/

OpenDataSoft aggregates 1,000+ open data portals from cities, governments,
and organizations worldwide — providing rich geospatial, temporal, and
transport datasets ideal for data engineering practice.

Key design decision:
    - Query simplification: ODS 'where' clause returns empty results
      for complex multi-word queries
    - Uses first word of query as the search keyword
"""

from langchain.tools import BaseTool
from pydantic import BaseModel
import requests
import json


class ODSSearchInput(BaseModel):
    """Validates LLM input before _run() executes."""
    query: str
    max_results: int = 10


class OpenDataSoftTool(BaseTool):
    """
    LangChain tool for searching OpenDataSoft federated data portals.

    Returns datasets with publisher info, record counts, and direct URLs.
    Covers 1,000+ public data portals in a single search.
    """

    name: str = "opendatasoft_search"
    description: str = """Search OpenDataSoft for public datasets across 1000+ portals.
    Input: JSON with 'query' and optional 'max_results'.
    Output: JSON list with name, publisher, records, url."""
    args_schema: type = ODSSearchInput

    def _run(self, query: str, max_results: int = 10, **kwargs) -> str:
        """
        Searches OpenDataSoft catalog API.

        Steps:
            1. Simplify query to first keyword (ODS where clause is strict)
            2. Call ODS Explore v2.1 catalog API
            3. Extract dataset metadata from nested response structure
            4. Return normalized JSON array

        ODS response structure:
            {
                "results": [
                    {
                        "dataset_id": "...",
                        "dataset": {
                            "metas": {
                                "default": {
                                    "title": "...",
                                    "publisher": "...",
                                    "keyword": [...]
                                }
                            }
                        }
                    }
                ]
            }

        Returns:
            str: JSON array of dataset objects, or JSON error object
        """
        # Simplify multi-word query to single keyword
        # ODS 'where' clause is strict — single keywords work best
        simple_query = query.split()[0] if len(query.split()) > 2 else query
        query = simple_query

        try:
            resp = requests.get(
                "https://data.opendatasoft.com/api/explore/v2.1/catalog/datasets",
                params={
                    "where": f"'{query}'",  # ODS full-text search syntax
                    "limit": max_results
                },
                timeout=10,
            )

            # Handle API errors
            if resp.status_code != 200 or not resp.text.strip():
                return json.dumps({"error": f"ODS API error: {resp.status_code}"})

            results = resp.json().get("results", [])

            if not results:
                return json.dumps({"message": "No ODS datasets found", "query": query})

            datasets = []
            for ds in results[:max_results]:
                # Navigate the nested ODS response structure
                meta = ds.get("dataset", {}).get("metas", {}).get("default", {})
                datasets.append({
                    "name": meta.get("title", ""),
                    "publisher": meta.get("publisher", ""),    # Organization that published
                    "records": ds.get("dataset", {}).get("size", 0),  # Number of records
                    "url": f"https://data.opendatasoft.com/explore/dataset/{ds.get('dataset_id', '')}",
                    "keyword": meta.get("keyword", [])[:5],   # Topic tags
                })
            return json.dumps(datasets, indent=2)

        except Exception as e:
            return json.dumps({"error": f"ODS tool error: {str(e)}"})

    async def _arun(self, *args, **kwargs):
        """Async version — not implemented."""
        raise NotImplementedError("Async not supported")