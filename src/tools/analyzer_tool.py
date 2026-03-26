"""
src/tools/analyzer_tool.py — Dataset Relevance Analyzer Tool
=============================================================
Analyzes and ranks a list of datasets by relevance to data engineering use cases.

This tool is called by the agent AFTER collecting results from platform tools.
It scores each dataset using a simple heuristic scoring system and returns
the datasets sorted by relevance score (highest first).

Scoring criteria:
    Size score (0-3):   Based on number of instances/rows
    Popularity (0-3):   Based on votes (Kaggle) or downloads (HuggingFace)
    Feature score (0-1): Based on number of features/columns
    DE keyword (0-1):   Whether description contains DE-relevant terms

Key design decision:
    - Accepts both valid JSON and Python dict strings (single quotes)
      because the LLM sometimes formats output with single quotes
      instead of valid JSON double quotes
    - Uses ast.literal_eval as fallback for Python-style dicts
"""

from langchain.tools import BaseTool
from pydantic import BaseModel
import json
import ast    # Used to parse Python-style dict strings from LLM


class AnalyzerInput(BaseModel):
    """Validates LLM input — expects a JSON array string of datasets."""
    datasets_json: str


class DatasetAnalyzerTool(BaseTool):
    """
    LangChain tool that scores and ranks datasets by DE relevance.

    Called by the agent after collecting results from multiple platforms.
    Combines all platform results and returns a ranked unified list.
    """

    name: str = "dataset_analyzer"
    description: str = """Analyze and rank a list of datasets by relevance and quality.
    Input: a list of datasets as a JSON string or Python list string.
    Output: Ranked and annotated JSON list."""
    args_schema: type = AnalyzerInput

    def _run(self, datasets_json: str, **kwargs) -> str:
        """
        Parses, scores, and ranks a list of datasets.

        Input parsing strategy (in order):
            1. Try standard JSON.loads() — handles valid JSON
            2. Try ast.literal_eval() — handles Python single-quote dicts
            3. Try replacing ' with " — handles mixed quote formats

        Scoring breakdown:
            Size score:
                > 100,000 instances → +3 points (large dataset)
                > 10,000 instances  → +2 points (medium dataset)
                > 1,000 instances   → +1 point  (small dataset)

            Popularity score:
                > 100 votes         → +3 points (Kaggle)
                > 1,000 downloads   → +3 points (HuggingFace)

            Feature score:
                > 10 features/cols  → +1 point

            DE keyword bonus:
                Contains DE term    → +1 point (max 1 per dataset)
                DE terms: etl, pipeline, warehouse, streaming, batch,
                          transaction, time series, sensor, log, retail

        Args:
            datasets_json: JSON or Python string containing list of dataset dicts

        Returns:
            str: JSON array sorted by relevance_score descending
        """
        # ── Input Parsing ─────────────────────────────────────────────────────
        try:
            # Attempt 1: Standard JSON parsing (most common case)
            datasets = json.loads(datasets_json)
        except (json.JSONDecodeError, ValueError):
            try:
                # Attempt 2: Python dict format (LLM uses single quotes)
                # e.g. "[{'name': 'iris', 'url': 'http://...'}]"
                datasets = ast.literal_eval(datasets_json)
            except Exception:
                try:
                    # Attempt 3: Replace single quotes with double quotes
                    fixed = datasets_json.replace("'", '"')
                    datasets = json.loads(fixed)
                except Exception as e:
                    return json.dumps({"error": f"Could not parse input: {str(e)}"})

        # Ensure we have a list (single dataset may be passed as a dict)
        if not isinstance(datasets, list):
            datasets = [datasets]

        # ── Scoring Loop ──────────────────────────────────────────────────────
        for ds in datasets:
            score = 0

            # Size score: prefer larger datasets (more DE practice value)
            # Checks multiple field names since different platforms use different keys
            instances = ds.get("num_instances") or ds.get("size") or ds.get("totalBytes", 0)
            try:
                instances = int(instances)
            except (TypeError, ValueError):
                instances = 0

            if instances > 100000:
                score += 3    # Large: >100K rows — good for big data pipelines
            elif instances > 10000:
                score += 2    # Medium: >10K rows — good for ETL practice
            elif instances > 1000:
                score += 1    # Small: >1K rows — basic practice

            # Popularity score: community validation of dataset quality
            if ds.get("votes", 0) > 100:
                score += 3    # Kaggle: highly voted datasets
            if ds.get("downloads", 0) > 1000:
                score += 3    # HuggingFace: frequently downloaded

            # Feature score: more columns = richer pipeline practice
            features = ds.get("num_features") or ds.get("features", 0)
            try:
                features = int(features)
            except (TypeError, ValueError):
                features = 0
            if features > 10:
                score += 1

            # DE keyword bonus: reward datasets explicitly useful for DE
            desc = (ds.get("description", "") + ds.get("name", "")).lower()
            de_keywords = [
                "etl", "pipeline", "warehouse", "streaming", "batch",
                "transaction", "time series", "sensor", "log", "retail"
            ]
            for kw in de_keywords:
                if kw in desc:
                    score += 1
                    break   # Max 1 point from keyword bonus

            # Attach score to dataset object
            ds["relevance_score"] = score

        # Sort by relevance_score descending (highest first)
        ranked = sorted(datasets, key=lambda x: x.get("relevance_score", 0), reverse=True)
        return json.dumps(ranked, indent=2)

    async def _arun(self, *args, **kwargs):
        """Async version — not implemented."""
        raise NotImplementedError("Async not supported")