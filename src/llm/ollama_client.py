"""
src/llm/ollama_client.py — Local LLM & Embeddings Factory
===========================================================
Provides factory functions for creating:
    - LLM: Qwen3 8B running locally via Ollama (used by the agent for reasoning)
    - Embeddings: nomic-embed-text via Ollama (used by ChromaDB for semantic search)

All inference runs 100% locally on your M4 MacBook Air.
No data is sent to any cloud service.

Models required (pull before first run):
    ollama pull qwen3:8b
    ollama pull nomic-embed-text
"""

#  LangChain Ollama integration 
from langchain_ollama import OllamaLLM, OllamaEmbeddings

#  Standard library 
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


def get_llm() -> OllamaLLM:
    """
    Returns a configured Qwen3 8B LLM instance via Ollama.

    Configuration:
        model       → Read from OLLAMA_MODEL in .env (default: qwen3:8b)
        base_url    → Ollama server URL (default: http://localhost:11434)
        temperature → 0.1 for near-deterministic, factual responses
        num_ctx     → 8192 token context window (safe for M4 16GB RAM)

    Why Qwen3 8B?
        - Strong reasoning and tool-use capabilities
        - 32K max context (we use 8K for RAM safety)
        - ~5.2GB RAM usage — fits in 16GB alongside embeddings model
        - ~20 tokens/second on M4 with Metal GPU acceleration

    Returns:
        OllamaLLM: Local LLM instance — no cloud calls ever made
    """
    return OllamaLLM(
        model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),          # Primary reasoning model
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),  # Local Ollama server
        temperature=0.1,    # Low temperature = more consistent, factual outputs
        num_ctx=8192,       # Context window — increase to 16384 if you have headroom
    )


def get_embeddings() -> OllamaEmbeddings:
    """
    Returns a configured nomic-embed-text embeddings model via Ollama.

    Used by ChromaDB (vector_store.py) to:
        - Convert dataset descriptions into vector embeddings
        - Enable semantic similarity search across discovered datasets
        - Allow queries like "find datasets similar to NYC taxi trips"

    Why nomic-embed-text?
        - Only ~274MB RAM — negligible overhead
        - 768-dimensional vectors — good quality for semantic search
        - 8192 token context window — handles long dataset descriptions
        - Runs alongside Qwen3 8B without memory pressure

    Returns:
        OllamaEmbeddings: Local embeddings model
    """
    return OllamaEmbeddings(
        model="nomic-embed-text",                                          # Embeddings model
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),   # Local Ollama server
    )