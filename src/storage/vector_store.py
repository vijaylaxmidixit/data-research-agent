"""
src/storage/vector_store.py — ChromaDB Vector Store
=====================================================
Provides semantic search over discovered datasets using vector embeddings.

What is a vector store?
    Instead of exact keyword matching (like SQL LIKE), a vector store converts
    text into numerical vectors (embeddings) that capture semantic meaning.
    This allows queries like "find datasets similar to time series sensor data"
    to return relevant results even if the exact words don't match.

How it works:
    1. Dataset description is converted to a 768-dimensional vector
       using nomic-embed-text model via Ollama
    2. Vector is stored in ChromaDB (persistent on disk)
    3. At query time, the query is also converted to a vector
    4. ChromaDB finds the most similar dataset vectors
    5. Returns the closest matching datasets

Storage location: ./data/chroma/ (configured via CHROMA_PATH in .env)

Why ChromaDB?
    - Runs embedded (no separate server needed)
    - Persistent storage — survives between sessions
    - Fast similarity search on M4 (uses Apple Accelerate framework)
    - Simple Python API

Usage example:
    store = DatasetVectorStore()

    # Add a dataset to the store
    store.add_dataset(
        dataset_id="kaggle:nyc-taxi",
        text="NYC Taxi trip records for pipeline practice",
        metadata={"platform": "kaggle", "url": "https://kaggle.com/..."}
    )

    # Search semantically
    results = store.search("urban transport time series")
    # Returns NYC Taxi even if "urban" or "transport" weren't in the original text
"""

# ── ChromaDB — embedded vector database ──────────────────────────────────────
import chromadb
from chromadb.config import Settings

# ── Local embeddings model via Ollama ─────────────────────────────────────────
from src.llm.ollama_client import get_embeddings

# ── Standard library ──────────────────────────────────────────────────────────
import os


class DatasetVectorStore:
    """
    Manages semantic search over discovered datasets using ChromaDB.

    Attributes:
        client     — ChromaDB persistent client (reads/writes to disk)
        collection — Named collection storing dataset embeddings
        embeddings — nomic-embed-text model for generating vectors
    """

    def __init__(self):
        """
        Initializes ChromaDB client and connects to the datasets collection.

        ChromaDB settings:
            - anonymized_telemetry=False: disables usage data sent to ChromaDB
            - PersistentClient: saves data to disk (survives restarts)
        """
        # Get storage path from environment, default to ./data/chroma/
        chroma_path = os.getenv("CHROMA_PATH", "./data/chroma")

        # Create ChromaDB client with persistent local storage
        # anonymized_telemetry=False ensures no data leaves your machine
        self.client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create the "datasets" collection
        # If collection already exists, connects to it (preserves existing data)
        # If it doesn't exist, creates a new empty collection
        self.collection = self.client.get_or_create_collection("datasets")

        # Initialize the embeddings model (nomic-embed-text via Ollama)
        self.embeddings = get_embeddings()

    def add_dataset(self, dataset_id: str, text: str, metadata: dict):
        """
        Adds or updates a dataset in the vector store.

        Uses upsert (update + insert) so re-running the agent doesn't
        create duplicate entries for the same dataset.

        Args:
            dataset_id: Unique identifier (e.g. "kaggle:nyc-taxi-trips")
            text: Text to embed (dataset name + description combined)
            metadata: Additional fields to store (platform, url, size, etc.)
        """
        # Convert text to 768-dimensional embedding vector
        embedding = self.embeddings.embed_query(text)

        # Store in ChromaDB
        # upsert = update if exists, insert if new
        self.collection.upsert(
            ids=[dataset_id],         # Unique identifier
            embeddings=[embedding],   # 768-dim vector representation
            documents=[text],         # Original text (for display)
            metadatas=[metadata]      # Additional searchable fields
        )

    def search(self, query: str, n_results: int = 5):
        """
        Finds the most semantically similar datasets to a query.

        Process:
            1. Convert query text to embedding vector
            2. ChromaDB computes cosine similarity against all stored vectors
            3. Returns top n_results most similar datasets

        Args:
            query: Natural language search query
                   e.g. "urban transport time series for streaming pipeline"
            n_results: Number of similar datasets to return (default: 5)

        Returns:
            dict: ChromaDB query result with keys:
                  'documents': list of original text
                  'metadatas': list of metadata dicts
                  'distances': similarity scores (lower = more similar)
        """
        # Convert query to embedding for similarity comparison
        q_embed = self.embeddings.embed_query(query)

        # Find most similar datasets in the collection
        return self.collection.query(
            query_embeddings=[q_embed],
            n_results=n_results
        )