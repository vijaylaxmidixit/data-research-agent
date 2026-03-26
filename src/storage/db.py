"""
src/storage/db.py — SQLite Database Initialization
====================================================
Creates and manages the local SQLite database for persisting
agent research data across sessions.

Database location: ./data/agent.db (configured via DB_PATH in .env)

Tables:
    datasets  — stores discovered dataset metadata from all platforms
    searches  — logs every research query the agent runs
    reports   — tracks generated Markdown report files

Why SQLite?
    - Zero configuration — no server needed
    - Built into Python standard library
    - Perfect for local single-user agent storage
    - Portable — the entire DB is one file

Usage:
    # Initialize database (run once on first setup)
    python -c "from src.storage.db import init_db; init_db()"

    # Or run this file directly
    python src/storage/db.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import sqlite3    # Built-in Python SQLite interface — no install needed
import os
from dotenv import load_dotenv

# Load environment variables to get DB_PATH
load_dotenv()

# Database file path — read from .env, fallback to ./data/agent.db
# Important: This must be a FILE path, not a directory path
# Wrong: ./data/agent/db  ← creates a folder named "agent" with file "db" inside
# Right: ./data/agent.db  ← creates a file named "agent.db" in data/ folder
DB_PATH = os.getenv("DB_PATH", "./data/agent.db")


def init_db():
    """
    Initializes the SQLite database and creates all required tables.

    Uses CREATE TABLE IF NOT EXISTS so it's safe to run multiple times
    without destroying existing data.

    Schema:
        datasets table:
            id          — unique identifier (platform:dataset_ref)
            name        — human-readable dataset name
            platform    — source platform (kaggle/huggingface/uci/datagov/ods)
            url         — direct link to dataset
            size_mb     — file size in megabytes
            description — dataset description text
            license     — data license (CC0, MIT, etc.)
            created_at  — auto-set timestamp when record is inserted

        searches table:
            id           — auto-increment primary key
            query        — the research query that was run
            platforms    — comma-separated list of platforms searched
            result_count — total datasets found
            created_at   — timestamp

        reports table:
            id         — auto-increment primary key
            query      — research query that generated this report
            filepath   — path to the saved .md file
            created_at — timestamp
    """
    # Create data/ directory if it doesn't exist
    # dirname extracts the folder from the file path
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    # Connect to SQLite database
    # If agent.db doesn't exist, SQLite creates it automatically
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create all tables in a single transaction
    # executescript runs multiple SQL statements at once
    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS datasets (
            id          TEXT PRIMARY KEY,
            name        TEXT,
            platform    TEXT,
            url         TEXT,
            size_mb     REAL,
            description TEXT,
            license     TEXT,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS searches (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            query        TEXT,
            platforms    TEXT,
            result_count INTEGER,
            created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS reports (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            query      TEXT,
            filepath   TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Commit the transaction and close connection
    conn.commit()
    conn.close()
    print("✓ Database initialized at", DB_PATH)


# ── Direct execution ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Allow running this file directly to initialize the database:
    # python src/storage/db.py
    init_db()