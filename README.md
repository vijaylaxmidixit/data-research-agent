# dataset-research-agent 🔍

A local AI agent that autonomously discovers, retrieves, and analyzes public datasets useful for data engineering practice — running 100% on your machine using Qwen3 8B via Ollama.

## What it does

1. **Discovers** datasets across 5 open data platforms
2. **Retrieves** metadata, schemas, sizes, and descriptions
3. **Analyzes** and ranks datasets by DE relevance
4. **Generates** a structured Markdown report saved locally

## Data Platforms

| Platform | Method | Auth |
|----------|--------|------|
| Kaggle | REST API | Free account |
| HuggingFace Hub | REST API | Free account |
| UCI ML Repository | Curated list | None |
| Data.gov | CKAN API | None |
| OpenDataSoft | Explore API | None |

## Tech Stack

- **LLM**: Qwen3 8B via [Ollama](https://ollama.ai) (local, no cloud)
- **Agent**: LangChain ReAct pattern
- **Embeddings**: nomic-embed-text via Ollama
- **Vector Store**: ChromaDB (local)
- **Database**: SQLite
- **CLI**: Typer + Rich

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- 16GB RAM minimum
- Python 3.11+
- [Ollama](https://ollama.ai) installed

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/vijaylaxmidixit/data-research-agent.git
cd data-research-agent
```

### 2. Install Ollama and pull models

```bash
brew install ollama
ollama serve &
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

### 3. Create virtual environment and install dependencies

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` with your API keys:

```env
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
HUGGINGFACE_TOKEN=hf_your_token
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
EMBED_MODEL=nomic-embed-text
DB_PATH=./data/agent.db
CHROMA_PATH=./data/chroma
REPORTS_PATH=./reports
```

Get your free API keys:
- **Kaggle**: kaggle.com → Settings → API → Create New Token
- **HuggingFace**: huggingface.co → Settings → Access Tokens → New Token (Read)

### 5. Initialize the database

```bash
mkdir -p data/chroma reports logs
python -c "from src.storage.db import init_db; init_db()"
```

## Usage

```bash
# Basic research query
python main.py --query "ETL pipeline datasets"

# With custom result limit
python main.py --query "time series sensor data" --max-results 10

# Short flag
python main.py -q "retail transactions" -n 5
```

## Project Structure

```
data-research-agent/
├── main.py                     # CLI entry point
├── .env                        # API keys (git-ignored)
├── .env.example                # Template for .env
├── requirements.txt            # Python dependencies
├── src/
│   ├── agent/
│   │   └── research_agent.py   # ReAct agent + prompt
│   ├── llm/
│   │   └── ollama_client.py    # LLM + embeddings factory
│   ├── tools/
│   │   ├── kaggle_tool.py      # Kaggle REST API
│   │   ├── huggingface_tool.py # HuggingFace Hub API
│   │   ├── uci_tool.py         # UCI ML Repository (curated)
│   │   ├── datagov_tool.py     # Data.gov CKAN API
│   │   ├── ods_tool.py         # OpenDataSoft API
│   │   ├── analyzer_tool.py    # Dataset relevance scorer
│   │   └── report_tool.py      # Markdown report generator
│   └── storage/
│       ├── db.py               # SQLite schema + init
│       └── vector_store.py     # ChromaDB semantic search
├── data/                       # git-ignored
│   ├── agent.db                # SQLite database
│   └── chroma/                 # ChromaDB vector store
├── reports/                    # Generated Markdown reports
└── tests/                      # Unit tests
```

## Security

- All LLM inference runs locally — no data sent to cloud
- Agent has no filesystem access beyond project directory
- API keys stored in `.env` (never committed to Git)
- ChromaDB telemetry disabled

## RAM Usage (M4 16GB)

| Component | RAM |
|-----------|-----|
| macOS | ~4.0 GB |
| Qwen3 8B | ~5.2 GB |
| nomic-embed-text | ~0.3 GB |
| Python Agent | ~0.8 GB |
| ChromaDB | ~0.3 GB |
| VS Code | ~1.2 GB |
| **Total** | **~11.8 GB** |

