# LLM RAG Challenge

Coursework for LLM Intro course (AITH). This project implements an AI agent capable of gathering research papers and performing Retrieval-Augmented Generation (RAG) to answer questions based on the collected knowledge.

## Setup

This project uses `uv` for dependency management.

### 1. Install Dependencies
Ensure you have `uv` installed, then sync the dependencies:

```bash
uv sync
```

### 2. Configure Environment
Create a `.env` file from the example:

```bash
cp env.example .env
```

Edit the `.env` file to add your API keys.

**Note:**
- `OPENAI_API_KEY`: Required for the LLM (if using OpenAI models).
- `OPENAI_API_BASE`: Optional, for compatible servers.
- `LLM_MODEL_NAME`: Name of the model to use (default: `gpt-4o-mini`).
- Embeddings are configured to use a local HuggingFace model (`BAAI/bge-small-en-v1.5`) by default, so no API key is needed for indexing.

## Usage

The project provides a CLI entry point `main.py` with several commands:

### Gather Data
Download relevant research papers from ArXiv (Deep Learning, LLMs, Agents, etc.) into `data/raw`:

```bash
uv run python main.py --gather
```

### Build Index
Process the documents in `data/raw` and build a vector search index, saving it to `data/processed`:

```bash
uv run python main.py --index
```

### Query System
Ask a question to retrieve relevant information from the indexed documents. Currently configured to return the top 3 most relevant text chunks:

```bash
uv run python main.py --query "What are the latest trends in Reinforcement Learning?"
```
