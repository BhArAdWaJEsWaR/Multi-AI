# Multi-Agent Local AI System

## Setup

1. Make sure Ollama is running and these models are pulled:
   - qwen3:4b (coordinator)
   - qwen2.5-coder:7b (coding agent)
   - deepseek-r1:7b (math agent)
   - gemma3:4b (writing agent)
   - llama3.2:latest (general agent + validator)

2. Install Python dependencies:
   ```
   pip install -r requirements.txt --break-system-packages
   ```

3. Run:
   ```
   python -m core.orchestrator
   ```

## Structure

- `core/agents.py` — agent definitions (model + system prompt + Ollama call)
- `core/memory.py` — ChromaDB-based shared vector memory (RAG context)
- `core/orchestrator.py` — coordinator logic: classify -> route -> validate -> store

## How it works

1. Coordinator (qwen3:4b) classifies the query into coding / math / writing / general.
2. Relevant memory is retrieved from ChromaDB and passed as context.
3. The specialist agent generates a response.
4. A validator agent (llama3.2) checks the response for errors/inconsistencies.
5. The exchange is stored back into vector memory for future queries.

## Notes

- First run downloads the `all-MiniLM-L6-v2` embedding model (~80MB) for ChromaDB.
- ChromaDB persists to `./chroma_store` by default.
- To add a new agent: add an entry to `AGENTS` in `core/agents.py`, add its label
  to `VALID_LABELS` in `core/orchestrator.py`, and update the coordinator's
  system prompt to mention the new category.
