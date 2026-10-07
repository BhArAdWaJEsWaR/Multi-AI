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

For image requests, the vision model is an extraction stage only: it performs OCR
and describes relevant visual content, then the extracted content is routed
through the normal coding / math / writing / general classifier and difficulty
tiers. The selected specialist answers the request; the vision model does not
answer or determine the final category.

## Conversation behavior

The active chat keeps the last 24 user/assistant turns in temporary process
memory and sends them to every specialist, including when routing switches
domains. This gives follow-up messages their normal chat meaning without
persisting ordinary questions and answers. The temporary chat disappears when
the process exits.

Only explicit preference updates are written to ChromaDB for persistence across
restarts. Existing older Q&A records may remain in the ChromaDB directory, but
new ordinary exchanges are no longer added there.

Models are not forcibly unloaded when the topic changes, and ordinary
conversation uses a fast path (no difficulty-estimator or validator call).
Pass `use_validator=True` to `Orchestrator` when response validation is
preferred over latency.

## Notes

- First run downloads the `all-MiniLM-L6-v2` embedding model (~80MB) for ChromaDB.
- ChromaDB persists to `./chroma_store` by default.
- To add a new agent: add an entry to `AGENTS` in `core/agents.py`, add its label
  to `VALID_LABELS` in `core/orchestrator.py`, and update the coordinator's
  system prompt to mention the new category.
