# Model Persistence and Memory Management Improvements

## Summary of Changes

This implementation addresses the issues of repeated model loading/unloading and inefficient memory usage by introducing a persistent model loading system with smart tier management.

## Key Features Implemented

### 1. Model Registry System (`core/agents.py`)

Added a `ModelRegistry` class that:

- **Tracks loaded models internally**: Maintains an internal set of loaded models synced with Ollama state
- **Caches agent instances**: Reuses Agent objects instead of creating new ones each time
- **Implements smart tier persistence**: If a higher-tier model is already loaded, it will be reused for lower-tier queries instead of loading a new model
- **Syncs with Ollama on startup**: Automatically syncs with actual Ollama state on initialization
- **Refreshes state periodically**: Syncs with Ollama before each query to handle external model unloads

```python
class ModelRegistry:
    def get_or_create_agent(self, domain: str, tier: str, force_tier3: bool = False) -> tuple[Agent, str]:
        # If higher-tier model is loaded, reuse it for lower-tier queries
        # Returns: (agent, actual_tier_used)
```

### 2. Topic-Aware Model Unloading (`core/orchestrator.py`)

Modified the orchestrator to:

- **Detect topic changes**: Uses the existing embedding similarity system to detect when the user changes topics
- **Unload old domain models**: When switching from coding to math (or vice versa), automatically unload models from the previous domain to free memory
- **Preserve current domain models**: Keeps models loaded for the current topic for faster subsequent queries

```python
# In classify() method - after topic detection
if (self._last_task_type is not None and 
    self._last_task_type in TIERED_MODELS and 
    self._last_task_type != task_type):
    print(f"[model_registry] Topic changed from {self._last_task_type} to {task_type}, unloading old domain models")
    _model_registry.unload_domain_models(self._last_task_type)
```

### 3. Improved Keep-Alive Settings (`config.py`)

Increased the `keep_alive` durations to reduce model reloading:

- **Low profile**: `specialist_keep_alive: "30m"` (was 10m), `validator_keep_alive: "1h"` (was 30m)
- **Mid profile**: `specialist_keep_alive: "45m"` (was 15m), `validator_keep_alive: "1h"` (was 30m)
- **High profile**: `specialist_keep_alive: "1h"` (was 30m), `validator_keep_alive: "2h"` (was 30m)

### 4. Enhanced User Feedback (`core/orchestrator.py`)

Added visual indicators to show when models are being reused vs loaded:

```python
# Shows "REUSING LOADED" if model was already in memory
# Shows "LOADING" if model needs to be loaded
status = "REUSING LOADED" if was_loaded else "LOADING"
print(f"[{task_type}/{tier} ({agent.model}) - {status}] ", end="", flush=True)
```

## How It Works

### Example Scenario:

1. **User asks a coding question (easy difficulty)**
   - System loads `qwen2.5-coder:1.5b` (tier1)
   - Model stays loaded with 30m keep_alive

2. **User asks another coding question (medium difficulty)**
   - System would normally load `qwen2.5-coder:3b` (tier2)
   - But if tier1 is still loaded, it checks if tier2 is available
   - If tier2 isn't loaded, it may reuse tier1 if tier1 can handle the task
   - Or load tier2 and keep both loaded

3. **User asks a math question (topic change)**
   - System detects topic change via embedding similarity
   - Unloads all coding models to free memory
   - Loads appropriate math model

4. **User asks another math question**
   - Reuses the already-loaded math model (fast response)

## Memory Management Benefits

1. **Reduced model loading overhead**: Models stay loaded longer, reducing startup time
2. **Smart tier reuse**: Higher-tier models can handle lower-tier queries, avoiding unnecessary model switches
3. **Topic-aware cleanup**: Automatically unloads models from unused domains
4. **Agent caching**: Reuses Agent objects instead of creating new instances
5. **Configurable persistence**: Keep-alive times can be adjusted per hardware profile

## Testing

The implementation has been tested with:

1. **Unit tests**: Model registry functionality (caching, unloading, tier selection)
2. **Integration tests**: Full orchestrator workflow with topic changes
3. **Syntax validation**: All Python files compile without errors

## Configuration

To adjust the behavior, modify these settings in `config.py`:

- `specialist_keep_alive`: How long to keep specialist models loaded
- `validator_keep_alive`: How long to keep validator models loaded
- `sim_threshold`: Controls when a topic change is detected (lower = more sensitive)

## Files Modified

1. `core/agents.py`: Added ModelRegistry class, agent caching, and improved memory prompts
2. `core/orchestrator.py`: Integrated model registry, added topic-aware unloading, and improved memory context handling
3. `config.py`: Improved keep_alive settings for better persistence

## Additional Memory System Improvements

Fixed the memory system to better retain and use contextual information:

1. **Enhanced Agent Prompts**: Updated all agent system prompts (especially general tier1/tier2/tier3) to explicitly instruct the LLM to use provided context from previous conversations
2. **Improved Context Messaging**: Changed the context message format to be more directive: "IMPORTANT: Use this context from previous conversations to answer the user's question"
3. **Memory Debug Output**: Added debug output to show what context is being retrieved and stored
4. **Hybrid Search with Recency Weighting**: Implemented time-based ranking to prioritize recent conversations over semantically similar but old ones
5. **Automatic Memory Cleanup**: System automatically removes stale incorrect responses on startup
6. **Coding Agent Preference Following**: Updated coding agents to strictly follow user preferences (like "I prefer recursion over iteration") from context
7. **Preference Statement Detection**: Added logic to detect simple preference statements (short queries containing "prefer", "like", "want", etc.) and classify them as "general" instead of letting dataset-knn override to "coding"
8. **Tiered General Agents**: Implemented tiered general agents with preference-aware prompts to properly handle user preferences and confirm they'll be remembered

The memory system now properly:
- Stores Q&A pairs in ChromaDB with embeddings and timestamps
- Retrieves relevant context using hybrid search (semantic similarity + recency weighting)
- Provides that context to agents with clear instructions to use it
- Agents are now explicitly told to remember names, preferences, and past discussions
- Coding agents follow user preferences for implementation approaches
- Preference statements are correctly classified as general conversation, not coding tasks

## Backward Compatibility

The changes are fully backward compatible:

- Existing code using `make_tiered_agent()` still works
- The model registry is used internally but doesn't change the public API
- Default behavior is improved but can be disabled if needed