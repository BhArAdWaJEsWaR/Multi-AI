"""
core/orchestrator.py
Routes a user query to the appropriate specialist agent, optionally validates
the result, and stores the exchange in vector memory.
"""

from core.agents import AGENTS
from core.memory import VectorMemory

VALID_LABELS = {"coding", "math", "writing", "general"}


class Orchestrator:
    def __init__(self, use_memory: bool = True, use_validator: bool = True):
        self.use_memory = use_memory
        self.use_validator = use_validator
        self.memory = VectorMemory() if use_memory else None

    def classify(self, query: str) -> str:
        """Ask the coordinator model to label the query's task type."""
        raw = AGENTS["coordinator"].run(query).lower().strip()

        for label in VALID_LABELS:
            if label in raw:
                return label

        return "general"  # fallback if classification is unclear

    def handle(self, query: str, verbose: bool = True) -> dict:
        # 1. Retrieve relevant context from memory
        context = ""
        if self.use_memory:
            context = self.memory.query(query)
            if verbose and context:
                print(f"[memory] Retrieved {len(context)} chars of context\n")

        # 2. Classify the task
        task_type = self.classify(query)
        if verbose:
            print(f"[coordinator] Routed to: {task_type}\n")

        # 3. Dispatch to the specialist agent
        agent = AGENTS[task_type]
        response = agent.run(query, context=context)
        if verbose:
            print(f"[{task_type}] {response}\n")

        # 4. Optional validation pass
        if self.use_validator:
            validation_input = f"Original query: {query}\n\nAgent response:\n{response}"
            validated = AGENTS["validator"].run(validation_input)
            if verbose:
                print(f"[validator] {validated}\n")

            if validated.startswith("REVISED:"):
                response = validated.split("REVISED:", 1)[1].strip()
            elif validated.startswith("VALID:"):
                pass  # keep original response
            # else: leave response unchanged if validator format unexpected

        # 5. Store the exchange in memory for future retrieval
        if self.use_memory:
            self.memory.add(f"Q: {query}\nA: {response}", metadata={"task_type": task_type})

        return {"task_type": task_type, "response": response}


if __name__ == "__main__":
    orch = Orchestrator()

    print("Multi-agent system ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        result = orch.handle(q)
        print(f"\n=== Final Answer ({result['task_type']}) ===\n{result['response']}\n")
