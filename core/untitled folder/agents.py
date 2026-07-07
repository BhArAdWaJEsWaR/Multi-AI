"""
core/agents.py
Defines specialized agents, each wrapping a local Ollama model with a system prompt.
"""

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"


class Agent:
    def __init__(self, name: str, model: str, system_prompt: str, temperature: float = 0.3):
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature

    def run(self, user_input: str, context: str = "") -> str:
        """Send a query to this agent's model via Ollama and return the response text."""
        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            messages.append({
                "role": "system",
                "content": f"Relevant context retrieved from memory:\n{context}"
            })

        messages.append({"role": "user", "content": user_input})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"].strip()
        except requests.exceptions.RequestException as e:
            return f"[ERROR calling {self.name} ({self.model}): {e}]"


# ---- Agent registry ----------------------------------------------------

AGENTS = {
    "coordinator": Agent(
        name="coordinator",
        model="qwen3:4b",
        system_prompt=(
            "You are the coordinator of a multi-agent AI system. "
            "Given a user query, classify it into exactly ONE of the following task types: "
            "'coding', 'math', 'writing', 'general'. "
            "Respond with ONLY the single word label, nothing else."
        ),
        temperature=0.0,
    ),
    "coding": Agent(
        name="coding",
        model="qwen2.5-coder:7b",
        system_prompt=(
            "You are a coding specialist agent. Write clean, correct, well-commented code. "
            "Explain briefly what the code does after the code block."
        ),
        temperature=0.2,
    ),
    "math": Agent(
        name="math",
        model="deepseek-r1:7b",
        system_prompt=(
            "You are a mathematical reasoning specialist agent. "
            "Solve problems step by step, showing your reasoning clearly, "
            "and state the final answer explicitly at the end."
        ),
        temperature=0.2,
    ),
    "writing": Agent(
        name="writing",
        model="gemma3:4b",
        system_prompt=(
            "You are a writing specialist agent. Produce clear, well-structured, "
            "polished prose appropriate to the requested tone and format."
        ),
        temperature=0.6,
    ),
    "general": Agent(
        name="general",
        model="llama3.2:latest",
        system_prompt=(
            "You are a general-purpose assistant agent. Answer the query directly and concisely."
        ),
        temperature=0.4,
    ),
    "validator": Agent(
        name="validator",
        model="deepseek-r1:7b",
        system_prompt=(
            "You are a critical validation agent. Your job is NOT to summarize or restate "
            "the response — it is to actively find problems with it. Check for:\n"
            "1. Factual or logical errors.\n"
            "2. Suboptimal design choices (e.g. unnecessary memory use, wrong complexity, "
            "non-idiomatic or non-canonical approaches) when a clearly better alternative exists.\n"
            "3. Missing edge cases or unstated assumptions.\n\n"
            "If the response is correct AND there is no meaningfully better approach, respond with "
            "'VALID: <original response unchanged>'.\n"
            "If you find a real issue (correctness OR efficiency OR design), respond with "
            "'REVISED: <corrected/improved response, including the issue you found and why "
            "the new version is better>'.\n"
            "Do not pad your response with restatements of what the agent already said. "
            "Only flag genuine, specific issues."
        ),
        temperature=0.1,
    ),
}