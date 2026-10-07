"""
core/agents.py
Agent definitions driven by config.py — swap hardware profile to change models.

Supports:
- Tiered sub-agents (tier1: small/fast → tier2: medium → tier3: large/accurate)
- Vision agent (image/screenshot analysis via Ollama /api/chat with images)
"""

import json
import base64
import requests
import time
from config import (
    ACTIVE_MODELS,
    ACTIVE_BEHAVIOUR,
    TIERED_MODELS,
    DIFFICULTY_TIER,
    VISION_MODEL,
)

OLLAMA_URL = "http://localhost:11434/api/chat"


def _encode_image(image_path: str, max_dim: int = 1024) -> str:
    """Read an image from disk, auto-downscaling high-res images to max_dim (default 1024)
    to prevent Apple Silicon Metal out-of-memory errors in Ollama vision transformers.
    Returns base64-encoded string for Ollama."""
    try:
        from PIL import Image
        import io
        with Image.open(image_path) as im:
            if max(im.size) > max_dim:
                im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            if im.mode in ("RGBA", "P"):
                im = im.convert("RGB")
            im.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")


class Agent:
    def __init__(
        self,
        name: str,
        model: str,
        system_prompt: str,
        temperature: float = 0.3,
        keep_alive: str = "10m",
    ):
        self.name          = name
        self.model         = model
        self.system_prompt = system_prompt
        self.temperature   = temperature
        self.keep_alive    = keep_alive

    def run(
        self,
        user_input: str,
        context: str = "",
        stream: bool = True,
        image_path: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        messages = [{"role": "system", "content": self.system_prompt}]

        if context:
            messages.append({
                "role": "system",
                "content": f"CONTEXT FROM PREVIOUS CONVERSATIONS (READ CAREFULLY):\n{context}\n\nINSTRUCTION: You MUST use the above context to answer. If the context contains someone's name (like 'my name is X'), remember it. If asked 'what is my name' and context says 'my name is Bharadwaj', you MUST answer 'Your name is Bharadwaj'. CRITICAL: If the context mentions user preferences like 'I prefer recursion over iteration', you MUST follow that preference in your solution. For coding tasks, if user prefers recursion, provide recursive solutions only.",
            })

        # Keep the current chat as real role-separated messages. This is more
        # reliable than asking a model to reconstruct a conversation from a
        # semantically retrieved memory snippet.
        if history:
            messages.extend(
                {"role": item["role"], "content": item["content"]}
                for item in history
                if item.get("role") in {"user", "assistant"} and item.get("content")
            )

        user_msg = {"role": "user", "content": user_input}
        if image_path:
            user_msg["images"] = [_encode_image(image_path)]
        messages.append(user_msg)

        payload = {
            "model":      self.model,
            "messages":   messages,
            "stream":     stream,
            "keep_alive": self.keep_alive,
            "options":    {"temperature": self.temperature},
        }

        try:
            resp = requests.post(OLLAMA_URL, json=payload, stream=stream, timeout=300)
            resp.raise_for_status()

            if not stream:
                return resp.json()["message"]["content"].strip()

            full_response = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                print(token, end="", flush=True)
                full_response += token
                if chunk.get("done"):
                    break
            print()
            return full_response.strip()

        except requests.exceptions.RequestException as e:
            return f"[ERROR calling {self.name} ({self.model}): {e}]"


# ── Tiered agent factory ──────────────────────────────────────────────────────

_DOMAIN_SYSTEM_PROMPTS = {
    "coding": {
        "tier1": (
            "You are a fast, lightweight coding assistant. Handle simple tasks: "
            "syntax fixes, one-liners, basic scripts, short functions. "
            "Write concise Python code blocks. Keep explanations brief. "
            "CRITICAL: If the context mentions user preferences (like 'I prefer recursion over iteration' "
            "or specific coding styles), you MUST follow those preferences in your solutions. "
            "For example, if context says 'I prefer recursion over iteration', you MUST provide recursive solutions, "
            "not iterative ones. This is non-negotiable - user preferences override default approaches."
        ),
        "tier2": (
            "You are a mid-size coding specialist. Handle algorithms, data structures, "
            "medium-complexity programs, debugging, and refactoring. "
            "Write clean, correct, well-commented code in Python code blocks. "
            "CRITICAL: Always follow user preferences from context - if they prefer recursion over iteration, "
            "use recursive solutions ONLY. Do not provide iterative alternatives. If they prefer specific styles "
            "or approaches, use those exclusively. User preferences are more important than conventional approaches."
        ),
        "tier3": (
            "You are an expert coding specialist. Handle complex systems, architecture "
            "design, advanced algorithms, multi-file projects, and tricky edge cases. "
            "Write production-quality, well-commented code. Always use Python code blocks. "
            "After the code block, explain the approach and any important edge cases. "
            "CRITICAL: You must strictly follow user preferences from context. If they prefer recursion "
            "over iteration, ALL solutions must be recursive, no exceptions. If they have other preferences, "
            "honor them completely. User preferences are the highest priority - even if a recursive solution "
            "is less efficient, you must still provide it if that's what the user prefers."
        ),
    },
    "math": {
        "tier1": (
            "You are a fast math assistant. Handle arithmetic, basic algebra, fractions, "
            "simple word problems. Give short, direct answers with minimal steps."
        ),
        "tier2": (
            "You are a mid-size math specialist. Handle calculus, linear algebra, probability, "
            "statistics, and moderate proofs. Show clear step-by-step reasoning."
        ),
        "tier3": (
            "You are an expert mathematical reasoning specialist. Handle advanced proofs, "
            "complex multi-step problems, number theory, analysis, topology, and "
            "mathematical modelling. Solve problems step by step, showing your reasoning "
            "clearly. State the final answer explicitly at the end."
        ),
    },
    "writing": {
        "tier1": (
            "You are a fast writing assistant. Draft short emails, messages, bullet points, "
            "quick summaries. Keep it concise and direct."
        ),
        "tier2": (
            "You are a writing specialist. Produce clear, well-structured essays, reports, "
            "cover letters, and articles. Pay attention to tone and organization."
        ),
        "tier3": (
            "You are an expert writing specialist. Produce polished, long-form prose: "
            "research reports, creative writing, detailed documentation, persuasive "
            "essays. Focus on flow, voice, nuance, and rhetorical impact."
        ),
    },
    "general": {
        "tier1": (
            "You are a fast general-purpose assistant. Answer simple facts, yes/no questions, "
            "definitions. Be concise. IMPORTANT: Pay close attention to any context provided from "
            "previous conversations - if it contains information about the user (like their name, "
            "preferences, or past discussions), use that information in your response. When someone "
            "states a preference (like 'I prefer recursion over iteration'), acknowledge it and "
            "confirm you'll remember it for future coding tasks."
        ),
        "tier2": (
            "You are a mid-size general-purpose assistant. Explain concepts, discuss topics, "
            "give balanced opinions. Answer clearly with moderate detail. IMPORTANT: Always use "
            "context from previous conversations to maintain continuity - remember names, preferences, "
            "and past discussions mentioned in the context. When users state preferences, acknowledge "
            "them and confirm you'll follow them in relevant future tasks."
        ),
        "tier3": (
            "You are a thoughtful general-purpose assistant. Engage in deep discussion, "
            "nuanced analysis, and complex multi-part questions. Provide thorough, "
            "well-reasoned answers. CRITICAL: You must use context from previous conversations "
            "to maintain conversation continuity. Remember details like names, preferences, and "
            "past discussions, and reference them appropriately in your responses. When users "
            "express preferences, acknowledge them clearly and confirm you'll honor them in "
            "relevant future interactions."
        ),
    },
}

_TIER_TEMPERATURES = {"tier1": 0.2, "tier2": 0.3, "tier3": 0.2}


def make_tiered_agent(domain: str, tier: str) -> Agent:
    """Create a tiered sub-agent for a given domain and tier."""
    model = TIERED_MODELS[domain][tier]
    system_prompt = _DOMAIN_SYSTEM_PROMPTS[domain][tier]
    temp = _TIER_TEMPERATURES[tier]
    ka = ACTIVE_BEHAVIOUR["specialist_keep_alive"]
    return Agent(name=f"{domain}_{tier}", model=model,
                 system_prompt=system_prompt, temperature=temp, keep_alive=ka)

def make_tiered_general_agent(tier: str) -> Agent:
    """Create a tiered general agent for handling preferences and general conversation."""
    model = TIERED_MODELS["general"][tier]
    system_prompt = _DOMAIN_SYSTEM_PROMPTS["general"][tier]
    temp = _TIER_TEMPERATURES[tier]
    ka = ACTIVE_BEHAVIOUR["specialist_keep_alive"]
    return Agent(name=f"general_{tier}", model=model,
                 system_prompt=system_prompt, temperature=temp, keep_alive=ka)


def get_tier_for_difficulty(difficulty: str) -> str:
    """Map 'easy'/'medium'/'hard' → 'tier1'/'tier2'/'tier3'."""
    return DIFFICULTY_TIER.get(difficulty, "tier2")


# ── Model registry for persistent loading ──────────────────────────────────────

class ModelRegistry:
    """Registry to track loaded models and enable agent reuse."""
    
    def __init__(self):
        self._loaded_models: set[str] = set()
        self._agent_cache: dict[str, Agent] = {}  # key: f"{domain}_{tier}"
        self._ollama_url = "http://localhost:11434"
        self._model_load_times: dict[str, float] = {}  # Track when models were loaded
        self._sync_with_ollama()  # Sync with actual Ollama state on init
    
    def _sync_with_ollama(self) -> None:
        """Sync internal tracking with actual Ollama state."""
        try:
            import requests
            import time
            resp = requests.get(f"{self._ollama_url}/api/ps", timeout=5)
            actual_loaded = {m["name"] for m in resp.json().get("models", [])}
            self._loaded_models = actual_loaded
            for model in actual_loaded:
                self._model_load_times[model] = time.time()
        except Exception:
            pass  # If sync fails, start empty
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is currently loaded in Ollama (using internal tracking)."""
        return model_name in self._loaded_models
    
    def get_or_create_agent(self, domain: str, tier: str, force_tier3: bool = False) -> tuple[Agent, str]:
        """
        Get or create an agent for the given domain and tier.
        Implements smart tier persistence: if a higher-tier model is already loaded,
        reuse it for lower-tier queries instead of loading a new model.
        
        Returns: (agent, actual_tier_used)
        """
        cache_key = f"{domain}_{tier}"
        
        # Check if we have a cached agent for this exact tier
        if cache_key in self._agent_cache:
            return self._agent_cache[cache_key], tier
        
        # Check for smart tier persistence - if higher tier is loaded, reuse it
        if not force_tier3:
            target_model = TIERED_MODELS[domain][tier]
            
            # Check if target model is in our tracked loaded models
            if target_model in self._loaded_models:
                agent = make_tiered_agent(domain, tier)
                self._agent_cache[cache_key] = agent
                return agent, tier
            
            # Check if a higher-tier model is already loaded and reuse it
            tier_order = ["tier3", "tier2", "tier1"]
            current_tier_idx = tier_order.index(tier)
            
            for higher_tier in tier_order[:current_tier_idx]:  # Check tiers above current
                higher_model = TIERED_MODELS[domain][higher_tier]
                if higher_model in self._loaded_models:
                    # Reuse the higher-tier model
                    higher_cache_key = f"{domain}_{higher_tier}"
                    if higher_cache_key not in self._agent_cache:
                        self._agent_cache[higher_cache_key] = make_tiered_agent(domain, higher_tier)
                    print(f"[model_registry] Reusing loaded {higher_tier} model for {tier} query")
                    return self._agent_cache[higher_cache_key], higher_tier
        
        # Create new agent and mark model as loaded
        agent = make_tiered_agent(domain, tier)
        self._agent_cache[cache_key] = agent
        self._loaded_models.add(agent.model)
        self._model_load_times[agent.model] = time.time()
        return agent, tier
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a specific model from Ollama and remove from tracking."""
        try:
            import requests
            requests.post(
                f"{self._ollama_url}/api/generate",
                json={"model": model_name, "keep_alive": 0},
                timeout=5,
            )
            self._loaded_models.discard(model_name)
            self._model_load_times.pop(model_name, None)
            return True
        except Exception:
            return False
    
    def unload_domain_models(self, domain: str, keep_tier: str | None = None) -> None:
        """
        Unload all models for a domain except optionally keeping one tier.
        Useful when topic changes to free up memory.
        Also clears the agent cache for the domain.
        """
        for tier, model in TIERED_MODELS[domain].items():
            if keep_tier and tier == keep_tier:
                continue
            # Always remove from cache
            cache_key = f"{domain}_{tier}"
            self._agent_cache.pop(cache_key, None)
            # Only unload from Ollama if we're tracking it as loaded
            if model in self._loaded_models:
                print(f"[model_registry] Unloading {domain}/{tier} ({model})")
                self.unload_model(model)
    
    def clear_cache(self) -> None:
        """Clear the agent cache (does not unload models from Ollama)."""
        self._agent_cache.clear()
    
    def get_cache_info(self) -> dict:
        """Get information about cached agents and loaded models."""
        return {
            "cached_agents": list(self._agent_cache.keys()),
            "loaded_models": list(self._loaded_models),
            "cache_size": len(self._agent_cache),
        }
    
    def refresh_state(self) -> None:
        """Refresh internal state with actual Ollama state (call periodically)."""
        self._sync_with_ollama()


# Global model registry instance
_model_registry = ModelRegistry()


# ── Build master agent registry from active config ────────────────────────────

m  = ACTIVE_MODELS
bh = ACTIVE_BEHAVIOUR

AGENTS = {
    "coordinator": Agent(
        name="coordinator",
        model=m["coordinator"],
        system_prompt=(
            "You are the coordinator of a multi-agent AI system. "
            "Given a user query, classify it into exactly ONE of: "
            "'coding', 'math', 'writing', 'general', 'vision'. "
            "Respond with ONLY the single word label, nothing else."
        ),
        temperature=0.0,
        keep_alive="30m",
    ),
    "difficulty_estimator": Agent(
        name="difficulty_estimator",
        model=m["cheap_validator"],
        system_prompt=(
            "Estimate the difficulty of the user's query. "
            "Reply in this EXACT format, nothing else:\n"
            "DIFFICULTY: <easy or medium or hard>\n"
            "RATIONALE: <one short sentence>\n\n"
            "Criteria:\n"
            "- easy: simple facts, syntax questions, one-step reasoning, short emails\n"
            "- medium: multi-step reasoning, algorithms, essays, 2+ part questions\n"
            "- hard: complex systems, advanced proofs, architecture, multi-file code, research-level"
        ),
        temperature=0.0,
        keep_alive=bh["validator_keep_alive"],
    ),
    "coding": Agent(
        name="coding",
        model=m["coding"],
        system_prompt=(
            "You are an expert coding specialist. Handle complex systems, architecture "
            "design, advanced algorithms, multi-file projects, and tricky edge cases. "
            "Write production-quality, well-commented code. Always use Python code blocks. "
            "After the code block, explain the approach and any important edge cases. "
            "CRITICAL: You must strictly follow user preferences from context. If they prefer recursion "
            "over iteration, ALL solutions must be recursive, no exceptions. If they have other preferences, "
            "honor them completely. User preferences are the highest priority - even if a recursive solution "
            "is less efficient, you must still provide it if that's what the user prefers."
        ),
        temperature=0.2,
        keep_alive=bh["specialist_keep_alive"],
    ),
    "math": Agent(
        name="math",
        model=m["math"],
        system_prompt=_DOMAIN_SYSTEM_PROMPTS["math"]["tier3"],
        temperature=0.2,
        keep_alive=bh["specialist_keep_alive"],
    ),
    "writing": Agent(
        name="writing",
        model=m["writing"],
        system_prompt=_DOMAIN_SYSTEM_PROMPTS["writing"]["tier3"],
        temperature=0.6,
        keep_alive=bh["specialist_keep_alive"],
    ),
    "general": Agent(
        name="general",
        model=m["general"],
        system_prompt=(
            "You are a thoughtful general-purpose assistant. Engage in deep discussion, "
            "nuanced analysis, and complex multi-part questions. Provide thorough, "
            "well-reasoned answers. CRITICAL: You must use context from previous conversations "
            "to maintain conversation continuity. Remember details like names, preferences, and "
            "past discussions, and reference them appropriately in your responses. When users "
            "express preferences, acknowledge them clearly and confirm you'll honor them in "
            "relevant future interactions."
        ),
        temperature=0.4,
        keep_alive=bh["specialist_keep_alive"],
    ),
    "vision": Agent(
        name="vision",
        model=m["vision"],
        system_prompt=(
            "You are the vision extraction/OCR stage of a multi-agent system. "
            "Do not answer the user's question and do not choose a category or difficulty. "
            "Only extract factual information from the image for a downstream specialist.\n"
            "- Transcribe visible text accurately, preserving code and equations.\n"
            "- Describe relevant UI elements, diagrams, charts, tables, and relationships.\n"
            "- If text is unreadable, say so instead of guessing.\n"
            "Return a concise, structured image description containing only observations."
        ),
        temperature=0.2,
        keep_alive=bh["specialist_keep_alive"],
    ),
    "cheap_validator": Agent(
        name="cheap_validator",
        model=m["cheap_validator"],
        system_prompt=(
            "You are a minimal, targeted quality validator. Your role is to check whether the agent's "
            "answer matches the user's request and constraints, and to repair only when needed.\n\n"
            "Do NOT rewrite correct answers just to make them 'better'. Do NOT regenerate the entire answer "
            "when the original is already correct.\n\n"
            "Your job:\n"
            "1. Check if the answer is correct, relevant, and complete for the given query.\n"
            "2. Check whether it respects explicit user preferences, names, tone, and prior context.\n"
            "3. If it is correct, output PASS and stop.\n"
            "4. If it is wrong or incomplete, output FAIL and list only the actual issues.\n"
            "5. When fixing, provide only the minimal necessary corrected content or corrected snippet.\n\n"
            "Reply in this EXACT format:\n"
            "STATUS: <PASS or FAIL>\n"
            "ISSUES: <list of actual issues, or 'none'>\n"
            "REPAIR: <minimal corrected content only if needed; omit entirely when PASS>\n\n"
            "Rules:\n"
            "  - PASS means the answer is already good and should be kept. Do not produce a REPAIR.\n"
            "  - FAIL means the answer is wrong, incomplete, or violates user constraints.\n"
            "  - Prefer minimal repairs over full rewrites.\n"
            "  - If the answer already uses prior context correctly, do not flag it as a problem.\n"
            "  - Keep the repair focused on the failing part only.\n"
            "  - NEVER echo the original answer as the REPAIR. If you cannot provide a materially different fix, choose PASS."
        ),
        temperature=0.1,
        keep_alive=bh["validator_keep_alive"],
    ),
    "deep_validator": Agent(
        name="deep_validator",
        model=m["deep_validator"],
        system_prompt=(
            "You are a focused repair validator. Your job is to confirm whether the original answer is correct, "
            "and to do targeted repair only when necessary.\n\n"
            "Do not rewrite a good answer just because it can be 'improved'. The default should be: keep the answer.\n\n"
            "Your tasks:\n"
            "1. Verify whether the reported issues are real.\n"
            "2. Look for any hidden errors or missing constraints the quick validator missed.\n"
            "3. If the answer is valid, return PASS and do not provide a repair.\n"
            "4. If it is invalid, return FAIL and provide only the minimal corrected answer or corrected section.\n\n"
            "IMPORTANT: If a concern is about prior conversation context, names, or user preferences, it is not a real issue unless the answer clearly violates the request.\n\n"
            "Reply in this EXACT format:\n"
            "STATUS: <PASS or FAIL>\n"
            "CONFIRMED_ISSUES: <actual problems, or 'none'>\n"
            "REPAIR: <minimal corrected content only if needed; omit entirely when PASS>\n\n"
            "Rules:\n"
            "  - PASS means keep the original answer unchanged.\n"
            "  - FAIL means produce the smallest fix required.\n"
            "  - Do not rewrite correct explanations, working code, or valid logic just to rephrase it.\n"
            "  - NEVER return the original answer verbatim as a REPAIR. If no real repair is needed, return PASS."
        ),
        temperature=0.1,
        keep_alive=bh["deep_validator_keep_alive"],
    ),
}
