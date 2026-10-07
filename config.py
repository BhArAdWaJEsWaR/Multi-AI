"""
config.py
─────────
Single place to control everything.
Switch HARDWARE_PROFILE to match your machine, or override individual fields.

Profiles:
  "low"  → 8 GB RAM  — embedding-only routing, no deep validator, small models
  "mid"  → 16 GB RAM — LLM coordinator, no deep validator, 7B models
  "high" → 32 GB RAM — LLM coordinator, deep validator enabled, larger models

Tiered sub-agents (difficulty-based escalation):
  coding: easy → 1.5B → medium → 3B → hard → 7B
  math:   easy → 1.5B → medium → 3B (vl) → hard → 7B
"""

HARDWARE_PROFILE = "mid"   # ← change this to "mid" or "high" on better hardware

# ── Tiered sub-agent model assignments ────────────────────────────────────────
# Each domain has 3 tiers: tier1 (small/fast), tier2 (medium), tier3 (large/accurate)
TIERED_MODELS = {
    "coding": {
        "tier1": "qwen2.5-coder:1.5b",   # easy: syntax, simple scripts, debugging
        "tier2": "qwen2.5-coder:3b",     # medium: algorithms, data structures
        "tier3": "qwen2.5-coder:7b",     # hard: complex systems, architecture
    },
    "math": {
        "tier1": "deepseek-r1:1.5b",     # easy: arithmetic, basic algebra
        "tier2": "mightykatun/qwen2.5-math:1.5b",  # medium: calculus, probability
        "tier3": "deepseek-r1:7b",       # hard: proofs, advanced math
    },
    "writing": {
        "tier1": "llama3.2:latest",      # easy: short messages, emails
        "tier2": "gemma3:4b",            # medium: essays, reports
        "tier3": "gemma3:4b",            # hard: long-form, creative writing
    },
    "general": {
        "tier1": "llama3.2:latest",      # easy: facts, simple questions, preferences
        "tier2": "qwen3:4b",             # medium: explanations, discussion
        "tier3": "qwen3:4b",             # hard: complex reasoning
    },
}

# Difficulty → tier mapping
DIFFICULTY_TIER = {
    "easy":   "tier1",
    "medium": "tier2",
    "hard":   "tier3",
}

# Vision model (for screenshot / image analysis)
VISION_MODEL = "qwen2.5vl:3b"

# ── Model assignments per profile ────────────────────────────────────────────
MODELS = {
    "low": {
        "coordinator":     "qwen3:4b",
        "coding":          TIERED_MODELS["coding"]["tier3"],
        "math":            TIERED_MODELS["math"]["tier3"],
        "writing":         TIERED_MODELS["writing"]["tier3"],
        "general":         TIERED_MODELS["general"]["tier3"],
        "vision":          VISION_MODEL,
        "cheap_validator": "phi4-mini",
        "deep_validator":  TIERED_MODELS["math"]["tier3"],
    },
    "mid": {
        "coordinator":     "qwen3:4b",
        "coding":          TIERED_MODELS["coding"]["tier3"],
        "math":            TIERED_MODELS["math"]["tier3"],
        "writing":         TIERED_MODELS["writing"]["tier3"],
        "general":         TIERED_MODELS["general"]["tier3"],
        "vision":          VISION_MODEL,
        "cheap_validator": "phi4-mini",
        "deep_validator":  TIERED_MODELS["math"]["tier3"],
    },
    "high": {
        "coordinator":     "qwen3:4b",
        "coding":          TIERED_MODELS["coding"]["tier3"],
        "math":            TIERED_MODELS["math"]["tier3"],
        "writing":         "gemma3:4b",
        "general":         TIERED_MODELS["general"]["tier3"],
        "vision":          VISION_MODEL,
        "cheap_validator": "phi4-mini",
        "deep_validator":  TIERED_MODELS["math"]["tier3"],
    },
}

# ── Behaviour flags per profile ───────────────────────────────────────────────
BEHAVIOUR = {
    "low": {
        "use_llm_coordinator": False,
        "use_deep_validator":  False,
        "use_tiered_agents":   True,    # enable 1.5B → 3B → 7B escalation
        "use_memory":          True,
        "sim_threshold":       0.40,
        "specialist_keep_alive": "30m",  # Increased for better persistence
        "validator_keep_alive":  "1h",   # Keep validator loaded longer
        "deep_validator_keep_alive": "0",
        "escalate_on_validator_fail": True,  # re-run with tier3 if validation fails
    },
    "mid": {
        "use_llm_coordinator": True,
        "use_deep_validator":  False,
        "use_tiered_agents":   True,
        "use_memory":          True,
        "sim_threshold":       0.50,
        "specialist_keep_alive": "45m",  # Increased for better persistence
        "validator_keep_alive":  "1h",   # Keep validator loaded longer
        "deep_validator_keep_alive": "10m",
        "escalate_on_validator_fail": True,
    },
    "high": {
        "use_llm_coordinator": True,
        "use_deep_validator":  True,
        "use_tiered_agents":   True,
        "use_memory":          True,
        "sim_threshold":       0.55,
        "specialist_keep_alive": "1h",   # Keep specialists loaded for 1 hour
        "validator_keep_alive":  "2h",   # Keep validators loaded longer
        "deep_validator_keep_alive": "30m",
        "escalate_on_validator_fail": True,
    },
}

# ── Local dataset paths ───────────────────────────────────────────────────────
DATASET_PATHS = {
    "math":     "/Users/bharadwaj/Downloads/Math Questions.json",
    "leetcode": "/Users/bharadwaj/Downloads/Leetcode Questions.csv",
}

# Default Hugging Face dataset for general QA retrieval
DEFAULT_HF_DATASET = "Magpie-Align/Magpie-Air-300K-Filtered"

# ── Active config (what the rest of the code imports) ────────────────────────
ACTIVE_MODELS    = MODELS[HARDWARE_PROFILE]
ACTIVE_BEHAVIOUR = BEHAVIOUR[HARDWARE_PROFILE]