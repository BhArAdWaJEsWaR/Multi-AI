"""
core/orchestrator.py

Pipeline per query:
  1. Detect if query includes an image → route to vision agent
  2. Embed query → topic shift detection (skip routing if same topic)
     On "low" profile: embedding-only routing, coordinator LLM never called
  3. Estimate difficulty (easy / medium / hard) using cheap validator
  4. Retrieve context: vector memory + local dataset examples + HF Magpie
  5. Tiered specialist agent generates response (streamed)
     - easy   → tier1 (1.5B class)
     - medium → tier2 (3B / 4B class)
     - hard   → tier3 (7B class)
  6. Validation:
       coding   → run the code; escalate tier on crash
       non-code → cheap_validator first; escalate tier on LOW confidence
  7. Keep the exchange in temporary in-process chat history
"""

import os
import re
import time
import numpy as np
import requests as _requests
from sentence_transformers import SentenceTransformer

from config import ACTIVE_BEHAVIOUR, TIERED_MODELS, MODELS, HARDWARE_PROFILE
from core.agents import (
    AGENTS,
    make_tiered_agent,
    make_tiered_general_agent,
    get_tier_for_difficulty,
    _model_registry,
)
from core.memory import VectorMemory
from core.code_validator import validate_code_response
from core.datasets import build_dataset_context, get_hf_general_context, DatasetRouter

# kNN difficulty confidence threshold. Below this → fall back to LLM estimator.
KNN_DIFF_THRESHOLD = 0.50
# kNN task override threshold: dataset says math/coding with ≥confidence → beat base (vision/general/math)
KNN_TASK_OVERRIDE_THRESHOLD = 0.50
# Higher bar for dataset to beat a "writing" base label (avoids triggering coding on pure prose requests)
KNN_TASK_OVERRIDE_WRITING_THRESHOLD = 0.80
# Higher bar for dataset to beat a "vision" base label when image is present
# (for text-only queries, spurious vision hits are overridden at the normal threshold)

VALID_LABELS       = {"coding", "math", "writing", "general", "vision"}
SIM_THRESHOLD      = ACTIVE_BEHAVIOUR["sim_threshold"]
USE_LLM_COORD      = ACTIVE_BEHAVIOUR["use_llm_coordinator"]
USE_DEEP_VAL       = ACTIVE_BEHAVIOUR["use_deep_validator"]
USE_TIERED         = ACTIVE_BEHAVIOUR["use_tiered_agents"]
ESCALATE_ON_FAIL   = ACTIVE_BEHAVIOUR["escalate_on_validator_fail"]
OLLAMA_URL         = "http://localhost:11434"

_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".heic")
_IMAGE_EXT_RE = re.compile(r"\.(png|jpg|jpeg|gif|bmp|webp|heic)$", re.IGNORECASE)
# Temporary chat context. It is never written to ChromaDB and disappears when
# the Orchestrator process exits.
MAX_CHAT_TURNS = 24


def resolve_file_path(candidate: str) -> str | None:
    """Resolve an image file path handling ~, quotes, macOS narrow no-break spaces, and spaces."""
    candidate = candidate.strip("'\"`<> \t\n")
    if not candidate:
        return None
    candidate = os.path.expanduser(candidate)
    if os.path.isfile(candidate):
        return candidate
    # Handle macOS screenshot narrow no-break space (\u202f) before AM/PM
    for search, repl in [(" PM", "\u202fPM"), (" AM", "\u202fAM"), ("\u202f", " ")]:
        alt = candidate.replace(search, repl)
        if os.path.isfile(alt):
            return alt
    # Check directory listing with whitespace-normalized comparison
    dirname, basename = os.path.split(candidate)
    dirname = dirname or "."
    if os.path.isdir(dirname):
        norm_base = "".join(basename.split()).lower()
        try:
            for f in os.listdir(dirname):
                if "".join(f.split()).lower() == norm_base:
                    resolved = os.path.join(dirname, f)
                    if os.path.isfile(resolved):
                        return resolved
        except OSError:
            pass
    return None


def extract_image_info(text: str, override_image: str | None = None) -> tuple[str | None, str, str | None]:
    """
    Extracts image path and prompt.
    Returns: (resolved_image_path, query_to_run, error_message)
    """
    if override_image:
        resolved = resolve_file_path(override_image)
        if resolved:
            return resolved, text or "Describe this image in detail.", None
        return None, text, f"Image file not found: '{override_image}'. Please check the path."

    # 1. 'image <path> [optional prompt]'
    if text.lower().startswith("image "):
        rest = text[6:].strip()
        # Direct file check
        resolved = resolve_file_path(rest)
        if resolved:
            return resolved, "Describe this image in detail.", None
        # Quoted path + prompt
        m = re.match(r'^[\'"]([^\'"]+)[\'"]\s*(.*)$', rest)
        if m:
            path_part, prompt_part = m.groups()
            resolved = resolve_file_path(path_part)
            if resolved:
                return resolved, prompt_part.strip() or "Describe this image in detail.", None
            return None, text, f"Image file not found: '{path_part}'."
        # Path with spaces ending in extension + prompt
        for ext in _IMAGE_EXTS:
            if ext in rest.lower():
                idx = rest.lower().find(ext) + len(ext)
                path_cand = rest[:idx].strip()
                prompt_cand = rest[idx:].strip()
                resolved = resolve_file_path(path_cand)
                if resolved:
                    return resolved, prompt_cand or "Describe this image in detail.", None

        return None, text, f"Image file not found: '{rest}'."

    # 2. Quoted file paths anywhere in query
    for m in re.finditer(r'[\'"]([^\'"]+\.(?:png|jpg|jpeg|gif|bmp|webp|heic))[\'"]', text, re.IGNORECASE):
        resolved = resolve_file_path(m.group(1))
        if resolved:
            return resolved, text, None

    # 3. Scan backward from image extension anywhere in text
    for m in re.finditer(r'\.(png|jpg|jpeg|gif|bmp|webp|heic)\b', text, re.IGNORECASE):
        end_idx = m.end()
        prefix = text[:end_idx]
        split_points = [0] + [i + 1 for i, ch in enumerate(prefix) if ch in ' \t\'"`<>(),:;']
        for start_idx in split_points:
            candidate = prefix[start_idx:].strip('\'"`<> \t')
            if any(candidate.lower().endswith(ext) for ext in _IMAGE_EXTS):
                resolved = resolve_file_path(candidate)
                if resolved:
                    return resolved, text, None

    return None, text, None


def _looks_like_image_path(text: str) -> str | None:
    img, _, _ = extract_image_info(text)
    return img


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# Terms that require whole-word matching to avoid false positives
# (e.g. "function" inside "functionality", "class" inside "classical").
_CODING_WORD_BOUNDARY_TERMS = (
    "function", "class", "method", "loop", "array", "list",
    "sort", "implement", "algorithm", "algorithms", "program", "script",
    "code", "debug", "parse", "regex",
    "python", "java", "javascript",
)
# Terms that are already unambiguous — substring match is safe
_CODING_EXACT_TERMS = (
    "write code", "write a function", "write a class", "write a program",
    "c++", "def ", "for ", "while ", "if ",
    "binary search", "quick sort", "merge sort",
    "api", "html", "css", "file io",
    "programming", "scripting", "coding",
)
_MATH_WORD_BOUNDARY_TERMS = (
    "solve", "calculate", "prove", "simplify",
    "mean", "median", "variance", "prime", "angle",
    "matrix", "algebra", "statistics", "geometry",
)
_MATH_EXACT_TERMS = (
    "probability", "birthday problem", "derivative", "integral",
    "equation", "factorial", "triangle", "sum of",
    "what is the probability", "find the probability",
    "sqrt", "standard deviation", "permutation", "combination",
)


def _word_match(q: str, terms: tuple) -> bool:
    """Return True if any term matches as a whole word in q."""
    return any(re.search(r"\b" + re.escape(t) + r"\b", q) for t in terms)


def _looks_like_math_query(query: str) -> bool:
    q = query.lower()
    has_math = (
        any(term in q for term in _MATH_EXACT_TERMS)
        or _word_match(q, _MATH_WORD_BOUNDARY_TERMS)
    )
    has_code = (
        any(term in q for term in _CODING_EXACT_TERMS)
        or _word_match(q, _CODING_WORD_BOUNDARY_TERMS)
    )
    return has_math and not has_code


def _looks_like_coding_query(query: str) -> bool:
    q = query.lower()
    return (
        any(term in q for term in _CODING_EXACT_TERMS)
        or _word_match(q, _CODING_WORD_BOUNDARY_TERMS)
    )


def _looks_like_general_explanation_query(query: str) -> bool:
    q = query.lower().strip()
    patterns = (
        "what is ", "what are ", "explain ", "describe ", "define ",
        "how does ", "how do ", "why does ", "why do ", "tell me about ",
        "difference between ", "compare ", "what is the difference between ",
    )
    return any(p in q for p in patterns) and not _looks_like_coding_query(query)


def _needs_long_term_memory(query: str) -> bool:
    """Only retrieve persistent Q&A memory when the user asks for it.

    Current-turn continuity is supplied separately through conversation history.
    Broad vector retrieval for every request can inject unrelated old answers.
    """
    q = query.lower()
    memory_signals = (
        "remember", "do you know", "what is my name", "who am i",
        "what did i say", "what did we discuss", "earlier", "before",
        "my preference", "my preferences", "what do you know about me",
        "forget", "previous conversation", "last time",
    )
    return any(signal in q for signal in memory_signals)


def unload_all_models() -> None:
    """Tell Ollama to unload every currently resident model to free RAM."""
    try:
        resp = _requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        for model in resp.json().get("models", []):
            _requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": model["name"], "keep_alive": 0},
                timeout=5,
            )
    except Exception:
        pass


# ── Embedding classifier ──────────────────────────────────────────────────────

_LABEL_EXAMPLES = {
    "coding": [
        "write a python function to sort a list",
        "implement quicksort algorithm in java",
        "debug this code and fix the error",
        "write a script to parse a CSV file",
        "code a binary search in C++",
        "implement a linked list class",
        "write a recursive fibonacci function",
        "code for bubble sort in python",
        "write unit tests for this function",
        "fix this syntax error in my program",
        "give me code for merge sort",
        "write a program that reads a file",
        "build me a responsive website with HTML and CSS",
        "create a landing page with flexbox and media queries",
        "write CSS for a mobile-first responsive navigation bar",
        "give me JavaScript code for a dark mode toggle",
        "make a React component for a todo list",
        "code a registration form with HTML form validation",
        "write a CSS grid layout for a dashboard",
        "create a carousel slider in vanilla JavaScript",
        "give me HTML and CSS for a pricing table",
        "write Tailwind classes for a responsive hero section",
        "code a login page with Bootstrap",
        "make an API call with fetch() in JavaScript",
        "write TypeScript interfaces for a user object",
        "build a calculator app using HTML CSS and JS",
        "how to center a div with flexbox",
        "code a hamburger menu for small screens",
    ],
    "math": [
        "solve x squared minus 4 equals zero",
        "calculate the derivative of x cubed",
        "what is 144 divided by 12",
        "find the integral of sin x",
        "prove that the sum of angles in a triangle is 180",
        "simplify this algebraic expression",
        "compute the probability of rolling a six",
        "what is 15 multiplied by 37",
        "find the eigenvalues of this matrix",
        "solve this system of linear equations",
        "what is the time complexity of merge sort",
        "calculate the mean and standard deviation",
        "what is 2 plus 3",
        "what is 2+3",
        "calculate 15 plus 27",
        "what is 25 times 4",
        "calculate 50 minus 18",
        "what is 100 divided by 5",
    ],
    "writing": [
        "write an email to my professor asking for an extension",
        "draft a cover letter for a software engineer position",
        "write an essay about climate change",
        "compose a professional apology message",
        "write a poem about the ocean",
        "summarize this paragraph in simpler words",
        "draft a formal complaint letter",
        "write a LinkedIn post about my project",
        "write a thank you note to my supervisor",
        "compose a resignation letter",
        "write a report on renewable energy",
        "draft a message to my team about the deadline",
        "explain in words how responsive web design works",
        "describe the principles of good UI design in a paragraph",
        "write a short blog post about why CSS matters",
    ],
    "general": [
        "what is my name",
        "who am I",
        "hi hello how are you good morning",
        "what is machine learning",
        "explain how the internet works",
        "who invented the telephone",
        "why does the sky appear blue",
        "what did we discuss earlier",
        "tell me about recursion",
        "what is the difference between RAM and ROM",
        "my name is bharadwaj",
        "do you remember what I said",
        "what did I ask you before",
    ],
    "vision": [
        "analyze this screenshot",
        "what's in this image",
        "describe this picture",
        "read the text in this png",
        "extract code from this screenshot",
        "what error is shown in this jpg",
        "explain the diagram in the image",
        "look at this image and tell me",
        "what does this chart show",
        "decode this qr code image",
    ],
}


class EmbeddingClassifier:
    def __init__(self, embedder: SentenceTransformer):
        self.embedder = embedder
        self.label_embeddings: dict[str, np.ndarray] = {}
        for label, examples in _LABEL_EXAMPLES.items():
            vecs = embedder.encode(examples, normalize_embeddings=True)
            mean = vecs.mean(axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-10)
            self.label_embeddings[label] = mean

    def classify(self, query_emb: np.ndarray, allow_vision: bool = True) -> str:
        candidates = self.label_embeddings if allow_vision else {
            k: v for k, v in self.label_embeddings.items() if k != "vision"
        }
        scores = {
            label: cosine_similarity(query_emb, emb)
            for label, emb in candidates.items()
        }
        for label, score in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"  {label:10s} {score:.3f}")
        return max(scores, key=scores.__getitem__)


# ── Difficulty estimator ──────────────────────────────────────────────────────

_DIFF_RE = re.compile(r"DIFFICULTY:\s*(easy|medium|hard)", re.IGNORECASE)


def estimate_difficulty(query: str, task_type: str) -> tuple[str, str]:
    """
    Estimate query difficulty (easy/medium/hard) using the cheap validator model.
    Falls back to 'medium' if parsing fails.
    Returns (difficulty, rationale).
    """
    raw = AGENTS["difficulty_estimator"].run(query, stream=False)
    m = _DIFF_RE.search(raw)
    difficulty = m.group(1).lower() if m else "medium"
    rationale = ""
    for line in raw.splitlines():
        if line.upper().startswith("RATIONALE:"):
            rationale = line.split(":", 1)[1].strip()
            break
    return difficulty, rationale


# ── Orchestrator ───────────────────────────────────────────────────────────────

class Orchestrator:
    def __init__(self, use_memory: bool = True, use_validator: bool = True):
        self.use_memory    = use_memory
        self.use_validator = use_validator
        self.memory        = VectorMemory() if use_memory else None

        print("[system] Loading embedding model...", flush=True)
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        print("[system] Embedding model ready.", flush=True)
        
        # Clean up memory: remove incorrect "I don't know your name" responses
        if self.use_memory:
            print("[system] Cleaning up memory...", flush=True)
            removed = self.memory.remove_by_pattern("I don't know your name")
            if removed > 0:
                print(f"[system]   Removed {removed} stale memory entries", flush=True)

        self._last_embedding: np.ndarray | None = None
        self._last_task_type: str | None        = None
        self._same_topic_streak: int            = 0
        self._conversation_history: list[dict[str, str]] = []

        # ── Authoritative preference store ────────────────────────────────────
        # Universal preference store across all domains (coding, writing, math, general, all)
        # Format: {key: {"domain": str, "instruction": str, "confirmation": str}}
        self._active_preferences: dict[str, dict] = {}

        self._emb_classifier = EmbeddingClassifier(self.embedder)

        print("[system] Building dataset embedding router (Math + Leetcode)...", flush=True)
        self._dataset_router = DatasetRouter(self.embedder)
        self._dataset_router.build()
        if self._dataset_router._embs is None:
            print("[system]   ⚠ Dataset router has no anchors (dataset missing?).", flush=True)
        else:
            print(f"[system]   Dataset router ready ({len(self._dataset_router._texts)} anchors, "
                  f"{self._dataset_router.k_neighbours}-NN).", flush=True)

        print(f"[system] Pre-warming validator ({AGENTS['cheap_validator'].model})...",
              end=" ", flush=True)
        AGENTS["cheap_validator"].run("hi", stream=False)
        print("ready.\n", flush=True)

        # Restore preferences from memory on startup
        if self.use_memory:
            self._load_preferences_from_memory()

    def _load_preferences_from_memory(self) -> None:
        """Restore all saved preferences across all domains from vector memory on startup."""
        pref_entries = self.memory.get_by_metadata({"type": "preference_update"})
        latest_by_key: dict[str, tuple[float, dict]] = {}
        invalid_keys: set[str] = set()
        for doc_text, meta in pref_entries:
            key = meta.get("pref_key", "")
            instruction = meta.get("instruction", "")
            domain = meta.get("domain", "all")
            confirmation = meta.get("confirmation", "")
            timestamp = float(meta.get("updated_at", meta.get("timestamp", 0)) or 0)
            # Older parser versions persisted fragments of read-only questions
            # as preference keys. They must never be restored as instructions.
            if key and instruction and not self._is_valid_preference(key, instruction):
                invalid_keys.add(key)
                continue
            if key and instruction and self._is_valid_preference(key, instruction):
                current = latest_by_key.get(key)
                if current is None or timestamp >= current[0]:
                    latest_by_key[key] = (timestamp, {
                        "domain": domain,
                        "instruction": instruction,
                        "confirmation": confirmation,
                        "updated_at": timestamp,
                    })

        for key in invalid_keys:
            self.memory.remove_by_metadata({"type": "preference_update", "pref_key": key})

        # The newest saved value for a key is authoritative, regardless of the
        # order returned by the vector store.
        self._active_preferences = {}
        replaced_keys: set[str] = set()
        for key, (timestamp, value) in sorted(latest_by_key.items(), key=lambda item: item[1][0]):
            replaced_keys.update(self._apply_preference(key, value, timestamp))
        for key in replaced_keys:
            self.memory.remove_by_metadata({"type": "preference_update", "pref_key": key})
        if self._active_preferences:
            summary_list = [f"{k} [{v.get('domain', 'all')}]" for k, v in self._active_preferences.items()]
            print(f"[system] Restored preferences from memory: {', '.join(summary_list)}", flush=True)

    @staticmethod
    def _is_valid_preference(key: str, instruction: str) -> bool:
        """Reject malformed records created by the old preference-query parser."""
        invalid_fragments = (
            "ece_in_coding",
            "ece_in_",
            "ence_in_",
            "ence_do_you_remember",
            "do_you_remember",
            "what_is_my",
            "which_coding",
            "explain_",
            "anatom",
        )
        text = f"{key} {instruction}".lower()
        return not any(fragment in text for fragment in invalid_fragments)

    @staticmethod
    def _replacement_terms(instruction: str) -> set[str]:
        """Return the alternatives explicitly superseded by a preference."""
        match = re.search(
            r"\b(?:over|instead of|rather than|not)\b(.+)$",
            instruction.lower(),
        )
        if not match:
            return set()
        return {
            token for token in re.findall(r"[a-z0-9]+", match.group(1))
            if len(token) > 2
        }

    @staticmethod
    def _preference_text(value: dict) -> str:
        return " ".join((
            str(value.get("instruction", "")),
            str(value.get("key", "")),
        )).lower()

    def _apply_preference(self, key: str, value: dict, timestamp: float) -> set[str]:
        """Apply a preference while letting a newer explicit alternative win."""
        instruction = value.get("instruction", "")
        replacement_terms = self._replacement_terms(instruction)
        domain = value.get("domain", "all")
        removed_keys: set[str] = set()

        # Treat a newer explicit preference for one approach as replacing an
        # older preference for its common alternative.
        conflict_groups = (
            ({"recursive", "recursion"}, {"iterative", "iteration"}),
        )
        current_terms = set(re.findall(r"[a-z0-9]+", self._preference_text({"key": key, **value})))
        if replacement_terms:
            for old_key, old_value in list(self._active_preferences.items()):
                same_scope = old_value.get("domain") in (domain, "all") or domain == "all"
                old_text = self._preference_text({**old_value, "key": old_key})
                if same_scope and any(term in old_text.split() for term in replacement_terms):
                    self._active_preferences.pop(old_key, None)
                    removed_keys.add(old_key)
        for old_key, old_value in list(self._active_preferences.items()):
            if old_key == key or old_value.get("domain") not in (domain, "all") and domain != "all":
                continue
            old_terms = set(re.findall(r"[a-z0-9]+", self._preference_text({"key": old_key, **old_value})))
            if any((current_terms & first and old_terms & second) or
                   (current_terms & second and old_terms & first)
                   for first, second in conflict_groups):
                self._active_preferences.pop(old_key, None)
                removed_keys.add(old_key)

        self._active_preferences[key] = {
            **value,
            "updated_at": timestamp,
        }
        return removed_keys

    def _extract_explicit_preference(self, query: str) -> dict | None:
        """Fast deterministic parser for clear preference statements.

        This is the universal memory layer: it writes any explicit preference into a
        generic preference store and infers the appropriate domain from the user's
        wording, rather than hard-coding one particular preference.
        """
        q = query.strip()
        if not q:
            return None
        q_lower = q.lower()

        # Explicit name preference / personal info
        name_match = re.search(r"(?:my name is|call me|i am|i'm)\s+([a-zA-Z][a-zA-Z' -]+)", q, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip()
            return {
                "type": "SET",
                "domain": "general",
                "key": "user_name",
                "instruction": f"Your name is {name}.",
                "confirmation": f"Got it — I will remember your name is {name}.",
            }

        preference_markers = (
            "prefer", "preference", "like", "always", "never",
            "from now on", "for now on", "use", "instead of", "rather than",
            "my choice is", "i choose", "i would like"
        )
        if not any(marker in q_lower for marker in preference_markers):
            # Avoid treating generic 'I want to learn X' or 'I need to know X' as persistent preferences
            if re.search(r"\b(i|we|you)\s+(want|need)\s+to\s+(learn|know|understand|get|see|read|ask|explain)\b", q_lower):
                return None
            if "want" not in q_lower and "need" not in q_lower:
                return None
            # If 'want'/'need' is present but not used as a preference-signal, skip it.
            return None

        # Domain inference from the content of the preference phrase.
        domain = "all"
        if any(word in q_lower for word in ["code", "coding", "function", "algorithm", "python", "java", "javascript", "c++", "recursion", "iteration", "loop", "class", "script", "program"]):
            domain = "coding"
        elif any(word in q_lower for word in ["write", "writing", "email", "essay", "tone", "style", "sentence", "paragraph", "report", "summary"]):
            domain = "writing"
        elif any(word in q_lower for word in ["math", "equation", "proof", "derivative", "integral", "probability", "matrix", "statistics", "step by step", "steps"]):
            domain = "math"
        elif any(word in q_lower for word in ["name", "persona", "voice", "respond", "talk", "greet", "hello", "friendly", "formal"]):
            domain = "general"

        # Extract the preference phrase after markers.
        phrase = None
        for marker in ("prefer ", "preference is ", "i like ", "i want ", "i need ", "i would like ", "always use ", "use ", "from now on use ", "my choice is "):
            idx = q_lower.find(marker)
            if idx != -1:
                start = idx + len(marker)
                phrase = q[start:].strip()
                break
        if phrase is None:
            # Generic fallback: extract after 'prefer'/'like' or after 'i prefer'.
            for marker in ("prefer", "like"):
                idx = q_lower.find(marker)
                if idx != -1:
                    phrase = q[idx + len(marker):].strip()
                    break

        if not phrase:
            return None

        # Clean up common trailing wrappers and keep it concise.
        phrase = phrase.strip(" .,!?:;\n\r\t")
        phrase = re.sub(r"^(over|than|to|for|in|on|with)\s+", "", phrase, flags=re.I)
        phrase = re.sub(r"\b(?:the|a|an)\b\s+", "", phrase, flags=re.I)
        phrase = phrase.strip()

        if not phrase or len(phrase) > 220:
            return None

        phrase = phrase.strip(" \t\n\r.,;:!?'")
        if phrase.lower().startswith("to "):
            phrase = phrase[3:].strip()

        instruction = f"Use {phrase} when relevant."
        if phrase.lower().startswith("be "):
            instruction = phrase[3:].capitalize() + " when relevant."
        if phrase.lower().startswith("not ") or phrase.lower().startswith("don't ") or phrase.lower().startswith("do not "):
            instruction = f"Avoid {phrase[4:].strip()} when relevant."
        if " instead of " in q_lower:
            instruction = f"Prefer {phrase} over the alternative options unless the user explicitly requests otherwise."
        if " over " in q_lower and "prefer" in q_lower:
            instruction = f"Prefer {phrase} over the alternatives unless the user explicitly requests otherwise."

        return {
            "type": "SET",
            "domain": domain,
            "key": re.sub(r"[^a-z0-9_]+", "_", phrase.lower())[:40] or "preference",
            "instruction": instruction,
            "confirmation": "Got it — I will remember that preference and apply it in relevant future responses.",
        }

    def _detect_preference_update(self, query: str) -> dict | None:
        """
        Universal, domain-agnostic preference engine using fast local LLM.
        Detects setting, updating, clearing, or querying persistent preferences
        across all domains: coding, writing, math, general, or all.
        Returns a structured dict or None.
        """
        q_lower = query.lower()
        # Questions about an existing preference must never be sent to the
        # setter classifier. They are read-only inspection requests.
        asks_about_preference = (
            ("what" in q_lower or "which" in q_lower or "show" in q_lower
             or "tell me" in q_lower or "do you remember" in q_lower)
            and ("preference" in q_lower or "prefer" in q_lower
                 or "choice" in q_lower or "setting" in q_lower)
        )
        if asks_about_preference:
            domain = "all"
            for candidate in ("coding", "writing", "math", "general"):
                if candidate in q_lower:
                    domain = candidate
                    break
            return {"type": "SHOW", "domain": domain}

        direct = self._extract_explicit_preference(query)
        if direct:
            return direct

        pref_triggers = (
            "prefer", "preference", "choice", "always", "never", "from now on",
            "henceforth", "use", "using", "switch", "change", "update", "set",
            "reset", "stop", "remove", "delete", "clear", "like", "want", "need",
            "my name", "call me", "show", "what are my", "tone", "style", "format"
        )
        has_preference_trigger = any(word in q_lower for word in pref_triggers)
        has_memory_question = any(signal in q_lower for signal in (
            "remember", "do you know", "what did i say", "what did we discuss",
            "my preference", "my preferences", "what do you know about me",
            "what is my name", "who am i", "previous conversation", "last time",
        ))
        # Do not send ordinary short questions to the preference LLM. Small
        # typos such as "exlpain the human anatomy" can otherwise be
        # hallucinated as SHOW/SET actions by the cheap classifier.
        if not has_preference_trigger and not has_memory_question:
            return None
        if not has_preference_trigger and has_memory_question and len(query.split()) > 12:
            return None

        model = MODELS[HARDWARE_PROFILE]["cheap_validator"]
        system_prompt = (
            "You are an intelligent preference and system directive analyzer for a multi-agent AI assistant.\n"
            "Your task is to analyze the user's message and determine whether they are setting, changing, removing, or asking about a PERSISTENT preference, rule, or persona for future interactions.\n\n"
            "Domains:\n"
            "- coding: rules for code generation (e.g. recursion vs iteration, language, formatting, libraries, comments, type hints)\n"
            "- writing: rules for text generation (e.g. tone, structure, bullet points, length, persona, reading level)\n"
            "- math: rules for mathematical tasks (e.g. step-by-step derivations, fractions vs decimals, LaTeX)\n"
            "- general: personal details or persona (e.g. user name, user background, greeting style)\n"
            "- all: global rules (e.g. language, conciseness, formatting)\n\n"
            "Guidelines:\n"
            "- If the user is asking to solve a specific problem, write code for a prompt, asking a question, or giving one-off instructions for a single query (e.g. 'can you give me code for bubble sort', 'what is 2+2', 'write a poem about cats', 'use quicksort on [3,1,2]'), output:\n"
            "IS_PREFERENCE: NO\n\n"
            "- If the user is asking what their current preferences or settings are (e.g. 'what are my preferences', 'show my settings'), output:\n"
            "IS_PREFERENCE: SHOW\n\n"
            "- If the user IS stating, updating, or clearing a persistent rule/preference for future responses (e.g. 'from now on...', 'always...', 'never...', 'stop doing X', 'remove preference for X', 'change preference to X'), output EXACTLY:\n"
            "IS_PREFERENCE: YES\n"
            "ACTION: <SET or REMOVE>\n"
            "DOMAIN: <coding | writing | math | general | all>\n"
            "KEY: <short snake_case key identifying the preference, e.g. iteration_style, writing_tone, math_working, language, user_name>\n"
            "INSTRUCTION: <clear, imperative directive for how the assistant MUST behave, e.g. 'Use purely recursive solutions; never use loops.', 'Write all explanations simply like explaining to a 5-year-old.' or NONE if ACTION is REMOVE>\n"
            "CONFIRMATION: <friendly, concise 1-sentence confirmation message confirming the change>"
        )

        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                "stream": False,
                "options": {"temperature": 0.0}
            }
            resp_raw = _requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=15)
            resp = resp_raw.json()["message"]["content"].strip()
        except Exception:
            return None

        is_pref_m = re.search(r"IS_PREFERENCE:\s*(YES|NO|SHOW)", resp, re.IGNORECASE)
        pref_type = is_pref_m.group(1).upper() if is_pref_m else "NO"
        if pref_type == "NO":
            return None
        if pref_type == "SHOW":
            return {"type": "SHOW"}

        action_m = re.search(r"ACTION:\s*(SET|REMOVE)", resp, re.IGNORECASE)
        domain_m = re.search(r"DOMAIN:\s*([a-zA-Z]+)", resp, re.IGNORECASE)
        key_m = re.search(r"KEY:\s*([a-zA-Z0-9_-]+)", resp, re.IGNORECASE)
        instr_m = re.search(r"INSTRUCTION:\s*(.+)", resp, re.IGNORECASE)
        conf_m = re.search(r"CONFIRMATION:\s*(.+)", resp, re.IGNORECASE)

        action = action_m.group(1).upper() if action_m else "SET"
        domain = domain_m.group(1).lower() if domain_m else "all"
        if domain not in ("coding", "writing", "math", "general", "all"):
            domain = "all"
        key = key_m.group(1).lower() if key_m else "preference"
        instr = instr_m.group(1).strip() if instr_m else ""
        if instr.upper() in ("NONE", "REMOVE", "N/A", "CLEAR"):
            instr = ""
        conf = conf_m.group(1).strip() if conf_m else f"Preference '{key}' updated."

        if action == "SET" and not instr:
            instr = conf

        return {
            "type": action,
            "domain": domain,
            "key": key,
            "instruction": instr,
            "confirmation": conf
        }

    def _embed(self, text: str) -> np.ndarray:
        return self.embedder.encode(text, normalize_embeddings=True)

    # ── Classification ────────────────────────────────────────────────────────

    def classify(
        self,
        query: str,
        query_emb: np.ndarray,
        image_path: str | None,
        from_image_extraction: bool = False,
    ):
        """
        Returns (task_type, method, dataset_kNN_result_dict_or_None).
        Priority: image → cached topic → preference statement → dataset kNN (strong vote) → LLM / emb classifier.
        """
        if image_path:
            return "vision", "image_detected", None

        # Strong keyword guardrails first. General explanations like "what is X" or
        # "explain X" must not be hijacked by the previous task's embedding cache.
        # For image-derived input, extracted words such as "function", "code", or
        # "algorithm" may describe a diagram rather than the user's intended task.
        # Let the coordinator/classifier evaluate the complete extracted content.
        if not from_image_extraction and _looks_like_general_explanation_query(query):
            print("[keyword] Detected general explanation, routing to general before topic cache")
            return "general", "keyword_general", None
        if not from_image_extraction and _looks_like_math_query(query) and not _looks_like_coding_query(query):
            print("[keyword] Detected math query, routing to math before dataset override")
            return "math", "keyword_math", None
        if not from_image_extraction and _looks_like_coding_query(query) and not _looks_like_math_query(query):
            print("[keyword] Detected coding query, routing to coding before dataset override")
            return "coding", "keyword_coding", None

        if not from_image_extraction and self._last_embedding is not None and self._last_task_type is not None:
            sim = cosine_similarity(query_emb, self._last_embedding)
            print(f"[embedding] similarity: {sim:.3f} (threshold: {SIM_THRESHOLD})")
            if sim >= SIM_THRESHOLD:
                self._same_topic_streak += 1
                return self._last_task_type, "cached", None

        self._same_topic_streak = 0
        # Step A: dataset kNN over Math + Leetcode anchors for a coding/math/ambiguous vote
        ds_res = self._dataset_router.analyse(query_emb)
        if ds_res.get("ready"):
            t, tc = ds_res.get("task"), ds_res.get("task_conf", 0.0)
            counts = ds_res.get("top_task_counts")
            parts = [f"{lab}={n}" for lab, n in sorted(counts.items(), key=lambda x: -x[1])]
            print(f"[dataset-knn] task vote: {t or 'ambiguous'} ({tc:.2f}) | "
                  f"neighbours: {', '.join(parts)} | cos max={ds_res.get('top_cosine_max',0):.3f}")

        # Step B: run base classifier (embedding or LLM coordinator) for the candidate label space
        allow_vision = (image_path is not None)
        if not USE_LLM_COORD:
            base_task = self._emb_classifier.classify(query_emb, allow_vision=allow_vision)
            base_method = "embedding"
        else:
            raw       = AGENTS["coordinator"].run(query, stream=False).lower().strip()
            labels    = VALID_LABELS if allow_vision else (VALID_LABELS - {"vision"})
            base_task = next((l for l in labels if l in raw), "general")
            base_method = "coordinator"

        # Step C: merge — dataset math/coding vote beats noise labels ONLY when
        # the dataset-router actually declared a task (NOT None = gated via cosine domain relevance)
        # AND confidence is high enough for the base label:
        #  - beats "vision" / "general" / "math" / "coding" at normal threshold (0.50)
        #  - beats "writing" only at ≥0.80 confidence AND strong absolute cosine max (≥0.45)
        #      (protects "draft an email about responsive design")
        #  - never beats "vision" if an actual image was passed
        final_task = base_task
        final_method = base_method
        
        if (not from_image_extraction
                and ds_res.get("ready")
                and ds_res.get("task") is not None
                and ds_res.get("task_conf", 0) >= KNN_TASK_OVERRIDE_THRESHOLD):
            ds_task = ds_res["task"]
            ds_conf = ds_res["task_conf"]
            ds_cos_max = ds_res.get("top_cosine_max", 0.0)
            safe_labels_noise = {"vision", "general", "math", "coding"}
            has_image = image_path is not None
            if ds_task == "math" and base_task in safe_labels_noise:
                final_task, final_method = "math", "dataset-knn"
            elif ds_task == "coding":
                if base_task in safe_labels_noise:
                    final_task, final_method = "coding", "dataset-knn-strong"
                elif (base_task == "writing"
                      and ds_conf >= KNN_TASK_OVERRIDE_WRITING_THRESHOLD
                      and ds_cos_max >= 0.45):
                    final_task, final_method = "coding", "dataset-knn-strong"
            _ = has_image

        return final_task, final_method, ds_res if ds_res.get("ready") else None

    def _extract_visual_content(self, image_path: str, user_request: str) -> str:
        """Extract image facts; classification and answering happen downstream."""
        extraction_prompt = (
            "Extract the useful contents of this image for another assistant. "
            "Do not answer the user's request, classify it, or recommend a model. "
            "Transcribe visible text, code, equations, labels, and relevant visual "
            "relationships. User request for relevance only:\n"
            f"{user_request}"
        )
        return AGENTS["vision"].run(
            extraction_prompt,
            stream=False,
            image_path=image_path,
        ).strip()

    # ── Tier selection ────────────────────────────────────────────────────────

    def _select_agent(self, task_type: str, difficulty: str, force_tier3: bool = False):
        """Pick the right tiered (or legacy) agent for the task & difficulty."""
        if task_type == "vision":
            return AGENTS["vision"], "vision"

        if not USE_TIERED or force_tier3:
            return AGENTS.get(task_type, AGENTS["general"]), "tier3"

        tier = "tier3" if force_tier3 else get_tier_for_difficulty(difficulty)
        
        # Use model registry for smart tier persistence
        if task_type in TIERED_MODELS:
            if task_type == "general":
                # Use tiered general agent for proper preference handling
                return _model_registry.get_or_create_agent("general", tier, force_tier3)
            return _model_registry.get_or_create_agent(task_type, tier, force_tier3)
        
        return make_tiered_agent(task_type, tier), tier

    # ── Validation + escalation ───────────────────────────────────────────────

    def _validate_coding(
        self,
        query: str,
        response: str,
        task_type: str,
        difficulty: str,
        dataset_context: str,
        verbose: bool,
    ) -> tuple[str, bool]:
        """Validate coding responses. Returns (final_response, escalated_flag)."""
        result = validate_code_response(response)

        if not result["has_code"]:
            if verbose:
                print("[code_validator] No executable Python code found, skipping.\n")
            return response, False

        if result["passed"]:
            if verbose:
                print(f"[code_validator] ✅ Passed → {result['output'][:200]}\n")
            return response, False

        if verbose:
            print(f"[code_validator] ❌ Failed:\n  {result['output']}\n")

        if not ESCALATE_ON_FAIL:
            if not USE_DEEP_VAL:
                return response, False
            # Deep validator repair (no tier escalation)
            if verbose:
                print("[deep_validator] Repairing...\n")
            fix_prompt = (
                f"Original query: {query}\n\nAgent response:\n{response}\n\n"
                f"The code failed with:\n{result['output']}\n\n"
                "Repair only the failing part or the minimal corrected variant, not the whole answer unless necessary. "
                "If the code is otherwise correct, keep it and only fix the broken section. "
                "Reply with REPAIR: <minimal corrected version>."
            )
            fixed = AGENTS["deep_validator"].run(fix_prompt, stream=False)
            for prefix in ("REPAIR:", "REVISED:", "VALID:"):
                if fixed.startswith(prefix):
                    return fixed.split(prefix, 1)[1].strip(), False
            return fixed, False

        # ── Tier escalation ──────────────────────────────────────────────────
        if verbose:
            print("[tiered] Escalating coding agent → tier3 for re-generation\n")
        agent, tier = self._select_agent("coding", difficulty, force_tier3=True)
        if verbose:
            print(f"[{task_type}/{tier}] ", end="", flush=True)

        retry_query = (
            f"{query}\n\n"
            f"IMPORTANT context: My previous code attempt failed with this error:\n"
            f"{result['output']}\n\n"
            f"Please produce a completely correct, working solution. "
            f"Here are some reference examples:\n{dataset_context}"
        )
        # For coding validation, we don't pass the full context to avoid confusion
        retry_context = dataset_context
        response2 = agent.run(
            retry_query,
            context=retry_context,
            stream=verbose,
            history=self._conversation_history,
        )
        if verbose:
            print()

        # Validate the re-attempt once more (subprocess only, no further escalation)
        result2 = validate_code_response(response2)
        if result2["passed"]:
            if verbose:
                print(f"[code_validator] ✅ Tier3 passed → {result2['output'][:200]}\n")
            return response2, True

        if verbose:
            print(f"[code_validator] ❌ Tier3 still failed. Returning best attempt.\n")
        return response2, True

    def _validate_general(
        self,
        query: str,
        response: str,
        task_type: str,
        difficulty: str,
        dataset_context: str,
        verbose: bool,
        context: str = "",
    ) -> tuple[str, bool]:
        """Validate non-coding responses. Returns (final_response, escalated_flag)."""
        # Check if memory context was provided (look for Q: format which indicates memory)
        memory_was_used = context and ("Q:" in context and "A:" in context)
        context_note = (
            "\n\nNOTE: This response was generated WITH context from previous conversations (memory). "
            "The model is SUPPOSED to use information from past conversations like names, preferences, etc."
            if memory_was_used else ""
        )

        cheap_input = f"Original query: {query}\n\nAgent response:\n{response}{context_note}"

        if verbose:
            print("[cheap_validator] ", end="", flush=True)

        cheap_result = AGENTS["cheap_validator"].run(cheap_input, stream=False)

        # Minimal, targeted validation: keep the original unless the validator 
        # explicitly flags a real issue and gives a repair.
        status = "PASS"
        issues = "none"
        repair = ""

        lines = cheap_result.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.upper().startswith("STATUS:"):
                val = stripped.split(":", 1)[1].strip().upper()
                if val in ("PASS", "FAIL"):
                    status = val
            elif stripped.upper().startswith("ISSUES:"):
                issues = stripped.split(":", 1)[1].strip()
            elif stripped.upper().startswith("REPAIR:"):
                repair = stripped.split(":", 1)[1].strip()

        if verbose:
            print(f"  → status: {status} | issues: {issues}\n")

        # If the validator says PASS, do not keep a non-empty "none" issue list
        # as an error signal. This keeps the logger clean and matches the model's intent.
        if status == "PASS" and issues.lower() == "none":
            issues = "none"

        if status == "PASS":
            return response, False

        # If the validator failed, use only the minimal repair it provided.
        # Reject identical echoes: the validator must produce a materially different fix.
        if repair and repair.strip() != response.strip():
            if verbose:
                print(f"[cheap_validator] ✏ Applying minimal repair ({len(repair)} chars)\n")
            return repair, False

        # repair was empty or identical to the original answer: keep original
        if not ESCALATE_ON_FAIL:
            if not USE_DEEP_VAL:
                return response, False
            if verbose:
                print("[deep_validator] No materially different repair produced; keeping original answer.\n")
            return response, False

        # ── FAIL without repair: escalate or deep-validate ─────────────────────
        if not ESCALATE_ON_FAIL:
            if not USE_DEEP_VAL:
                return response, False
            if verbose:
                print("[deep_validator] Checking targeted repair...\n")
            deep_input = (
                f"Original query: {query}\n\nAgent response:\n{response}\n\n"
                f"Quick check flagged: {issues}"
            )
            dr = AGENTS["deep_validator"].run(deep_input, stream=False)

            dr_status = "PASS"
            dr_repair = ""
            for line in dr.splitlines():
                stripped = line.strip()
                if stripped.upper().startswith("STATUS:"):
                    val = stripped.split(":", 1)[1].strip().upper()
                    if val in ("PASS", "FAIL"):
                        dr_status = val
                elif stripped.upper().startswith("REPAIR:"):
                    dr_repair = stripped.split(":", 1)[1].strip()

            if dr_status == "PASS":
                return response, False
            if dr_repair and dr_repair.strip() != response.strip():
                return dr_repair, False
            # Fallback for old format compatibility
            for prefix in ("REVISED:", "VALID:"):
                if dr.startswith(prefix):
                    candidate = dr.split(prefix, 1)[1].strip()
                    if candidate.strip() != response.strip():
                        return candidate, False
                    return response, False
            return response, False


        # ── Tier escalation ──────────────────────────────────────────────────
        if verbose:
            print(f"[tiered] Escalating {task_type} agent → tier3 for re-generation\n")
        agent, tier = self._select_agent(task_type, difficulty, force_tier3=True)
        if verbose:
            print(f"[{task_type}/{tier}] ", end="", flush=True)

        retry_query = (
            f"{query}\n\n"
            f"IMPORTANT context: A previous attempt had these issues:\n"
            f"{issues}\n\n"
            f"Please produce a correct, complete response. "
            f"Reference examples:\n{dataset_context}"
        )
        # Include the original context in the retry
        response2 = agent.run(
            retry_query,
            context=dataset_context + (f"\n\n{context}" if context else ""),
            stream=verbose,
            history=self._conversation_history,
        )
        if verbose:
            print()
        return response2, True

    # ── Main handler ──────────────────────────────────────────────────────────

    def handle(
        self,
        query: str,
        verbose: bool = True,
        override_image: str | None = None,
    ) -> dict:
        image_path, query, err = extract_image_info(query, override_image=override_image)
        if err and not image_path:
            if verbose:
                print(f"[system] {err}\n")
            return {
                "task_type":  "vision",
                "difficulty": "easy",
                "tier":       "vision",
                "escalated":  False,
                "image_path": None,
                "response":   err,
            }

        original_query = query
        visual_context = ""
        if image_path:
            visual_context = self._extract_visual_content(image_path, original_query)
            if not visual_context or visual_context.startswith("[ERROR"):
                return {
                    "task_type":  "general",
                    "difficulty": "easy",
                    "tier":       "vision",
                    "escalated":  False,
                    "image_path": image_path,
                    "response":   visual_context or "The image content could not be extracted.",
                }
            # The extracted text/observations are now treated as normal input.
            # This deliberately prevents the vision model from answering or
            # locking the request into the vision category.
            query = (
                f"User request:\n{original_query}\n\n"
                f"Extracted image content:\n{visual_context}"
            )

        # ── Preference update / inspection intercept ──────────────────────────
        # Universal intercept for all domains (coding, writing, math, general)
        # Run the deterministic parser first; it catches explicit statements like
        # 'i prefer recursion over iteration' before model-based parsing can fail.
        pref_update = self._detect_preference_update(query)
        if pref_update:
            if pref_update["type"] == "SHOW":
                requested_domain = pref_update.get("domain", "all")
                visible_preferences = {
                    key: value for key, value in self._active_preferences.items()
                    if requested_domain == "all"
                    or value.get("domain") in (requested_domain, "all")
                }
                if not visible_preferences:
                    scope = " across all domains" if requested_domain == "all" else f" for {requested_domain}"
                    msg = f"You currently have no active preferences{scope}."
                else:
                    scope = "across all domains" if requested_domain == "all" else f"for {requested_domain}"
                    lines = [f"Here are your active preferences {scope}:"]
                    for k, p in visible_preferences.items():
                        dom = p.get("domain", "all").upper()
                        lines.append(f"  • **{k}** [{dom}]: {p.get('instruction', '')}")
                    msg = "\n".join(lines)
                self._last_embedding = None
                self._last_task_type = None
                if verbose:
                    print(f"\n{msg}\n")
                return {
                    "task_type":  "general",
                    "difficulty": "easy",
                    "tier":       "preference",
                    "escalated":  False,
                    "image_path": None,
                    "response":   msg,
                }

            key = pref_update["key"]
            domain = pref_update["domain"]
            action = pref_update["type"]
            conf = pref_update["confirmation"]
            instruction = pref_update.get("instruction", "")

            if action == "REMOVE" or not instruction:
                # Remove matching preference key
                matched_keys = [k for k in self._active_preferences if k == key or key in k or k in key]
                if not matched_keys:
                    matched_keys = [key]
                for k in matched_keys:
                    self._active_preferences.pop(k, None)
                    if self.use_memory:
                        self.memory.remove_by_metadata({"type": "preference_update", "pref_key": k})
                conf_msg = conf or f"Preference '{key}' has been removed."
            else:
                updated_at = time.time()
                new_preference = {
                    "domain": domain,
                    "instruction": instruction,
                    "confirmation": conf,
                    "updated_at": updated_at,
                }
                removed_keys = self._apply_preference(key, new_preference, updated_at)
                if self.use_memory:
                    self.memory.remove_by_metadata({"type": "preference_update", "pref_key": key})
                    for removed_key in removed_keys:
                        self.memory.remove_by_metadata({
                            "type": "preference_update",
                            "pref_key": removed_key,
                        })
                    self.memory.add(
                        f"USER PREFERENCE [{domain.upper()}]: {instruction}\nRequest: {query}",
                        metadata={
                            "type": "preference_update",
                            "pref_key": key,
                            "domain": domain,
                            "instruction": instruction,
                            "confirmation": conf,
                            "updated_at": updated_at,
                        }
                    )
                self._last_embedding = None
                self._last_task_type = None
                conf_msg = conf or f"Preference '{key}' updated for {domain}."

            if verbose:
                print(f"[preference] {action} {key} [{domain}]: '{instruction}'")
                print(f"\n{conf_msg}\n")

            return {
                "task_type":  "general",
                "difficulty": "easy",
                "tier":       "preference",
                "escalated":  False,
                "image_path": None,
                "response":   conf_msg,
            }

        query_emb = self._embed(query)

        # ── Refresh model registry state ─────────────────────────────────────
        # Sync with actual Ollama state to handle external unloads
        _model_registry.refresh_state()

        # ── Classification (domain) ──────────────────────────────────────────
        task_type, method, ds_res = self.classify(
            query,
            query_emb,
            None,
            from_image_extraction=bool(image_path),
        )
        if verbose:
            print(f"[{method}] → {task_type}")
        
        # ── Difficulty estimation (for tiered routing) ───────────────────────
        # Priority 1: dataset kNN difficulty (cheapest, from Math + Leetcode labels)
        # Priority 2: LLM difficulty estimator (fallback when kNN confidence is low)
        difficulty = "medium"
        rationale  = ""
        diff_source = "fallback"
        if task_type != "vision":
            if ds_res is not None and task_type in ("coding", "math"):
                # Re-estimate difficulty constrained by the predicted task (for purer neighbour votes)
                ds_diff, ds_diff_conf, ds_diff_counts = self._dataset_router.estimate_difficulty(
                    query_emb, task_label=task_type
                )
                if ds_diff_conf >= KNN_DIFF_THRESHOLD:
                    difficulty = ds_diff
                    parts = [f"{l}={n}" for l, n in sorted(ds_diff_counts.items(), key=lambda x: -x[1])]
                    rationale = f"dataset kNN task-matched neighbours: {', '.join(parts)}"
                    diff_source = "dataset-knn"
                else:
                    # Try unconstrained
                    ds_diff2 = ds_res.get("diff", "medium")
                    ds_diff2_conf = ds_res.get("diff_conf", 0.0)
                    if ds_diff2_conf >= KNN_DIFF_THRESHOLD:
                        difficulty = ds_diff2
                        parts = [f"{l}={n}" for l, n in sorted(ds_res.get("top_diff_counts").items(), key=lambda x: -x[1])]
                        rationale = f"dataset kNN all neighbours: {', '.join(parts)}"
                        diff_source = "dataset-knn-unconstrained"

            if diff_source == "fallback" and task_type in ("general", "writing"):
                # Conversational requests do not benefit from a second LLM call
                # just to choose a tier; keep them on the fast path.
                difficulty, rationale = "easy", "conversation fast path"
                diff_source = "fast-path"
            elif diff_source == "fallback":
                difficulty, rationale = estimate_difficulty(query, task_type)
                diff_source = "llm"

            if verbose:
                print(f"[difficulty] {difficulty} ({diff_source}) — {rationale}")

        # ── Context assembly ─────────────────────────────────────────────────
        context_parts: list[str] = []

        # Put memory context LAST so it's most prominent
        ds_ctx = build_dataset_context(query, task_type, difficulty=difficulty, n_examples=2)
        if ds_ctx:
            context_parts.append(ds_ctx)
            if verbose:
                print(f"[dataset] local examples injected")

        if task_type == "general":
            hf_ctx = get_hf_general_context(query, n=2)
            if hf_ctx:
                context_parts.append(hf_ctx)
                if verbose:
                    print(f"[hf-magpie] similar conversations injected")

        mem_ctx = ""
        if self.use_memory and _needs_long_term_memory(query):
            mem_ctx = self.memory.query(query)
            if mem_ctx:
                context_parts.append(mem_ctx)
                if verbose:
                    print(f"[memory] {len(mem_ctx)} chars of context retrieved")

        if verbose:
            print()
        context = "\n\n".join(context_parts)

        # ── Universal Preference injection across ALL domains ──────────────────
        if self._active_preferences:
            applicable = [
                p["instruction"] for p in self._active_preferences.values()
                if p.get("instruction") and p.get("domain") in (task_type, "all")
            ]
            if applicable:
                pref_block = f"USER PREFERENCES FOR {task_type.upper()} (MANDATORY — obey these rules):\n" + \
                             "\n".join(f"  • {p}" for p in applicable)
                query = f"{pref_block}\n\nQuery: {query}"
                if verbose:
                    print(f"[preference] Injected {len(applicable)} {task_type} preference(s) into query")
                # When preferences are active, bump easy -> medium across all domains
                # so the agent has sufficient model capacity to follow instructions
                if difficulty == "easy":
                    difficulty = "medium"
                    if verbose:
                        print(f"[preference] Bumped difficulty easy → medium for preference adherence")

        # ── Agent selection & generation ─────────────────────────────────────
        agent, tier = self._select_agent(task_type, difficulty)
        self._last_embedding = query_emb
        self._last_task_type = task_type

        if verbose:
            # Check if model was already loaded before this request
            was_loaded = _model_registry.is_model_loaded(agent.model)
            status = "REUSING LOADED" if was_loaded else "LOADING"
            print(f"[{task_type}/{tier} ({agent.model}) - {status}] ", end="", flush=True)

        response = agent.run(
            query,
            context=context,
            stream=verbose,
            image_path=None,
            history=self._conversation_history,
        )
        if not verbose:
            print(f"[{task_type}/{tier}] {response}\n")
        else:
            print()

        # ── Validation (with tier escalation) ────────────────────────────────
        escalated = False
        if self.use_validator and task_type != "vision":
            if task_type == "coding":
                response, escalated = self._validate_coding(
                    query, response, task_type, difficulty, ds_ctx, verbose
                )
            else:
                response, escalated = self._validate_general(
                    query, response, task_type, difficulty, ds_ctx, verbose, context
                )

        # ── Store temporary session exchange ─────────────────────────────────
        # Normal conversations are intentionally ephemeral. Only explicit
        # preference updates above are written to persistent memory.
        self._conversation_history.extend([
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ])
        self._conversation_history = self._conversation_history[-(MAX_CHAT_TURNS * 2):]

        return {
            "task_type":  task_type,
            "difficulty": difficulty,
            "tier":       tier,
            "escalated":  escalated,
            "image_path": image_path,
            "response":   response,
        }


# ── CLI entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    orch = Orchestrator(use_validator=True)
    print("Multi-agent system ready. Commands:")
    print("  'exit' | 'quit'    → quit")
    print("  'image <path>'     → analyse an image / screenshot")
    print("  'diff'             → show last difficulty/tier breakdown\n")

    last_meta = {}

    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "diff" and last_meta:
            print(f"  last: {last_meta}")
            continue

        img, prompt, err = extract_image_info(q)
        if err and not img:
            print(f"[system] {err}\n")
            continue

        result = orch.handle(prompt, override_image=img)
        last_meta = {k: v for k, v in result.items() if k != "response"}

        header = f"Final Answer ({result['task_type']} | {result['difficulty']} | {result['tier']}"
        if result["escalated"]:
            header += " | ESCALATED"
        header += ")"
        print(f"\n{'='*60}\n{header}\n{'='*60}\n{result['response']}\n")
