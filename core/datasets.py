"""
core/datasets.py
────────────────
Local dataset loaders + Hugging Face dataset integration +
DatasetRouter (embedding-based kNN task classification + difficulty estimation
using the local Math and Leetcode datasets as semantic anchors).

Supported:
- Math Questions.json    → list[{question, answer, difficulty, category}]
- Leetcode Questions.csv → list[{Question_No, Question, Acceptance, Difficulty, ...}]
- Hugging Face datasets  → lazy-loaded, cached in memory
"""

import json
import csv
import os
import random
from collections import Counter
from typing import Any
import numpy as np
from config import DATASET_PATHS, DEFAULT_HF_DATASET


# ── Local dataset loaders ─────────────────────────────────────────────────────

def load_math_dataset(path: str = DATASET_PATHS["math"]) -> list[dict]:
    """Load Math Questions.json. Returns list of dicts with keys:
    question, answer, difficulty, category."""
    with open(path) as f:
        return json.load(f)


def load_leetcode_dataset(path: str = DATASET_PATHS["leetcode"]) -> list[dict]:
    """Load Leetcode Questions.csv. Returns list of dicts keyed by header row."""
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({k.strip(): v.strip() for k, v in row.items()})
    return rows


# ── Dataset cache ─────────────────────────────────────────────────────────────

_DATASET_CACHE: dict[str, Any] = {}


def get_dataset(name: str, force_reload: bool = False) -> Any:
    """Get a dataset by name, caching in memory. Names: 'math', 'leetcode',
    or any Hugging Face dataset identifier like 'Magpie-Align/Magpie-Air-300K-Filtered'."""
    if name in _DATASET_CACHE and not force_reload:
        return _DATASET_CACHE[name]

    if name == "math":
        data = load_math_dataset()
    elif name == "leetcode":
        data = load_leetcode_dataset()
    else:
        data = _load_hf_dataset(name)

    _DATASET_CACHE[name] = data
    return data


# ── Hugging Face dataset loader (lazy import to avoid hard dep) ───────────────

def _load_hf_dataset(name: str, split: str = "train") -> Any:
    """Load a Hugging Face datasets.Dataset object (cached by HF automatically)."""
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "Hugging Face 'datasets' package not installed. "
            "Run: pip install datasets --break-system-packages"
        ) from e
    return load_dataset(name, split=split)


# ── Search / lookup helpers ───────────────────────────────────────────────────

def search_math(query: str, n: int = 5) -> list[dict]:
    """Simple substring search over math dataset questions + categories."""
    ds = get_dataset("math")
    ql = query.lower()
    scored = []
    for item in ds:
        text = (item.get("question", "") + " " + item.get("category", "")).lower()
        if ql in text:
            scored.append(item)
        if len(scored) >= n * 10:
            break
    return scored[:n]


def search_leetcode(query: str, difficulty: str | None = None, n: int = 5) -> list[dict]:
    """Simple substring search over Leetcode questions, optionally filtered by
    difficulty ('Easy', 'Medium', 'Hard')."""
    ds = get_dataset("leetcode")
    ql = query.lower()
    results = []
    for item in ds:
        q = item.get("Question", "").lower()
        d = item.get("Difficulty", "").strip()
        if difficulty and d.lower() != difficulty.lower():
            continue
        if ql in q or ql in d.lower():
            results.append(item)
        if len(results) >= n:
            break
    return results


def random_math(difficulty: str | None = None, category: str | None = None) -> dict:
    """Pick a random math question, optionally filtered by difficulty/category."""
    import random
    ds = get_dataset("math")
    pool = [
        x for x in ds
        if (difficulty is None or x.get("difficulty") == difficulty)
        and (category is None or category.lower() in x.get("category", "").lower())
    ]
    return random.choice(pool) if pool else {}


def random_leetcode(difficulty: str | None = None) -> dict:
    """Pick a random Leetcode question, optionally filtered by difficulty."""
    import random
    ds = get_dataset("leetcode")
    pool = ds if difficulty is None else [
        x for x in ds if x.get("Difficulty", "").strip().lower() == difficulty.lower()
    ]
    return random.choice(pool) if pool else {}


# ── Build memory-insertable context strings ───────────────────────────────────

def build_dataset_context(
    query: str,
    task_type: str,
    difficulty: str | None = None,
    n_examples: int = 2,
) -> str:
    """
    Retrieve relevant examples from local datasets as RAG context for agents.
    Returns a formatted string suitable for injecting into the agent system prompt.
    """
    parts = []

    if task_type in ("math", "general"):
        examples = search_math(query, n=n_examples)
        if examples:
            parts.append("Relevant math dataset examples:")
            for i, ex in enumerate(examples, 1):
                parts.append(
                    f"  {i}. [{ex.get('difficulty','?')} / {ex.get('category','?')}] "
                    f"Q: {ex.get('question','')}\n     A: {ex.get('answer','')}"
                )

    if task_type in ("coding", "general"):
        diff_map = {"easy": "Easy", "medium": "Medium", "hard": "Hard"}
        lc_diff = diff_map.get(difficulty) if difficulty else None
        examples = search_leetcode(query, difficulty=lc_diff, n=n_examples)
        if examples:
            parts.append("Relevant Leetcode dataset examples:")
            for i, ex in enumerate(examples, 1):
                parts.append(
                    f"  {i}. [#{ex.get('Question_No','?')} / {ex.get('Difficulty','?')}] "
                    f"{ex.get('Question','')[:120]}"
                )

    return "\n".join(parts)


# ── Hugging Face Magpie dataset context (general QA) ──────────────────────────

_HF_DATASET_SAMPLE_CACHE: list[dict] = []


def get_hf_general_context(query: str, n: int = 2) -> str:
    """Pull similar conversations from the default HF Magpie dataset and format as context."""
    global _HF_DATASET_SAMPLE_CACHE
    if not _HF_DATASET_SAMPLE_CACHE:
        try:
            ds = _load_hf_dataset(DEFAULT_HF_DATASET, split="train[:200]")
            _HF_DATASET_SAMPLE_CACHE = [
                {"conversations": item["conversations"], "uuid": item["uuid"]}
                for item in ds
            ]
        except Exception:
            return ""

    ql = query.lower()
    matches = []
    for item in _HF_DATASET_SAMPLE_CACHE:
        conv_text = " ".join(
            m.get("value", "") for m in item["conversations"]
        ).lower()
        if any(w in conv_text for w in ql.split() if len(w) > 3):
            matches.append(item)
        if len(matches) >= n:
            break

    if not matches:
        return ""

    lines = ["Similar Magpie conversations (for reference):"]
    for i, item in enumerate(matches, 1):
        turns = item["conversations"]
        human = next((m["value"] for m in turns if m.get("from") == "human"), "")
        gpt = next((m["value"] for m in turns if m.get("from") == "gpt"), "")
        lines.append(f"  {i}. Q: {human[:150]}")
        lines.append(f"     A: {gpt[:200]}")
    return "\n".join(lines)


# ── DatasetRouter: kNN routing + difficulty via dataset embeddings ────────────

_LEETCODE_DIFFICULTY_MAP = {"Easy": "easy", "Medium": "medium", "Hard": "hard"}
_MATH_DIFFICULTY_MAP = {
    "easy": "easy", "medium": "medium", "hard": "hard",
    "Easy": "easy", "Medium": "medium", "Hard": "hard",
}


def _normalize_difficulty(value: str | None) -> str | None:
    return (value or "").strip().lower() or None


class DatasetRouter:
    """
    Embeds stratified samples from the local Math (10K) and Leetcode (2.9K)
    datasets and uses k-nearest-neighbours cosine similarity to:
      1. Classify a query as "math" vs "coding" vs "ambiguous"
      2. Estimate difficulty (easy/medium/hard) from the k-neighbours' labels
    Stratification keeps classes balanced so the router doesn't bias towards
    the most common difficulty in each dataset.
    """

    samples_per_difficulty = 250  # per (dataset × difficulty) bucket → ~1,500 anchors total
    k_neighbours = 7

    # Absolute semantic relevance gate: if the single nearest dataset anchor has
    # cosine similarity below this, the query is considered "outside dataset domains"
    # (i.e. not a math / coding question) and classify_task returns None / ambiguous.
    # Prevents 7/7 "coding" majority votes on writing/general queries that just happen
    # to sit nearest to Leetcode anchors because no other label anchors exist.
    domain_relevance_cos_gate = 0.35

    def __init__(self, embedder):
        self.embedder = embedder
        self._texts: list[str] = []
        self._tasks: list[str] = []
        self._diffs: list[str] = []
        self._embs: np.ndarray | None = None
        self._built = False

    def build(self, seed: int = 42) -> None:
        """Build the anchor index from Math + Leetcode datasets. Idempotent."""
        if self._built:
            return
        random.seed(seed)

        anchors: list[tuple[str, str, str]] = []  # (text, task, difficulty)

        # ── Math dataset: (question + " " + category) → task=math ──────────
        try:
            math = get_dataset("math")
            buckets = {"easy": [], "medium": [], "hard": []}
            for item in math:
                d = _MATH_DIFFICULTY_MAP.get((item.get("difficulty") or "").strip())
                if d not in buckets:
                    continue
                text = (item.get("question", "") + " " + item.get("category", "")).strip()
                if text:
                    buckets[d].append(text)
            for diff, texts in buckets.items():
                take = min(self.samples_per_difficulty, len(texts))
                for t in random.sample(texts, take):
                    anchors.append((t, "math", diff))
        except Exception:
            pass

        # ── Leetcode dataset: Question → task=coding ────────────────────────
        try:
            lc = get_dataset("leetcode")
            buckets = {"easy": [], "medium": [], "hard": []}
            for item in lc:
                d = _LEETCODE_DIFFICULTY_MAP.get((item.get("Difficulty") or "").strip())
                if d not in buckets:
                    continue
                text = (item.get("Question", "") or "").strip()
                if text:
                    buckets[d].append(text)
            for diff, texts in buckets.items():
                take = min(self.samples_per_difficulty, len(texts))
                for t in random.sample(texts, take):
                    anchors.append((t, "coding", diff))
        except Exception:
            pass

        if not anchors:
            self._built = True
            return

        self._texts = [a[0] for a in anchors]
        self._tasks = [a[1] for a in anchors]
        self._diffs = [a[2] for a in anchors]

        # Precompute L2-normalized embeddings → cosine = dot product
        raw = self.embedder.encode(self._texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(raw, axis=1, keepdims=True) + 1e-10
        self._embs = raw / norms
        self._built = True

    def _knn(self, query_emb: np.ndarray):
        """Return (task_votes, diff_votes, neighbor_tasks, neighbor_diffs, max_cosine)."""
        if self._embs is None:
            return None
        q = query_emb / (np.linalg.norm(query_emb) + 1e-10)
        sims = self._embs @ q
        top_k = min(self.k_neighbours, len(sims))
        idx = np.argpartition(-sims, top_k - 1)[:top_k]
        order = idx[np.argsort(-sims[idx])]
        neighbor_tasks = [self._tasks[i] for i in order]
        neighbor_diffs = [self._diffs[i] for i in order]
        top_cosines = sims[order]
        return neighbor_tasks, neighbor_diffs, top_cosines

    def classify_task(self, query_emb: np.ndarray):
        """
        Returns (task_label, confidence, votes_counter).
        task_label ∈ {"math", "coding", None} — None = dataset vote is not decisive.
        Confidence ∈ [0,1] = fraction of k neighbours agreeing on the winning task.
        Gate: even with a confident kNN majority, if the absolute nearest anchor has
        cosine similarity < domain_relevance_cos_gate, return None (query is outside
        the dataset domains, i.e. is general/writing/vision not math/coding).
        """
        res = self._knn(query_emb)
        if res is None:
            return None, 0.0, Counter()
        tasks, _, cosines = res
        if float(np.max(cosines)) < self.domain_relevance_cos_gate:
            counts = Counter(tasks)
            (label, n), = counts.most_common(1)
            return None, n / len(tasks), counts
        counts = Counter(tasks)
        (label, n), = counts.most_common(1)
        confidence = n / len(tasks)
        if confidence < 0.5:
            return None, confidence, counts
        return label, confidence, counts

    def estimate_difficulty(self, query_emb: np.ndarray, task_label: str | None = None):
        """
        Majority-vote difficulty over the k-nearest dataset neighbours.
        Optionally constrain to neighbours whose task matches task_label.
        Returns (difficulty ∈ {easy,medium,hard}, confidence, votes_counter).
        Falls back to "medium" / confidence 0.
        """
        res = self._knn(query_emb)
        if res is None:
            return "medium", 0.0, Counter()
        tasks, diffs, _ = res
        if task_label is not None:
            filtered = [d for t, d in zip(tasks, diffs) if t == task_label]
            if not filtered:
                filtered = diffs
            diffs_to_count = filtered
        else:
            diffs_to_count = diffs
        counts = Counter(diffs_to_count)
        (label, n), = counts.most_common(1)
        confidence = n / len(diffs_to_count)
        return label, confidence, counts

    def analyse(self, query_emb: np.ndarray, verbose: bool = False) -> dict:
        """Full kNN analysis dict used by orchestrator.
        Uses classify_task() so the domain-relevance cosine-gate AND the majority
        gate are both applied — prevents spurious 7/7 "coding" votes on queries
        that don't actually resemble any math/coding dataset anchor."""
        res = self._knn(query_emb)
        if res is None:
            return {"task": None, "task_conf": 0.0, "diff": "medium", "diff_conf": 0.0, "top_task_counts": Counter(), "top_diff_counts": Counter(), "ready": False}
        tasks, diffs, cosines = res
        task_counts = Counter(tasks)
        diff_counts = Counter(diffs)
        (diff_label, dn), = diff_counts.most_common(1)
        diff_conf = dn / len(diffs)

        gated_task_label, gated_task_conf, _ = self.classify_task(query_emb)
        raw_top = task_counts.most_common(1)
        raw_task_conf = raw_top[0][1] / len(tasks) if raw_top else 0.0
        out = {
            "task": gated_task_label,
            "task_conf": gated_task_conf if gated_task_label is not None else raw_task_conf,
            "diff": diff_label,
            "diff_conf": diff_conf,
            "top_task_counts": task_counts,
            "top_diff_counts": diff_counts,
            "top_cosine_avg": float(np.mean(cosines)),
            "top_cosine_max": float(np.max(cosines)),
            "ready": True,
        }
        return out
