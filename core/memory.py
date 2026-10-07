"""
core/memory.py
Shared vector memory using ChromaDB for retrieval-augmented context across agents.
"""

import os
import time
import uuid

# Suppress ChromaDB telemetry mismatch:
# "capture() takes 1 positional argument but 3 were given"
os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
try:
    from chromadb.telemetry.product.posthog import Posthog
    Posthog.capture = lambda self, event: None
except Exception:
    pass

import chromadb
from chromadb.utils import embedding_functions


class VectorMemory:
    def __init__(self, path: str = "./chroma_store", collection_name: str = "agent_memory"):
        self.client = chromadb.PersistentClient(path=path)

        # Lightweight local embedding model (runs on CPU to prevent Apple Silicon Metal OOM)
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu",
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedder,
        )

    def add(self, text: str, metadata: dict | None = None) -> None:
        """Store a piece of text (e.g. a query/response pair) in memory."""
        # Collection count is not a safe ID source after deletions or failed
        # inserts. Use a UUID so every memory write is independently unique.
        doc_id = f"doc_{uuid.uuid4().hex}"
        # Add timestamp to metadata for recency weighting
        if metadata is None:
            metadata = {}
        metadata["timestamp"] = time.time()
        metadata["doc_id"] = doc_id
        
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[doc_id],
        )

    def query(self, text: str, n_results: int = 5) -> str:
        """
        Retrieve the most relevant stored snippets for a given query.
        Uses hybrid search: semantic similarity + recency weighting.
        """
        if self.collection.count() == 0:
            return ""

        n_results = min(n_results, self.collection.count())
        
        # Get more results than needed for reranking
        results = self.collection.query(query_texts=[text], n_results=n_results * 2)

        docs = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        
        if not docs:
            return ""

        # Rerank by recency: boost recent documents
        current_time = time.time()
        scored_docs = []
        for doc, meta in zip(docs, metadatas):
            timestamp = meta.get("timestamp", 0)
            age_hours = (current_time - timestamp) / 3600 if timestamp > 0 else 9999
            
            # Recency boost: newer documents get higher scores
            # Age decay: multiply by 1/(1 + age_in_hours)
            recency_boost = 1.0 / (1.0 + age_hours * 0.1)  # Decay over hours
            
            # Also boost if it contains personal info (names, preferences)
            personal_boost = 1.5 if any(keyword in doc.lower() for keyword in 
                                       ["my name is", "i prefer", "i like", "my favorite"]) else 1.0
            
            final_score = recency_boost * personal_boost
            scored_docs.append((final_score, doc))
        
        # Sort by score and take top n_results
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for score, doc in scored_docs[:n_results]]
        
        return "\n---\n".join(top_docs)

    def get_by_metadata(self, where: dict) -> list[tuple[str, dict]]:
        """
        Fetch all documents matching a ChromaDB metadata filter.
        Returns list of (document_text, metadata) sorted by timestamp descending.
        Example: get_by_metadata({"type": "preference_update"})
        """
        if self.collection.count() == 0:
            return []
        try:
            results = self.collection.get(where=where, include=["documents", "metadatas"])
            docs      = results.get("documents", []) or []
            metadatas = results.get("metadatas", []) or []
            pairs = list(zip(docs, metadatas))
            # Sort newest first
            pairs.sort(key=lambda x: x[1].get("timestamp", 0), reverse=True)
            return pairs
        except Exception:
            return []
    
    def clear_memory(self) -> None:
        """Clear all stored memory (useful for resetting conversation state)."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embedder,
        )
    
    def remove_by_pattern(self, pattern: str) -> int:
        """Remove memory entries that contain a specific pattern (e.g., 'I don't know your name')."""
        if self.collection.count() == 0:
            return 0
        
        # Get all documents
        all_results = self.collection.get()
        docs = all_results.get("documents", [])
        ids = all_results.get("ids", [])
        
        # Find matching IDs
        ids_to_remove = []
        for doc, doc_id in zip(docs, ids):
            if pattern.lower() in doc.lower():
                ids_to_remove.append(doc_id)
        
        # Remove them
        if ids_to_remove:
            self.collection.delete(ids=ids_to_remove)
            print(f"[memory] Removed {len(ids_to_remove)} entries containing '{pattern}'")
        
        return len(ids_to_remove)

    def remove_by_metadata(self, where: dict) -> int:
        """Remove memory entries matching a ChromaDB where filter."""
        if self.collection.count() == 0:
            return 0
        try:
            results = self.collection.get(where=where)
            ids = results.get("ids", [])
            if ids:
                self.collection.delete(ids=ids)
                return len(ids)
        except Exception:
            pass
        return 0
