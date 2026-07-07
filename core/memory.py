"""
core/memory.py
Shared vector memory using ChromaDB for retrieval-augmented context across agents.
"""

import chromadb
from chromadb.utils import embedding_functions


class VectorMemory:
    def __init__(self, path: str = "./chroma_store", collection_name: str = "agent_memory"):
        self.client = chromadb.PersistentClient(path=path)

        # Lightweight local embedding model (downloads once, runs on CPU)
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedder,
        )
        self._counter = self.collection.count()

    def add(self, text: str, metadata: dict | None = None) -> None:
        """Store a piece of text (e.g. a query/response pair) in memory."""
        doc_id = f"doc_{self._counter}"
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id],
        )
        self._counter += 1

    def query(self, text: str, n_results: int = 3) -> str:
        """Retrieve the most relevant stored snippets for a given query."""
        if self.collection.count() == 0:
            return ""

        n_results = min(n_results, self.collection.count())
        results = self.collection.query(query_texts=[text], n_results=n_results)

        docs = results.get("documents", [[]])[0]
        if not docs:
            return ""

        return "\n---\n".join(docs)
