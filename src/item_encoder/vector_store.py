"""
Vector Store for Item Embeddings.

Uses ChromaDB to store and retrieve movie embeddings.
Supports nearest-neighbor search for collaborative and content-based signals.
"""

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings
import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-based vector store for movie item embeddings.
    """

    def __init__(self, persist_dir: str, collection_name: str = "items"):
        """
        Args:
            persist_dir: Directory for ChromaDB persistent storage.
            collection_name: Name of the ChromaDB collection.
        """
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "VectorStore initialized at '%s', collection='%s' (%d items)",
            persist_dir, collection_name, self.collection.count(),
        )

    def add_items(
        self,
        movie_ids: list,
        embeddings: dict,
        metadata: dict = None,
    ):
        """
        Add movie embeddings to the vector store.

        Args:
            movie_ids: List of movie IDs.
            embeddings: dict mapping movie_id → numpy array.
            metadata: Optional dict mapping movie_id → metadata dict.
        """
        ids = [str(mid) for mid in movie_ids]
        embs = [embeddings[mid].tolist() for mid in movie_ids]
        metas = [metadata.get(mid, {}) for mid in movie_ids] if metadata else None

        self.collection.upsert(
            ids=ids,
            embeddings=embs,
            metadatas=metas,
        )
        logger.info("Added/updated %d items in vector store", len(ids))

    def query_similar(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        exclude_id: str = None,
    ) -> list:
        """
        Find k most similar items to a query embedding.

        Args:
            query_embedding: Query vector (numpy array).
            k: Number of results.
            exclude_id: Optional ID to exclude from results (self-match).

        Returns:
            List of dicts: [{"id": str, "distance": float, "metadata": dict}, ...]
        """
        n_results = k + 1 if exclude_id else k

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, self.collection.count()),
        )

        items = []
        for i in range(len(results["ids"][0])):
            item_id = results["ids"][0][i]
            if exclude_id and item_id == str(exclude_id):
                continue
            items.append({
                "id": int(item_id),
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
            })

        return items[:k]

    def get_embedding(self, movie_id: int) -> np.ndarray:
        """
        Retrieve the stored embedding for a specific movie.

        Returns:
            numpy array or None if not found.
        """
        result = self.collection.get(ids=[str(movie_id)], include=["embeddings"])
        if result["embeddings"]:
            return np.array(result["embeddings"][0])
        return None

    def get_all_embeddings(self) -> dict:
        """
        Retrieve all stored embeddings.

        Returns:
            dict mapping movie_id (int) → numpy array.
        """
        result = self.collection.get(include=["embeddings"])
        embeddings = {}
        for id_str, emb in zip(result["ids"], result["embeddings"]):
            embeddings[int(id_str)] = np.array(emb)
        return embeddings

    @property
    def count(self) -> int:
        return self.collection.count()
