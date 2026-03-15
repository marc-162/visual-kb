import logging
import os

import chromadb

from src.constants import CHROMA_DB_DIR, CHROMA_DISTANCE_METRIC, COLLECTION_NAME
from src.embedder import embed_image, embed_text

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CHROMA_PATH = os.path.join(_PROJECT_ROOT, CHROMA_DB_DIR)


class VectorStore:
    """Thin wrapper around a ChromaDB collection for image embeddings."""

    def __init__(self, collection_name: str = COLLECTION_NAME) -> None:
        self.client = chromadb.PersistentClient(path=_CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": CHROMA_DISTANCE_METRIC},
        )

    def index_image(
        self,
        image_path: str,
        image_id: str,
        metadata: dict | None = None,
    ) -> None:
        """Embed and upsert a single image."""
        embedding = embed_image(image_path)
        self.collection.upsert(
            ids=[image_id],
            embeddings=[embedding],
            metadatas=[metadata or {"path": image_path}],
        )

    def search_by_text(self, query: str, n_results: int = 3) -> dict:
        """Return the *n_results* nearest neighbours for a text query."""
        query_embedding = embed_text(query)
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

    def search_by_image(self, image_path: str, n_results: int = 3) -> dict:
        """Return the *n_results* nearest neighbours for an image query."""
        query_embedding = embed_image(image_path)
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )

    def count(self) -> int:
        """Number of items in the collection."""
        return self.collection.count()

    def clear_collection(self) -> None:
        """Delete every document in the collection."""
        res = self.collection.get()
        ids = res["ids"]
        if ids:
            self.collection.delete(ids=ids)
