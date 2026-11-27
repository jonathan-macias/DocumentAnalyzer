"""SEMANTIC index of the document catalog (persistent Chroma).

Classification does not need hybrid search: the catalog holds a handful of descriptions and
what matters is MEANING ("SAT certificate with RFC and tax regime" ≈ CONSTANCIA_SITUACION_FISCAL).
It is used as a shortlist of candidates that the LLM then confirms (retrieve → decide).

The index is rebuilt only when the hash of the YAML definitions changes (avoids re-embedding
on every startup).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.catalog.registry import CatalogRegistry

logger = logging.getLogger(__name__)


@dataclass
class CatalogCandidate:
    document_type: str
    category: str
    similarity: float


class CatalogIndex:
    def __init__(
        self, registry: CatalogRegistry, embeddings: Embeddings, persist_dir: Path, collection: str
    ) -> None:
        self.registry = registry
        self.persist_dir = persist_dir
        self.collection = collection
        self.embeddings = embeddings
        self.vectorstore: Chroma | None = None

    def _hash_file(self) -> Path:
        return self.persist_dir / f"{self.collection}.hash"

    def build(self) -> None:
        """Create/update the persistent collection (called from the FastAPI lifespan)."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore = Chroma(
            collection_name=self.collection,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
            collection_metadata={"hnsw:space": "cosine"},
        )
        hash_file = self._hash_file()
        current = hash_file.read_text().strip() if hash_file.exists() else ""
        if current == self.registry.fingerprint and self.vectorstore._collection.count() > 0:
            logger.info("Catalog index is up to date (%s)", current)
            return

        logger.info("Rebuilding catalog semantic index…")
        self.vectorstore.reset_collection()
        docs = [
            Document(
                page_content=t.embedding_text(),
                metadata={"document_type": t.type, "category": t.category},
            )
            for t in self.registry.types.values()
        ]
        self.vectorstore.add_documents(docs, ids=[d.metadata["document_type"] for d in docs])
        hash_file.write_text(self.registry.fingerprint)

    async def asearch(self, text: str, k: int = 4) -> list[CatalogCandidate]:
        if self.vectorstore is None:
            raise RuntimeError("CatalogIndex.build() has not been called.")
        # Chroma returns cosine DISTANCE (0 = identical, 2 = opposite) → similarity = 1 - distance
        results = await self.vectorstore.asimilarity_search_with_score(text, k=k)
        return [
            CatalogCandidate(
                document_type=doc.metadata["document_type"],
                category=doc.metadata["category"],
                similarity=round(1.0 - float(distance), 4),
            )
            for doc, distance in results
        ]
