"""Per-document HYBRID retrieval (BM25 + embeddings).

    EnsembleRetriever(retrievers=[bm25, semantic], weights=[0.4, 0.6])

- BM25 (lexical): finds exact labels and codes → "RFC", "CURP", "Código Postal", "63067".
- Embeddings (semantic): finds paraphrases → "calle" ≈ "Nombre de Vialidad".
- EnsembleRetriever merges both rankings with weighted Reciprocal Rank Fusion (RRF).

The vector index is an EPHEMERAL (in-memory) Chroma collection per task: `task-<id>`.
It is deleted once the analysis finishes.
"""

import re
import unicodedata

import chromadb
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def bm25_preprocess(text: str) -> list[str]:
    """BM25 tokenizer: lowercase, accent-free, words/numbers only.

    Without it BM25Retriever falls back to `text.split()`, so "RFC:" != "rfc" and
    "Régimen" != "regimen".
    """
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"\w+", text)


class HybridIndex:
    """Wraps BM25 + ephemeral Chroma + EnsembleRetriever for a single document."""

    def __init__(
        self,
        client: chromadb.ClientAPI,
        collection_name: str,
        documents: list[Document],
        embeddings: Embeddings,
        k: int = 4,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
    ) -> None:
        self.collection_name = collection_name
        self._client = client
        self.bm25 = BM25Retriever.from_documents(
            documents, k=k, preprocess_func=bm25_preprocess
        )
        self.vectorstore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        self._documents = documents
        self.k = k
        self.retriever = EnsembleRetriever(
            retrievers=[self.bm25, self.vectorstore.as_retriever(search_kwargs={"k": k})],
            weights=[bm25_weight, vector_weight],
        )

    async def aindex(self) -> None:
        ids = [f"{self.collection_name}-{d.metadata['chunk']}" for d in self._documents]
        await self.vectorstore.aadd_documents(self._documents, ids=ids)

    async def aretrieve(self, query: str) -> list[Document]:
        return await self.retriever.ainvoke(query)

    def close(self) -> None:
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:  # noqa: BLE001 - best-effort cleanup
            pass
