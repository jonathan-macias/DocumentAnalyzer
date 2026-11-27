"""Chunking for KYC documents.

Default strategy: "recursive".
1. Split by PAGE (no chunk crosses pages → metadata.page is reliable).
2. Within a page, split by SECTION (headings such as "Datos del domicilio registrado").
3. Each section is split with RecursiveCharacterTextSplitter measured in TOKENS (tiktoken).
4. Every chunk gets a cheap contextual header (no LLM involved):
       [Página 1 | Sección: Datos del domicilio registrado]
   A lightweight form of "contextual retrieval": the header helps both BM25 (section words)
   and embeddings (context).

Alternative strategy: "semantic" (SemanticChunker from langchain_experimental), for comparison.
It splits where the embedding distance between consecutive sentences jumps. On
"Label: value" forms it tends to split worse, costs one embedding per sentence and is not
deterministic.
"""

import re

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_HEADING_RE = re.compile(r"^[A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñü() ]{3,70}:?$")


def is_heading(line: str) -> bool:
    """Heuristic: short line with no digits/values that ends with ':' or starts with 'Datos'."""
    stripped = line.strip()
    if not stripped or "   " in stripped or not _HEADING_RE.match(stripped):
        return False
    return stripped.endswith(":") or stripped.lower().startswith("datos ")


def split_sections(page_text: str) -> list[tuple[str, str]]:
    """Return [(section_title, text)] preserving page order."""
    sections: list[tuple[str, list[str]]] = [("Encabezado", [])]
    for line in page_text.splitlines():
        if is_heading(line):
            sections.append((line.strip().rstrip(":"), [line]))
        else:
            sections[-1][1].append(line)

    # A "section" that only contains its own title is actually an empty label
    # (e.g. "Nombre Comercial:" with no value) → merge it into the previous section.
    merged: list[tuple[str, list[str]]] = []
    for title, lines in sections:
        body = [ln for ln in lines[1:] if ln.strip()] if title != "Encabezado" else lines
        if merged and title != "Encabezado" and not body:
            merged[-1][1].extend(lines)
        else:
            merged.append((title, list(lines)))
    return [(title, "\n".join(lines).strip()) for title, lines in merged if "".join(lines).strip()]


def _header(page: int, section: str | None) -> str:
    return f"[Página {page} | Sección: {section}]" if section else f"[Página {page}]"


def chunk_recursive(
    pages: list[tuple[int, str]], chunk_size: int = 400, chunk_overlap: int = 60
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )
    docs: list[Document] = []
    for page_number, text in pages:
        for section, section_text in split_sections(text):
            for piece in splitter.split_text(section_text):
                docs.append(
                    Document(
                        page_content=f"{_header(page_number, section)}\n{piece}",
                        metadata={"page": page_number, "section": section, "chunk": len(docs)},
                    )
                )
    return docs


def chunk_semantic(pages: list[tuple[int, str]], embeddings: Embeddings) -> list[Document]:
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "CHUNKING_STRATEGY=semantic requires 'uv sync --extra semantic'."
        ) from exc

    chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
    docs: list[Document] = []
    for page_number, text in pages:
        if not text.strip():
            continue
        for piece in chunker.split_text(text):
            docs.append(
                Document(
                    page_content=f"{_header(page_number, None)}\n{piece}",
                    metadata={"page": page_number, "section": "", "chunk": len(docs)},
                )
            )
    return docs


def build_chunks(
    pages: list[tuple[int, str]],
    strategy: str,
    embeddings: Embeddings | None = None,
    chunk_size: int = 400,
    chunk_overlap: int = 60,
) -> list[Document]:
    if strategy == "semantic":
        if embeddings is None:
            raise ValueError("SemanticChunker requires embeddings.")
        return chunk_semantic(pages, embeddings)
    return chunk_recursive(pages, chunk_size, chunk_overlap)
