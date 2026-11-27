"""Graph nodes. Each node receives the state and returns ONLY the keys it updates.

Error handling:
- `@guard("name")` catches any exception raised by the node and stores it in state["error"].
- The routers in builder.py send the flow to `finalize` as soon as there is an error
  → success=false.
- LLM calls use `.with_retry()` for transient failures.
"""

import asyncio
import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import chromadb
import tiktoken
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, create_model

from app.catalog.registry import OTHER_CATEGORY, UNKNOWN_TYPE, CatalogRegistry
from app.core.config import Settings
from app.graph.state import AnalyzeState
from app.llm.prompts import CATEGORY_PROMPT, EXTRACTION_PROMPT, TYPE_PROMPT
from app.schemas.analyze import AnalyzeResponse
from app.services.catalog_index import CatalogIndex
from app.services.chunking import build_chunks
from app.services.document_loader import DocumentLoadError, decode_base64, download
from app.services.pdf_parser import ParsedDocument, PdfParseError, parse_document
from app.services.postprocess import run_postprocess
from app.services.retrieval import HybridIndex
from app.services.vision_ocr import ocr_pages_without_text

logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")
CLASSIFY_MAX_TOKENS = 6000
CATALOG_QUERY_CHARS = 2000

BUSINESS_ERRORS = (DocumentLoadError, PdfParseError)


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def truncate_tokens(text: str, max_tokens: int) -> str:
    tokens = _ENCODING.encode(text)
    return text if len(tokens) <= max_tokens else _ENCODING.decode(tokens[:max_tokens])


# --------------------------------------------------------------------------- dependencies


@dataclass
class AnalyzeDeps:
    settings: Settings
    llm: BaseChatModel
    vision_llm: BaseChatModel
    embeddings: Embeddings
    registry: CatalogRegistry
    catalog_index: CatalogIndex
    chroma_client: chromadb.ClientAPI  # ephemeral client for per-task indexes


@dataclass
class TaskResources:
    """Per-task objects that must NOT go into the state (non-serializable or heavy)."""

    base64_payload: str | None = None
    content: bytes = b""
    parsed: ParsedDocument | None = None
    index: HybridIndex | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def close(self) -> None:
        if self.index:
            self.index.close()
        self.base64_payload = None
        self.content = b""
        self.parsed = None
        self.index = None


# --------------------------------------------------------------------------- dynamic models


def make_prediction_model(name: str, codes: list[str], label_field: str) -> type[BaseModel]:
    """Structured output with a DYNAMIC Literal built from the catalog.

    The LLM can only answer with a valid catalog code → no free-text parsing needed.
    """
    literal = Literal[tuple(codes)]  # type: ignore[valid-type]
    return create_model(
        name,
        **{
            label_field: (literal, Field(description="Código elegido del catálogo.")),
            "score": (float, Field(description="Confianza entre 0 y 1.")),
            "reasoning": (str, Field(description="Justificación breve en español.")),
        },
    )


def _clamp(value: float) -> float:
    return round(min(max(float(value), 0.0), 1.0), 4)


def guard(step: str):
    """Decorator: catches node exceptions and turns them into state['error']."""

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(self: "AnalyzeNodes", state: AnalyzeState) -> dict[str, Any]:
            steps = [*state.get("steps", []), step]
            try:
                update = await fn(self, state) or {}
            except BUSINESS_ERRORS as exc:
                # Expected business errors keep their user-facing message
                logger.warning("[%s] %s: %s", state.get("task_id"), step, exc)
                return {"error": str(exc), "error_step": step, "steps": steps}
            except Exception as exc:  # noqa: BLE001
                # Unexpected errors: log the stack trace, return a generic message
                logger.exception("[%s] Unexpected error in %s", state.get("task_id"), step)
                return {
                    "error": f"Error interno en el paso '{step}': {type(exc).__name__}",
                    "error_step": step,
                    "steps": steps,
                }
            return {**update, "steps": steps}

        return wrapper

    return decorator


# --------------------------------------------------------------------------- nodes


class AnalyzeNodes:
    def __init__(self, deps: AnalyzeDeps) -> None:
        self.deps = deps
        self.s = deps.settings
        self.resources: dict[str, TaskResources] = {}

        # Prediction models are built once: one for categories, one per category for types
        self._category_model = make_prediction_model(
            "CategoryPrediction", deps.registry.category_codes(), "document_category"
        )
        self._type_models = {
            cat: make_prediction_model("TypePrediction", deps.registry.type_codes(cat), "document_type")
            for cat in deps.registry.categories
        }

    # ---- per-task resources ----
    def open_task(self, task_id: str, base64_payload: str | None) -> None:
        self.resources[task_id] = TaskResources(base64_payload=base64_payload)

    def close_task(self, task_id: str) -> None:
        res = self.resources.pop(task_id, None)
        if res:
            res.close()

    def _res(self, state: AnalyzeState) -> TaskResources:
        return self.resources[state["task_id"]]

    def _structured(self, schema: type[BaseModel]):
        return self.deps.llm.with_structured_output(schema).with_retry(
            stop_after_attempt=self.s.max_retries
        )

    @staticmethod
    def _document_text(state: AnalyzeState) -> str:
        return "\n\n".join(
            f"--- Página {p['page_number']} ---\n{p['text']}" for p in state.get("pages", [])
        )

    # 1) ingest -------------------------------------------------------------
    @guard("ingest")
    async def ingest(self, state: AnalyzeState) -> dict[str, Any]:
        res = self._res(state)
        url = state["request"].get("document_url")
        if res.base64_payload:
            loaded = decode_base64(res.base64_payload, self.s.max_document_bytes)
            res.base64_payload = None  # free memory
        else:
            loaded = await download(
                url,
                self.s.max_document_bytes,
                self.s.download_timeout_seconds,
                allow_private=self.s.allow_private_urls,
            )
        res.content = loaded.content
        return {"mime_type": loaded.mime_type, "source": loaded.source}

    # 2) parse --------------------------------------------------------------
    @guard("parse")
    async def parse(self, state: AnalyzeState) -> dict[str, Any]:
        res = self._res(state)
        # PyMuPDF is synchronous → run it in a worker thread
        parsed = await asyncio.to_thread(
            parse_document, res.content, state["mime_type"], self.s.min_chars_per_page
        )
        res.parsed = parsed
        return {"pages": _pages_to_state(parsed)}

    # 3) vision_ocr (only when some page is a scan) -------------------------
    @guard("vision_ocr")
    async def vision_ocr(self, state: AnalyzeState) -> dict[str, Any]:
        parsed = self._res(state).parsed
        done = await ocr_pages_without_text(parsed, self.deps.vision_llm, self.s.vision_dpi)
        return {"pages": _pages_to_state(parsed), "ocr_pages": done}

    # 4) classify_category: retrieve (semantic) → decide (LLM) --------------
    @guard("classify_category")
    async def classify_category(self, state: AnalyzeState) -> dict[str, Any]:
        text = self._document_text(state)
        if not text.strip():
            raise PdfParseError("No se pudo obtener texto del documento.")

        candidates = await self.deps.catalog_index.asearch(
            text[:CATALOG_QUERY_CHARS], k=self.s.catalog_top_k
        )
        candidates_txt = "\n".join(
            f"- {c.document_type} (categoría {c.category}) similitud={c.similarity}"
            for c in candidates
        )
        chain = CATEGORY_PROMPT | self._structured(self._category_model)
        out = await chain.ainvoke(
            {
                "categories": self.deps.registry.describe_categories(),
                "candidates": candidates_txt or "(sin candidatos)",
                "document": truncate_tokens(text, CLASSIFY_MAX_TOKENS),
            }
        )
        return {
            "candidates": [c.__dict__ for c in candidates],
            "prediction": {
                "document_category": out.document_category,
                "category_prediction_score": _clamp(out.score),
                "category_reasoning": out.reasoning,
            },
        }

    # 5) classify_type -------------------------------------------------------
    @guard("classify_type")
    async def classify_type(self, state: AnalyzeState) -> dict[str, Any]:
        prediction = dict(state["prediction"])
        category = prediction["document_category"]
        if category == OTHER_CATEGORY:
            # No catalog category → unknown type, no LLM call needed
            prediction.update(
                document_type=UNKNOWN_TYPE,
                type_prediction_score=prediction["category_prediction_score"],
                type_reasoning="El documento no pertenece a ninguna categoría del catálogo.",
            )
            return {"prediction": prediction}

        chain = TYPE_PROMPT | self._structured(self._type_models[category])
        out = await chain.ainvoke(
            {
                "category": category,
                "types": self.deps.registry.describe_types(category),
                "document": truncate_tokens(self._document_text(state), CLASSIFY_MAX_TOKENS),
            }
        )
        prediction.update(
            document_type=out.document_type,
            type_prediction_score=_clamp(out.score),
            type_reasoning=out.reasoning,
        )
        return {"prediction": prediction}

    # 6) validate: threshold + expected category + choose extraction path ----
    @guard("validate")
    async def validate(self, state: AnalyzeState) -> dict[str, Any]:
        th = state["threshold"]
        p = state["prediction"]
        requested = (state["request"].get("document_category") or "").strip().upper() or None

        cat, cat_score = p["document_category"], p["category_prediction_score"]
        doc_type, type_score = p["document_type"], p["type_prediction_score"]

        is_valid_category = (
            cat != OTHER_CATEGORY and cat_score >= th and (requested is None or requested == cat)
        )
        is_valid_type = is_valid_category and doc_type != UNKNOWN_TYPE and type_score >= th

        # Extraction depends on prediction confidence, not on the expected category
        type_def = self.deps.registry.types.get(doc_type)
        can_extract = (
            type_def is not None
            and type_def.model is not None
            and cat_score >= th
            and type_score >= th
        )
        if not can_extract:
            path = "skip"
        elif self.s.extraction_mode in ("full", "retrieval"):
            path = self.s.extraction_mode
        else:  # auto: full context when the document fits, retrieval otherwise
            tokens = count_tokens(self._document_text(state))
            path = "full" if tokens <= self.s.full_context_max_tokens else "retrieval"

        return {
            "is_valid_category": is_valid_category,
            "is_valid_type": is_valid_type,
            "extraction_path": path,
        }

    # 7) index: chunking + hybrid index ("retrieval" path only) --------------
    @guard("index")
    async def index(self, state: AnalyzeState) -> dict[str, Any]:
        res = self._res(state)
        pages = [(p["page_number"], p["text"]) for p in state["pages"]]
        chunks = await asyncio.to_thread(
            build_chunks,
            pages,
            self.s.chunking_strategy,
            self.deps.embeddings,
            self.s.chunk_size_tokens,
            self.s.chunk_overlap_tokens,
        )
        if not chunks:
            raise PdfParseError("El documento no tiene texto indexable.")
        index = HybridIndex(
            client=self.deps.chroma_client,
            collection_name=f"task-{state['task_id']}",
            documents=chunks,
            embeddings=self.deps.embeddings,
            k=self.s.retrieval_k,
            bm25_weight=self.s.bm25_weight,
            vector_weight=self.s.vector_weight,
        )
        await index.aindex()
        res.index = index
        return {"chunks_count": len(chunks)}

    # 8) extract --------------------------------------------------------------
    @guard("extract")
    async def extract(self, state: AnalyzeState) -> dict[str, Any]:
        type_def = self.deps.registry.types[state["prediction"]["document_type"]]
        retrieved_meta: list[dict[str, Any]] = []

        if state["extraction_path"] == "retrieval":
            # One hybrid query per schema section, run concurrently;
            # results are de-duplicated by chunk id and put back in document order.
            index = self._res(state).index
            queries = type_def.retrieval_queries or [type_def.description]
            results = await asyncio.gather(*(index.aretrieve(q) for q in queries))
            unique = {d.metadata["chunk"]: d for docs in results for d in docs}
            ordered = [unique[k] for k in sorted(unique)]
            context = "\n\n".join(d.page_content for d in ordered)
            retrieved_meta = [
                {"chunk": d.metadata["chunk"], "page": d.metadata["page"], "section": d.metadata.get("section", "")}
                for d in ordered
            ]
            context_mode = f"fragmentos recuperados por búsqueda híbrida ({len(ordered)})"
        else:
            context = self._document_text(state)
            context_mode = "documento completo"

        chain = EXTRACTION_PROMPT | self._structured(type_def.model)
        out = await chain.ainvoke(
            {
                "document_type": type_def.type,
                "type_description": type_def.description.strip(),
                "extra_instructions": type_def.extra_instructions,
                "context_mode": context_mode,
                "context": context,
            }
        )
        return {"raw_result": out.model_dump(), "retrieved_chunks": retrieved_meta}

    # 9) postprocess ------------------------------------------------------------
    @guard("postprocess")
    async def postprocess(self, state: AnalyzeState) -> dict[str, Any]:
        type_def = self.deps.registry.types[state["prediction"]["document_type"]]
        return {"result": run_postprocess(state["raw_result"], type_def.postprocessor)}

    # 10) finalize: build the response and release resources --------------------
    async def finalize(self, state: AnalyzeState) -> dict[str, Any]:
        error = state.get("error")
        pages = [p["info"] for p in state.get("pages", [])]
        prediction = state.get("prediction")
        response = AnalyzeResponse(
            success=error is None,
            document_task_id=state["task_id"],
            input=state["input_echo"],
            prediction=prediction,
            is_valid_category=bool(state.get("is_valid_category")),
            is_valid_type=bool(state.get("is_valid_type")),
            result=state.get("result"),
            pages=pages,
            total_pages=len(pages),
            error_message=error,
        )
        self.close_task(state["task_id"])
        return {
            "response": response.model_dump(by_alias=True, mode="json"),
            "steps": [*state.get("steps", []), "finalize"],
        }


def _pages_to_state(parsed: ParsedDocument) -> list[dict[str, Any]]:
    return [
        {
            "page_number": p.page_number,
            "text": p.text,
            "info": p.info.model_dump(),
            "needs_ocr": p.needs_ocr,
            "ocr_applied": p.ocr_applied,
        }
        for p in parsed.pages
    ]
