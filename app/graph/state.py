"""Analysis graph state.

The state must be SERIALIZABLE because the checkpointer stores it after every node.
Only dicts/lists/strings travel here. Heavy or non-serializable objects
(PDF bytes, retriever, Chroma collection) live in `TaskResources` (see nodes.py).
"""

from typing import Any, Literal, TypedDict


class AnalyzeState(TypedDict, total=False):
    # --- input ---
    task_id: str
    request: dict[str, Any]  # AnalyzeRequest without the base64 payload
    input_echo: dict[str, Any]  # what is returned in `input`
    threshold: float

    # --- ingestion / parsing ---
    mime_type: str
    source: str
    pages: list[dict[str, Any]]  # [{page_number, text, info, needs_ocr, ocr_applied}]
    ocr_pages: list[int]

    # --- classification ---
    candidates: list[dict[str, Any]]  # semantic shortlist from the catalog
    prediction: dict[str, Any]
    is_valid_category: bool
    is_valid_type: bool

    # --- extraction ---
    extraction_path: Literal["full", "retrieval", "skip"]
    chunks_count: int
    retrieved_chunks: list[dict[str, Any]]
    raw_result: dict[str, Any] | None
    result: dict[str, Any] | None

    # --- control ---
    steps: list[str]
    error: str | None
    error_step: str | None
    response: dict[str, Any]
