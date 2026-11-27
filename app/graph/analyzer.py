"""Facade between the API and the graph: creates the task id, prepares the state and runs it."""

import logging
import uuid
from typing import Any

from app.graph.nodes import AnalyzeNodes
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse

logger = logging.getLogger(__name__)

# State keys NOT exposed by /trace (full document text and raw payloads)
_TRACE_EXCLUDE = {"pages", "response", "raw_result", "request", "input_echo"}


class DocumentAnalyzer:
    def __init__(self, nodes: AnalyzeNodes, graph, default_threshold: float) -> None:
        self.nodes = nodes
        self.graph = graph
        self.default_threshold = default_threshold

    @staticmethod
    def _config(task_id: str) -> dict[str, Any]:
        # thread_id = documentTaskId → each analysis is its own checkpointer "thread"
        return {"configurable": {"thread_id": task_id}}

    async def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        task_id = str(uuid.uuid4())
        threshold = request.threshold if request.threshold is not None else self.default_threshold

        b64 = request.document_base64
        input_echo = {
            "document_url": request.document_url,
            "document_base64": f"<base64: {len(b64)} caracteres>" if b64 else None,
            "file_name": request.file_name,
            "document_category": request.document_category,
            "threshold": threshold,
        }
        state = {
            "task_id": task_id,
            "request": request.model_dump(exclude={"document_base64"}),
            "input_echo": input_echo,
            "threshold": threshold,
            "steps": [],
            "error": None,
        }

        # The base64 payload goes to per-task resources, never into the checkpointed state
        self.nodes.open_task(task_id, b64)
        try:
            final = await self.graph.ainvoke(state, config=self._config(task_id))
        finally:
            self.nodes.close_task(task_id)  # idempotent: finalize already closes it
        logger.info(
            "[%s] steps=%s ok=%s", task_id, final.get("steps"), final.get("error") is None
        )
        return AnalyzeResponse.model_validate(final["response"])

    async def get_state(self, task_id: str) -> dict[str, Any] | None:
        snapshot = await self.graph.aget_state(self._config(task_id))
        return snapshot.values or None

    async def get_response(self, task_id: str) -> dict[str, Any] | None:
        values = await self.get_state(task_id)
        return values.get("response") if values else None

    async def get_trace(self, task_id: str) -> dict[str, Any] | None:
        values = await self.get_state(task_id)
        if not values:
            return None
        trace = {k: v for k, v in values.items() if k not in _TRACE_EXCLUDE}
        trace["pages"] = [
            {k: p[k] for k in ("page_number", "needs_ocr", "ocr_applied")}
            | {"chars": len(p["text"])}
            for p in values.get("pages", [])
        ]
        return trace
