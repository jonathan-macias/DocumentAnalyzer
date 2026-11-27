from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.graph.analyzer import DocumentAnalyzer
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse

router = APIRouter(prefix="/Document", tags=["Document"])


def _analyzer(request: Request) -> DocumentAnalyzer:
    return request.app.state.analyzer


@router.post(
    "/Analyze",
    response_model=AnalyzeResponse,
    response_model_by_alias=True,
    summary="Classify a document and extract its data",
)
async def analyze_document(body: AnalyzeRequest, request: Request) -> AnalyzeResponse:
    """Accepts `documentUrl` **or** `documentBase64`.

    - Body validation errors → 422.
    - Business errors (download, corrupted PDF, LLM) → 200 with `success=false` and `errorMessage`.
    """
    return await _analyzer(request).analyze(body)


@router.get(
    "/Analyze/{document_task_id}",
    summary="Get the result of a previous analysis (LangGraph checkpointer)",
)
async def get_analysis(document_task_id: str, request: Request) -> dict[str, Any]:
    response = await _analyzer(request).get_response(document_task_id)
    if response is None:
        raise HTTPException(status_code=404, detail="documentTaskId no encontrado.")
    return response


@router.get(
    "/Analyze/{document_task_id}/trace",
    summary="Graph trace: steps, semantic candidates, extraction path, retrieved chunks",
)
async def get_trace(document_task_id: str, request: Request) -> dict[str, Any]:
    trace = await _analyzer(request).get_trace(document_task_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="documentTaskId no encontrado.")
    return trace
