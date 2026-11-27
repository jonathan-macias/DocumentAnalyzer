"""Full graph tests with a fake LLM (no network)."""

import base64

from app.schemas.analyze import AnalyzeRequest
from tests.conftest import fake_llm


def _req(b64: str, **kw) -> AnalyzeRequest:
    return AnalyzeRequest(documentBase64=b64, **kw)


async def test_happy_path_full_context(make_analyzer, synthetic_pdf_b64):
    analyzer = make_analyzer()
    resp = await analyzer.analyze(_req(synthetic_pdf_b64, documentCategory="DOCUMENTO_FISCAL"))

    assert resp.success and resp.error_message is None
    assert resp.prediction.document_type == "CONSTANCIA_SITUACION_FISCAL"
    assert resp.is_valid_category and resp.is_valid_type
    assert resp.input.threshold == 0.80
    assert resp.input.document_base64.startswith("<base64:")
    assert resp.total_pages == 2 and resp.pages[0].width == 8.5
    assert resp.result["DatosIdentificacion"]["RFC"] == "PEPJ800101AB1"
    assert resp.result["ActividadEconomicaMayorPorcentaje"]["Porcentaje"] == 70

    trace = await analyzer.get_trace(resp.document_task_id)
    assert trace["extraction_path"] == "full"
    assert "index" not in trace["steps"]  # short path: nothing gets indexed
    assert trace["candidates"], "catalog semantic search must return candidates"


async def test_forced_retrieval_path_uses_hybrid_index(make_analyzer, synthetic_pdf_b64):
    analyzer = make_analyzer(extraction_mode="retrieval")
    resp = await analyzer.analyze(_req(synthetic_pdf_b64))

    assert resp.success
    trace = await analyzer.get_trace(resp.document_task_id)
    assert trace["extraction_path"] == "retrieval"
    assert "index" in trace["steps"]
    assert trace["chunks_count"] > 0 and trace["retrieved_chunks"]
    # per-task resources (ephemeral Chroma collection) are released
    assert resp.document_task_id not in analyzer.nodes.resources


async def test_category_mismatch_is_flagged(make_analyzer, synthetic_pdf_b64):
    resp = await make_analyzer().analyze(_req(synthetic_pdf_b64, documentCategory="IDENTIFICACION"))
    assert resp.success
    assert resp.is_valid_category is False and resp.is_valid_type is False
    assert resp.result is not None  # extracted per the prediction, but flagged as invalid


async def test_low_score_skips_extraction(make_analyzer, synthetic_pdf_b64):
    llm = fake_llm(cat_score=0.5, type_score=0.5)
    resp = await make_analyzer(llm).analyze(_req(synthetic_pdf_b64, threshold=0.9))
    assert resp.success
    assert resp.is_valid_category is False
    assert resp.result is None
    assert "ConstanciaSituacionFiscal" not in llm.calls


async def test_other_category_short_circuits_type(make_analyzer, synthetic_pdf_b64):
    llm = fake_llm(category="OTRO", cat_score=0.9)
    resp = await make_analyzer(llm).analyze(_req(synthetic_pdf_b64))
    assert resp.prediction.document_type == "DESCONOCIDO"
    assert "TypePrediction" not in llm.calls
    assert resp.result is None


async def test_invalid_document_returns_business_error(make_analyzer):
    b64 = base64.b64encode(b"no soy un pdf").decode()
    resp = await make_analyzer().analyze(_req(b64))
    assert resp.success is False
    assert "Formato no soportado" in resp.error_message
    assert resp.prediction is None and resp.total_pages == 0


async def test_unexpected_llm_error_is_reported(make_analyzer, synthetic_pdf_b64):
    llm = fake_llm()

    def boom(_schema):
        raise RuntimeError("timeout simulado")

    llm.responses["CategoryPrediction"] = boom
    resp = await make_analyzer(llm, max_retries=1).analyze(_req(synthetic_pdf_b64))
    assert resp.success is False
    assert "classify_category" in resp.error_message


async def test_get_response_from_checkpointer(make_analyzer, synthetic_pdf_b64):
    analyzer = make_analyzer()
    resp = await analyzer.analyze(_req(synthetic_pdf_b64))
    stored = await analyzer.get_response(resp.document_task_id)
    assert stored["documentTaskId"] == resp.document_task_id
    assert await analyzer.get_response("no-existe") is None
