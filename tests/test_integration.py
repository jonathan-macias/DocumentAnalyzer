"""REAL integration tests against OpenAI. Run with:  uv run pytest -m integration

Requires OPENAI_API_KEY and the sample PDF (SAMPLE_CSF_PATH or in the project root).
"""

import base64
import os

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def live_client():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")
    from app.main import app

    with TestClient(app) as client:  # "with" → runs the real lifespan
        yield client


@pytest.mark.parametrize("extraction_mode", ["full", "retrieval"])
def test_csf_end_to_end(live_client, sample_csf_path, extraction_mode):
    live_client.app.state.settings.extraction_mode = extraction_mode
    b64 = base64.b64encode(sample_csf_path.read_bytes()).decode()
    r = live_client.post(
        "/Document/Analyze",
        json={"documentBase64": b64, "documentCategory": "DOCUMENTO_FISCAL", "threshold": None},
    )
    body = r.json()
    assert r.status_code == 200, body
    assert body["success"], body["errorMessage"]
    assert body["prediction"]["documentCategory"] == "DOCUMENTO_FISCAL"
    assert body["prediction"]["documentType"] == "CONSTANCIA_SITUACION_FISCAL"
    assert body["isValidCategory"] and body["isValidType"]
    assert body["totalPages"] == 3

    result = body["result"]
    assert result["DatosIdentificacion"]["RFC"] == "ROZM650102D89"
    assert result["DatosIdentificacion"]["CURP"] == "ROZM650102MNTSXR05"
    assert result["DatosEmision"]["Fecha"] == "21/04/2026"
    assert result["DatosDomicilio"]["CodigoPostal"] == "63067"
    assert len(result["ActividadesEconomicas"]) == 2
    assert result["ActividadEconomicaMayorPorcentaje"]["Porcentaje"] == 98
