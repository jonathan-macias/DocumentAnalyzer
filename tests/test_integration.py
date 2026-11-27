"""REAL integration tests against OpenAI. Run with:  uv run pytest -m integration

Every committed sample in samples/ is analyzed end to end and compared with its golden file
in samples/expected/. Requires OPENAI_API_KEY.
"""

import base64
import os
import re
import unicodedata
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.conftest import SAMPLES_DIR, load_expected

pytestmark = pytest.mark.integration

EXPECTED = load_expected()


def norm(value: Any) -> str:
    """Compare ignoring accents, case, whitespace, commas and '$'."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[\s,$]", "", text).upper()


def get_path(data: Any, path: str) -> Any:
    for part in path.split("."):
        data = data.get(part) if isinstance(data, dict) else None
    return data


@pytest.fixture(scope="module")
def live_client():
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")
    from app.main import app

    with TestClient(app) as client:  # "with" → runs the real lifespan
        client.app.state.settings.extraction_mode = "auto"
        yield client


def analyze(client: TestClient, pdf_name: str, category: str | None, **settings) -> tuple[dict, dict]:
    original = {k: getattr(client.app.state.settings, k) for k in settings}
    for key, value in settings.items():
        setattr(client.app.state.settings, key, value)
    try:
        b64 = base64.b64encode((SAMPLES_DIR / pdf_name).read_bytes()).decode()
        r = client.post("/Document/Analyze", json={"documentBase64": b64, "documentCategory": category})
    finally:
        for key, value in original.items():
            setattr(client.app.state.settings, key, value)
    body = r.json()
    assert r.status_code == 200, body
    trace = client.get(f"/Document/Analyze/{body['documentTaskId']}/trace").json()
    return body, trace


@pytest.mark.parametrize("case", EXPECTED, ids=lambda c: c["file"].removesuffix(".pdf"))
def test_sample_end_to_end(live_client, case):
    is_other = case["documentCategory"] == "OTRO"
    body, trace = analyze(live_client, case["file"], None if is_other else case["documentCategory"])

    assert body["success"], body["errorMessage"]
    assert body["totalPages"] == case["pages"]
    assert body["prediction"]["documentCategory"] == case["documentCategory"]
    assert body["prediction"]["documentType"] == case["documentType"]
    assert trace["extraction_path"] == case["extractionPath"]

    if is_other:
        assert body["result"] is None and not body["isValidCategory"]
        return

    assert body["isValidCategory"] and body["isValidType"]
    result = body["result"]
    errors = []
    for path, expected in case["fields"].items():
        if norm(get_path(result, path)) != norm(expected):
            errors.append(f"{path}: got {get_path(result, path)!r}, expected {expected!r}")
    for path, expected in case["contains"].items():
        if norm(expected) not in norm(get_path(result, path)):
            errors.append(f"{path}: got {get_path(result, path)!r}, expected to contain {expected!r}")
    for path, expected in case["counts"].items():
        if len(get_path(result, path) or []) != expected:
            errors.append(f"{path}: expected {expected} items")
    assert not errors, "\n".join(errors)


def test_csf_forced_retrieval(live_client, sample_csf_path):
    """Short CSF forced through the hybrid retrieval path must extract the same data."""
    body, trace = analyze(live_client, sample_csf_path.name, "DOCUMENTO_FISCAL", extraction_mode="retrieval")
    assert body["success"], body["errorMessage"]
    assert trace["extraction_path"] == "retrieval" and trace["retrieved_chunks"]
    assert body["result"]["DatosIdentificacion"]["RFC"]
    assert len(body["result"]["ActividadesEconomicas"]) >= 1
