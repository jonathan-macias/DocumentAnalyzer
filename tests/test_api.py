"""HTTP tests. A fake analyzer is injected into app.state (the lifespan is not run)."""

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import app


@pytest.fixture
def client(make_analyzer):
    app.state.analyzer = make_analyzer()
    app.state.settings = None  # no API_KEY → authentication disabled
    return TestClient(app)  # no "with" → lifespan does not run (OpenAI is never touched)


@pytest.fixture
def secured_client(client):
    app.state.settings = Settings(openai_api_key="test-key", api_key="s3cret")
    yield client
    app.state.settings = None


def test_api_key_required_when_configured(secured_client, synthetic_pdf_b64):
    body = {"documentBase64": synthetic_pdf_b64}
    assert secured_client.post("/Document/Analyze", json=body).status_code == 401
    wrong = secured_client.post("/Document/Analyze", json=body, headers={"X-API-Key": "nope"})
    assert wrong.status_code == 401
    ok = secured_client.post("/Document/Analyze", json=body, headers={"X-API-Key": "s3cret"})
    assert ok.status_code == 200 and ok.json()["success"]
    # results and traces are protected too (they contain extracted personal data)
    task_id = ok.json()["documentTaskId"]
    assert secured_client.get(f"/Document/Analyze/{task_id}").status_code == 401
    assert secured_client.get(f"/Document/Analyze/{task_id}/trace").status_code == 401


def test_public_endpoints_do_not_need_api_key(secured_client):
    assert secured_client.get("/health").status_code == 200


def test_requires_exactly_one_source(client):
    r = client.post("/Document/Analyze", json={"documentCategory": "DOCUMENTO_FISCAL"})
    assert r.status_code == 422
    r = client.post("/Document/Analyze", json={"documentUrl": "https://x/y.pdf", "documentBase64": "AA=="})
    assert r.status_code == 422


def test_rejects_non_http_url(client):
    r = client.post("/Document/Analyze", json={"documentUrl": "file:///etc/passwd"})
    assert r.status_code == 422


def test_threshold_range(client, synthetic_pdf_b64):
    r = client.post("/Document/Analyze", json={"documentBase64": synthetic_pdf_b64, "threshold": 1.5})
    assert r.status_code == 422


def test_analyze_response_shape(client, synthetic_pdf_b64):
    r = client.post(
        "/Document/Analyze",
        json={"documentBase64": synthetic_pdf_b64, "documentCategory": "DOCUMENTO_FISCAL", "threshold": None},
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "success", "documentTaskId", "input", "prediction", "isValidCategory", "isValidType",
        "result", "pages", "totalPages", "errorMessage",
    }  # fmt: skip
    assert set(body["prediction"]) == {
        "documentCategory", "documentType", "categoryPredictionScore", "typePredictionScore",
        "categoryReasoning", "typeReasoning",
    }  # fmt: skip
    assert body["pages"][0] == {
        "pageNumber": 1, "angle": 0.0, "width": 8.5, "height": 11.0,
        "unit": "inch", "orientation": "Portrait",
    }  # fmt: skip

    task_id = body["documentTaskId"]
    assert client.get(f"/Document/Analyze/{task_id}").json()["documentTaskId"] == task_id
    assert client.get("/Document/Analyze/no-existe").status_code == 404
    assert "steps" in client.get(f"/Document/Analyze/{task_id}/trace").json()
