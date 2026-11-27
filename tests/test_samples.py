"""Offline checks of the committed sample documents (samples/) and their golden files."""

import pytest

from app.catalog.registry import UNKNOWN_TYPE, CatalogRegistry
from app.core.config import Settings
from app.graph.nodes import count_tokens
from app.services.pdf_parser import parse_document
from tests.conftest import SAMPLES_DIR, load_expected

EXPECTED = load_expected()
FULL_CONTEXT_MAX_TOKENS = Settings.model_fields["full_context_max_tokens"].default


def _id(case: dict) -> str:
    return case["file"].removesuffix(".pdf")


def test_there_is_a_sample_for_every_catalog_type():
    covered = {case["documentType"] for case in EXPECTED}
    assert set(CatalogRegistry.load().types) <= covered
    assert UNKNOWN_TYPE in covered  # negative case


@pytest.mark.parametrize("case", EXPECTED, ids=_id)
def test_sample_parses_as_expected(case):
    pdf = SAMPLES_DIR / case["file"]
    parsed = parse_document(pdf.read_bytes(), "application/pdf")

    assert parsed.total_pages == case["pages"]
    is_scan = case["file"].startswith("03_ine_escaneada")
    assert any(p.needs_ocr for p in parsed.pages) is is_scan

    if not is_scan:
        tokens = count_tokens(parsed.full_text)
        if case["extractionPath"] == "retrieval":
            assert tokens > FULL_CONTEXT_MAX_TOKENS  # "auto" mode must pick hybrid retrieval
        else:
            assert tokens <= FULL_CONTEXT_MAX_TOKENS


@pytest.mark.parametrize("case", [c for c in EXPECTED if c["documentType"] != UNKNOWN_TYPE], ids=_id)
def test_golden_file_paths_exist_in_extraction_model(case):
    """Every expected field path must exist in the Pydantic model of the document type."""
    model = CatalogRegistry.load().types[case["documentType"]].model
    derived = {"ActividadEconomicaMayorPorcentaje", "PrimerRegimen"}  # computed in postprocess
    for path in [*case["fields"], *case["contains"], *case["counts"]]:
        current = model
        for part in path.split("."):
            if part in derived:
                break
            assert part in current.model_fields, f"{path} not in {model.__name__}"
            annotation = current.model_fields[part].annotation
            current = annotation if isinstance(annotation, type) else current
