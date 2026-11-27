import base64

import pytest

from app.core.logging import mask_url
from app.services.chunking import build_chunks, is_heading, split_sections
from app.services.document_loader import (
    DocumentLoadError,
    assert_public_host,
    decode_base64,
    download,
    sniff_mime,
)
from app.services.pdf_parser import parse_document
from app.services.postprocess import normalize_date, run_postprocess
from app.services.retrieval import bm25_preprocess

MB = 1024 * 1024


# ---------------------------------------------------------------- loader
def test_decode_base64_pdf(synthetic_pdf):
    b64 = base64.b64encode(synthetic_pdf).decode()
    loaded = decode_base64(b64, 5 * MB)
    assert loaded.mime_type == "application/pdf"
    assert loaded.content == synthetic_pdf


def test_decode_base64_data_uri_and_no_padding(synthetic_pdf):
    b64 = base64.b64encode(synthetic_pdf).decode().rstrip("=")
    loaded = decode_base64(f"data:application/pdf;base64,{b64}", 5 * MB)
    assert loaded.content == synthetic_pdf


def test_decode_base64_invalid():
    with pytest.raises(DocumentLoadError, match="base64"):
        decode_base64("esto no es base64!!", 5 * MB)


def test_decode_base64_unsupported_format():
    with pytest.raises(DocumentLoadError, match="Formato no soportado"):
        decode_base64(base64.b64encode(b"hola mundo").decode(), 5 * MB)


def test_decode_base64_too_big(synthetic_pdf):
    with pytest.raises(DocumentLoadError, match="tamaño máximo"):
        decode_base64(base64.b64encode(synthetic_pdf).decode(), 10)


def test_sniff_mime():
    assert sniff_mime(b"%PDF-1.7") == "application/pdf"
    assert sniff_mime(b"\x89PNG\r\n\x1a\nxxxx") == "image/png"
    assert sniff_mime(b"\xff\xd8\xff\xe0") == "image/jpeg"
    assert sniff_mime(b"GIF89a") is None


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:5080/x.pdf",
        "http://localhost/x.pdf",
        "http://192.168.0.1/x.pdf",
        "http://10.0.0.5/x.pdf",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/x.pdf",
    ],
)
def test_ssrf_guard_rejects_non_public_hosts(url):
    with pytest.raises(DocumentLoadError, match="red privada o local"):
        assert_public_host(url)


def test_ssrf_guard_accepts_public_ip():
    assert_public_host("https://8.8.8.8/doc.pdf")  # IP literal: no DNS lookup needed


async def test_download_blocks_private_url_before_connecting():
    with pytest.raises(DocumentLoadError, match="red privada o local"):
        await download("http://127.0.0.1:5080/health", 5 * MB, timeout=2)


def test_mask_url_hides_sas_token():
    url = "https://acct.blob.core.windows.net/c/CSF.pdf?sv=2023&sig=SECRET"
    masked = mask_url(url)
    assert "SECRET" not in masked and masked.endswith("CSF.pdf?***")


# ---------------------------------------------------------------- parser
def test_parse_synthetic(synthetic_pdf):
    parsed = parse_document(synthetic_pdf, "application/pdf")
    assert parsed.total_pages == 2
    info = parsed.pages[0].info
    assert (info.width, info.height, info.unit, info.orientation) == (8.5, 11.0, "inch", "Portrait")
    assert "PEPJ800101AB1" in parsed.pages[0].text
    assert not any(p.needs_ocr for p in parsed.pages)


def test_parse_sample_csf(sample_csf_path):
    parsed = parse_document(sample_csf_path.read_bytes(), "application/pdf")
    assert parsed.total_pages == 3
    assert all(p.info.width == 8.5 and p.info.height == 11.0 for p in parsed.pages)
    # page 3 only holds a small QR code → must not trigger Vision OCR
    assert not any(p.needs_ocr for p in parsed.pages)


def test_parse_image_needs_ocr():
    import pymupdf

    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 200, 260), False)
    pix.clear_with(255)
    parsed = parse_document(pix.tobytes("png"), "image/png")
    assert parsed.total_pages == 1
    assert parsed.pages[0].needs_ocr


# ---------------------------------------------------------------- chunking
def test_heading_detection():
    assert is_heading("Datos del domicilio registrado")
    assert is_heading("Actividades Económicas:")
    assert not is_heading(" RFC:   ROZM650102D89")
    assert not is_heading("Código Postal:63067")


def test_empty_label_is_not_a_section():
    sections = split_sections("Datos de Identificación:\nRFC: X\nNombre Comercial:\nDatos del domicilio\nCP: 1")
    titles = [t for t, _ in sections]
    assert "Nombre Comercial" not in titles
    assert titles == ["Datos de Identificación", "Datos del domicilio"]


def test_chunks_keep_page_and_header():
    chunks = build_chunks([(1, "Regímenes:\nSueldos"), (2, "Obligaciones:\nISR")], "recursive")
    assert [c.metadata["page"] for c in chunks] == [1, 2]
    assert chunks[0].page_content.startswith("[Página 1 | Sección: Regímenes]")


def test_bm25_preprocess_removes_accents_and_punctuation():
    assert bm25_preprocess("Régimen: RFC, Código") == ["regimen", "rfc", "codigo"]


# ---------------------------------------------------------------- postprocess
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("21 DE ABRIL DE 2026", "21/04/2026"),
        ("1 de octubre de 1991", "01/10/1991"),
        ("2026-04-21", "21/04/2026"),
        ("31/03/2002", "31/03/2002"),
        ("ACTIVO", "ACTIVO"),
    ],
)
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


def test_postprocess_csf_derived_fields():
    from tests.conftest import CSF_RESULT

    out = run_postprocess(dict(CSF_RESULT), "csf")
    assert out["DatosEmision"] == {"Lugar": "GUADALAJARA, JALISCO", "Fecha": "05/03/2026"}
    assert out["DatosIdentificacion"]["FechaInicioOperaciones"] == "01/01/2010"
    assert out["ActividadEconomicaMayorPorcentaje"]["Orden"] == 2
    assert out["PrimerRegimen"]["Regimen"] == "Régimen de Sueldos y Salarios"
