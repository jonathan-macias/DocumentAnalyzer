"""Shared fixtures.

- Unit tests do NOT call OpenAI: they use a fake LLM + deterministic embeddings.
- A SYNTHETIC CSF-like PDF is generated (no real personal data).
- Committed sample documents with fictitious data live in samples/ (see
  scripts/generate_samples.py). SAMPLE_CSF_PATH can point to another CSF (e.g. a real one,
  which must never be committed).
"""

import base64
import json
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chromadb
import pymupdf
import pytest
from chromadb.config import Settings as ChromaSettings
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

from app.catalog.registry import CatalogRegistry
from app.core.config import Settings
from app.graph.analyzer import DocumentAnalyzer
from app.graph.builder import build_graph
from app.graph.nodes import AnalyzeDeps, AnalyzeNodes
from app.services.catalog_index import CatalogIndex

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_DIR = ROOT / "samples"
EXPECTED_DIR = SAMPLES_DIR / "expected"


def load_expected() -> list[dict]:
    """Golden files written by scripts/generate_samples.py."""
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(EXPECTED_DIR.glob("*.json"))]

SYNTHETIC_CSF_PAGES = [
    """CÉDULA DE IDENTIFICACIÓN FISCAL
CONSTANCIA DE SITUACIÓN FISCAL
Lugar y Fecha de Emisión
GUADALAJARA , JALISCO A 05 DE MARZO DE 2026
Datos de Identificación del Contribuyente:
RFC: PEPJ800101AB1
CURP: PEPJ800101HJCRRN09
Nombre (s): JUAN
Primer Apellido: PEREZ
Segundo Apellido: PEREZ
Fecha inicio de operaciones: 01 DE ENERO DE 2010
Estatus en el padrón: ACTIVO
Datos del domicilio registrado
Código Postal: 44100
Nombre de Vialidad: JUAREZ   Número Exterior: 100""",
    """Actividades Económicas:
1   Comercio al por menor   30   01/01/2010
2   Servicios de consultoría   70   01/01/2015
Regímenes:
Régimen de Sueldos y Salarios   01/01/2010""",
]


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page(width=612, height=792)  # US Letter: 8.5 x 11 in
        page.insert_text((50, 72), text, fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def synthetic_pdf() -> bytes:
    return make_pdf(SYNTHETIC_CSF_PAGES)


@pytest.fixture
def synthetic_pdf_b64(synthetic_pdf: bytes) -> str:
    return base64.b64encode(synthetic_pdf).decode()


@pytest.fixture
def sample_csf_path() -> Path:
    path = Path(os.getenv("SAMPLE_CSF_PATH", SAMPLES_DIR / "01_csf_constancia_situacion_fiscal.pdf"))
    if not path.exists():
        pytest.skip("Sample PDF not available (set SAMPLE_CSF_PATH).")
    return path


# --------------------------------------------------------------------------- fake LLM


class FakeStructuredLLM:
    """Mimics `llm.with_structured_output(Schema)` returning canned responses.

    `responses` maps schema name → function(schema) → instance.
    """

    def __init__(self, responses: dict[str, Callable[[type[BaseModel]], BaseModel]]):
        self.responses = responses
        self.calls: list[str] = []

    def with_structured_output(self, schema: type[BaseModel]):
        def _run(_: Any) -> BaseModel:
            self.calls.append(schema.__name__)
            return self.responses[schema.__name__](schema)

        return RunnableLambda(_run)


CSF_RESULT = {
    "DatosEmision": {"Lugar": "GUADALAJARA , JALISCO", "Fecha": "05 DE MARZO DE 2026"},
    "DatosIdentificacion": {
        "RFC": "PEPJ800101AB1",
        "CURP": "PEPJ800101HJCRRN09",
        "Nombre": "JUAN",
        "ApellidoPaterno": "PEREZ",
        "ApellidoMaterno": "PEREZ",
        "FechaInicioOperaciones": "01 DE ENERO DE 2010",
        "EstatusPadron": "ACTIVO",
        "FechaUltimoCambioEstado": "",
        "NombreComercial": "",
    },
    "DatosDomicilio": {
        "Direccion": "",
        "TipoVialidad": "",
        "Calle": "JUAREZ",
        "NumeroExterior": "100",
        "NumeroInterior": "",
        "Cruzamientos": "",
        "EntreCalle": "",
        "YCalle": "",
        "Colonia": "",
        "CodigoPostal": "44100",
        "Localidad": "",
        "Municipio": "",
        "Estado": "",
    },
    "ActividadesEconomicas": [
        {"Orden": 1, "ActividadEconomica": "Comercio al por menor", "Porcentaje": 30,
         "FechaInicio": "01/01/2010", "FechaFin": ""},
        {"Orden": 2, "ActividadEconomica": "Servicios de consultoría", "Porcentaje": 70,
         "FechaInicio": "01/01/2015", "FechaFin": ""},
    ],
    "Regimenes": [
        {"Regimen": "Régimen de Sueldos y Salarios", "FechaInicio": "01/01/2010", "FechaFin": ""}
    ],
}  # fmt: skip


def fake_llm(category="DOCUMENTO_FISCAL", cat_score=0.97, doc_type="CONSTANCIA_SITUACION_FISCAL",
             type_score=0.95) -> FakeStructuredLLM:  # fmt: skip
    return FakeStructuredLLM(
        {
            "CategoryPrediction": lambda s: s(
                document_category=category, score=cat_score, reasoning="Documento del SAT."
            ),
            "TypePrediction": lambda s: s(
                document_type=doc_type, score=type_score, reasoning="Es una CSF."
            ),
            "ConstanciaSituacionFiscal": lambda s: s.model_validate(CSF_RESULT),
        }
    )


@pytest.fixture
def make_analyzer(tmp_path):
    """Factory: builds the full graph with fake dependencies."""

    def _make(llm: FakeStructuredLLM | None = None, **settings_overrides) -> DocumentAnalyzer:
        settings = Settings(
            openai_api_key="test-key",
            app_env="test",
            chroma_dir=tmp_path / "chroma",
            **settings_overrides,
        )
        embeddings = DeterministicFakeEmbedding(size=64)
        registry = CatalogRegistry.load()
        catalog_index = CatalogIndex(registry, embeddings, settings.chroma_dir, "catalog_test")
        catalog_index.build()
        llm = llm or fake_llm()
        deps = AnalyzeDeps(
            settings=settings,
            llm=llm,
            vision_llm=llm,
            embeddings=embeddings,
            registry=registry,
            catalog_index=catalog_index,
            chroma_client=chromadb.EphemeralClient(ChromaSettings(anonymized_telemetry=False)),
        )
        nodes = AnalyzeNodes(deps)
        return DocumentAnalyzer(nodes, build_graph(nodes), settings.default_threshold)

    return _make
