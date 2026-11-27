"""FastAPI entry point.

`lifespan` builds every expensive object ONCE at startup:
settings → models → catalog → semantic index → nodes → compiled graph.
"""

import logging
import warnings
from contextlib import asynccontextmanager

import chromadb
from chromadb.config import Settings as ChromaSettings
from fastapi import FastAPI

from app.api.routes.document import router as document_router
from app.catalog.registry import CatalogRegistry
from app.core.config import get_settings
from app.core.logging import setup_logging
from app.graph.analyzer import DocumentAnalyzer
from app.graph.builder import build_graph
from app.graph.nodes import AnalyzeDeps, AnalyzeNodes
from app.llm.factory import get_chat_model, get_embeddings, get_vision_model
from app.services.catalog_index import CatalogIndex

# langchain-community is being sunset; its BM25Retriever is still used here
warnings.filterwarnings("ignore", message=".*langchain-community.*", category=DeprecationWarning)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)

    embeddings = get_embeddings(settings)
    registry = CatalogRegistry.load()
    catalog_index = CatalogIndex(
        registry, embeddings, settings.chroma_dir, settings.catalog_collection
    )
    catalog_index.build()

    deps = AnalyzeDeps(
        settings=settings,
        llm=get_chat_model(settings),
        vision_llm=get_vision_model(settings),
        embeddings=embeddings,
        registry=registry,
        catalog_index=catalog_index,
        chroma_client=chromadb.EphemeralClient(ChromaSettings(anonymized_telemetry=False)),
    )
    nodes = AnalyzeNodes(deps)
    graph = build_graph(nodes)

    app.state.settings = settings
    app.state.registry = registry
    app.state.graph = graph
    app.state.analyzer = DocumentAnalyzer(nodes, graph, settings.default_threshold)
    logger.info(
        "API ready: model=%s, types=%d, extraction=%s, chunking=%s",
        settings.llm_model,
        len(registry.types),
        settings.extraction_mode,
        settings.chunking_strategy,
    )
    yield


app = FastAPI(
    title="Document Analyze API",
    version="0.1.0",
    description="KYC document classification and extraction with LangChain + LangGraph + OpenAI.",
    lifespan=lifespan,
)
app.include_router(document_router)


@app.get("/health", tags=["Health"])
async def health() -> dict:
    return {"status": "ok"}


@app.get("/catalog", tags=["Catalog"], summary="Supported categories and document types")
async def catalog() -> dict:
    registry: CatalogRegistry = app.state.registry
    return {
        cat.category: [t.type for t in cat.types] for cat in registry.categories.values()
    }


@app.get("/graph", tags=["Graph"], summary="Mermaid diagram of the LangGraph graph")
async def graph_diagram() -> dict:
    return {"mermaid": app.state.graph.get_graph().draw_mermaid()}
