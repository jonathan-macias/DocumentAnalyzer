"""Central configuration.

- `BaseSettings` reads environment variables and `.env`.
- `get_settings()` is cached with `@lru_cache` → a single instance per process.
- `openai_api_key` is required: if missing, the app fails at startup (fail fast).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    # --- App ---
    app_name: str = "Document Analyze API"
    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"

    # --- OpenAI ---
    openai_api_key: str = Field(..., description="OpenAI API key (required)")
    llm_model: str = "gpt-6-luna"
    llm_reasoning_effort: str | None = "none"
    vision_model: str = "gpt-6-luna"
    embeddings_model: str = "text-embedding-3-small"
    llm_timeout_seconds: float = 120.0
    max_retries: int = 3

    # --- Ingestion ---
    max_document_mb: float = 20.0
    download_timeout_seconds: float = 30.0
    min_chars_per_page: int = 20  # below this a page is "textless" → candidate for Vision OCR
    vision_dpi: int = 150

    # --- Classification / validation ---
    default_threshold: float = 0.80
    catalog_top_k: int = 4

    # --- Chunking / retrieval ---
    chunking_strategy: Literal["recursive", "semantic"] = "recursive"
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 60
    retrieval_k: int = 4
    bm25_weight: float = 0.4
    vector_weight: float = 0.6

    # --- Extraction ---
    extraction_mode: Literal["auto", "full", "retrieval"] = "auto"
    full_context_max_tokens: int = 12000

    # --- Persistence ---
    chroma_dir: Path = BASE_DIR / "data" / "chroma"
    catalog_collection: str = "document_catalog"

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def max_document_bytes(self) -> int:
        return int(self.max_document_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
