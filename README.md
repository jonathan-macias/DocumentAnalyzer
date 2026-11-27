# Document Analyze API

A REST API that classifies Mexican KYC documents and extracts their data. Built with **FastAPI**, **LangChain**, **LangGraph**, **OpenAI** and **Chroma**.

```
POST /Document/Analyze   { "documentUrl" | "documentBase64", "documentCategory", "threshold" }
```

## How it works
The analysis runs as a LangGraph `StateGraph`:

1. **Ingest:** downloads the URL or decodes the base64 payload. The format is detected from magic bytes (PDF, PNG, JPEG, TIFF, WEBP).
2. **Parse:** PyMuPDF extracts per-page text and page metadata (size in inches, rotation, orientation).
3. **Vision OCR (conditional):** pages without a text layer are rendered and transcribed by an OpenAI vision model.
4. **Classify:** first the category, then the type.
   - A semantic search over the document catalog in Chroma produces a shortlist of candidates.
   - The LLM decides through structured output with a dynamic `Literal` of catalog codes.
5. **Validate:** applies the `threshold` and compares against the expected `documentCategory`. It also chooses the extraction path.
6. **Extract:** structured output with the Pydantic model of the predicted type.
   - Short documents are sent in full.
   - Long documents go through hybrid retrieval: BM25 + embeddings fused with RRF (`EnsembleRetriever`).
7. **Post-process:** deterministic date and whitespace normalization, plus derived fields.

## Supported documents
| Category | Types |
|---|---|
| `DOCUMENTO_FISCAL` | `CONSTANCIA_SITUACION_FISCAL` |
| `IDENTIFICACION` | `INE`, `PASAPORTE` |
| `COMPROBANTE_DOMICILIO` | `RECIBO_CFE`, `RECIBO_AGUA`, `RECIBO_TELEFONO` |
| `DOCUMENTO_BANCARIO` | `ESTADO_CUENTA` |

To add a type:
1. Create its Pydantic model in `app/catalog/models/` and register it in `EXTRACTION_MODELS`.
2. Add a YAML entry in `app/catalog/definitions/`.
3. Restart the API. The catalog index is re-embedded automatically when the YAML files change.

## Quick start
```powershell
uv sync
copy .env.example .env   # set OPENAI_API_KEY
uv run uvicorn app.main:app --reload
```
Swagger UI: http://127.0.0.1:8000/docs

## Endpoints
| Method | Path | Description |
|---|---|---|
| POST | `/Document/Analyze` | Classify and extract a document |
| GET | `/Document/Analyze/{documentTaskId}` | Previous result (from the LangGraph checkpointer) |
| GET | `/Document/Analyze/{documentTaskId}/trace` | Steps, semantic candidates, extraction path, retrieved chunks |
| GET | `/catalog` | Supported categories and types |
| GET | `/graph` | Mermaid diagram of the graph |
| GET | `/health` | Liveness probe |

## HTTP status codes
- **Invalid body → 422.** This covers a missing source, both sources at once, a non-http URL, or a threshold out of range.
- **Business errors → 200** with `success=false` and `errorMessage`. This covers a failed download, a corrupted PDF, an unsupported format, or an LLM error.

## Configuration
All settings live in `app/core/config.py` and can be overridden in `.env` (see `.env.example`). The most relevant ones:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` / `VISION_MODEL` | `gpt-6-luna` | Chat and vision models |
| `EMBEDDINGS_MODEL` | `text-embedding-3-small` | Embeddings for catalog and hybrid index |
| `DEFAULT_THRESHOLD` | `0.80` | Used when `threshold` is null |
| `EXTRACTION_MODE` | `auto` | `auto`, `full` or `retrieval` |
| `CHUNKING_STRATEGY` | `recursive` | `recursive` or `semantic` (requires `uv sync --extra semantic`) |
| `BM25_WEIGHT` / `VECTOR_WEIGHT` | `0.4` / `0.6` | Hybrid retrieval weights |

## Sample documents
[`samples/`](samples/README.md) contains one PDF per catalog type, plus three edge cases:
- a scanned INE, which goes through Vision OCR;
- a 19-page bank statement, which goes through hybrid retrieval;
- an out-of-catalog letter, which is classified as `OTRO`.

All data is **fictitious** and there are no logos. Each PDF has a golden file in `samples/expected/`. Regenerate them with:
```powershell
uv run python scripts/generate_samples.py
```

## Tests
```powershell
uv run pytest                 # unit tests + sample checks, no network
uv run pytest -m integration  # real OpenAI calls over every sample vs. its golden file
uv run ruff check .
```

> ⚠️ Real documents contain personal data. `*.pdf` is git-ignored (except `samples/*.pdf`) and real documents must never be committed.
