# syntax=docker/dockerfile:1
#
# Document Analyze API image (built for linux/arm64 in CI, works on amd64 too).
#
#   docker build -t document-analyzer:local .
#   docker run --rm -p 8000:8000 -e OPENAI_API_KEY=sk-... -e API_KEY=dev document-analyzer:local
#
# Secrets are NEVER baked in: OPENAI_API_KEY and API_KEY come from the environment
# (/opt/document-analyzer/.env on the Orange Pi).

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# ── 1. build: resolve dependencies with uv into /app/.venv ───────────────────────
FROM ${PYTHON_IMAGE} AS build
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken
WORKDIR /app

# Dependencies first (cached layer), project code later
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# Pre-download the tiktoken encoding so the container does not fetch it at startup
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# ── 2. runtime: only the virtualenv + application code ───────────────────────────
FROM ${PYTHON_IMAGE} AS runtime
RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/.tiktoken /app/.tiktoken
COPY app ./app

# /app/data holds the persistent Chroma catalog index (mounted as a volume)
RUN mkdir -p /app/data && chown -R app:app /app/data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken \
    ANONYMIZED_TELEMETRY=False \
    APP_ENV=production

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# One worker on purpose: the LangGraph checkpointer (GET /Document/Analyze/{id}) lives in memory
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
