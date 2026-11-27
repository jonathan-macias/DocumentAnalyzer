"""API key authentication for the /Document/* routes.

Clients send the key in the `X-API-Key` header. The expected value comes from the
`API_KEY` setting (only stored in the server's .env). When it is not configured
(local development) authentication is disabled; in production the app refuses to start
without it (see app/main.py).
"""

import secrets

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API key required to analyze documents and read results.",
)


async def require_api_key(request: Request, api_key: str | None = Security(api_key_header)) -> None:
    settings = getattr(request.app.state, "settings", None)
    expected = getattr(settings, "api_key", None)
    if not expected:
        return  # authentication disabled (development)
    # Constant-time comparison to avoid timing attacks
    if not api_key or not secrets.compare_digest(api_key.encode(), expected.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key inválida o ausente (header X-API-Key).",
            headers={"WWW-Authenticate": "ApiKey"},
        )
