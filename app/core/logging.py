"""Logging with masking of sensitive data.

URLs carrying a SAS token (`?sv=...&sig=...`) are credentials: they are never logged in full.
"""

import logging
from urllib.parse import urlsplit, urlunsplit


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def mask_url(url: str | None) -> str:
    """Strip the query string (where SAS tokens live) and the fragment."""
    if not url:
        return ""
    parts = urlsplit(str(url))
    masked_query = "***" if parts.query else ""
    return urlunsplit((parts.scheme, parts.netloc, parts.path, masked_query, ""))
