"""Ingestion: URL or base64 → bytes + mime type.

Documents arrive over the network or embedded in the JSON body, so everything works on
in-memory bytes instead of file paths.
"""

import asyncio
import base64
import binascii
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from app.core.logging import mask_url

logger = logging.getLogger(__name__)

_DATA_URI_RE = re.compile(r"^data:(?P<mime>[\w/+.-]+)?;base64,", re.IGNORECASE)

SUPPORTED_MIME = {"application/pdf", "image/png", "image/jpeg", "image/tiff", "image/webp"}


class DocumentLoadError(Exception):
    """Business error while fetching the document (reported as success=false)."""


@dataclass
class LoadedDocument:
    content: bytes
    mime_type: str
    source: str  # "url" | "base64"


def sniff_mime(data: bytes) -> str | None:
    """Detect the file type from magic bytes (extensions and headers are not trusted)."""
    if data.startswith(b"%PDF"):
        return "application/pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] in (b"II*\x00", b"MM\x00*"):
        return "image/tiff"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _validate(data: bytes, max_bytes: int) -> str:
    if not data:
        raise DocumentLoadError("El documento está vacío.")
    if len(data) > max_bytes:
        raise DocumentLoadError(
            f"El documento excede el tamaño máximo permitido ({max_bytes // (1024 * 1024)} MB)."
        )
    mime = sniff_mime(data)
    if mime not in SUPPORTED_MIME:
        raise DocumentLoadError("Formato no soportado. Se aceptan PDF, PNG, JPEG, TIFF y WEBP.")
    return mime


def decode_base64(payload: str, max_bytes: int) -> LoadedDocument:
    payload = payload.strip()
    payload = _DATA_URI_RE.sub("", payload)
    payload = re.sub(r"\s+", "", payload)
    # Tolerate base64 without padding
    payload += "=" * (-len(payload) % 4)
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise DocumentLoadError("El contenido de 'documentBase64' no es base64 válido.") from exc
    mime = _validate(data, max_bytes)
    return LoadedDocument(content=data, mime_type=mime, source="base64")


def assert_public_host(url: str) -> None:
    """SSRF guard: reject URLs whose host resolves to a non-public address.

    Blocks loopback (127.0.0.1, localhost), private LAN ranges (192.168.x.x, 10.x, 172.16/12),
    link-local (169.254.x.x, cloud metadata) and other reserved ranges, so a caller cannot use
    the server to probe the network it runs in.
    """
    host = urlsplit(url).hostname
    if not host:
        raise DocumentLoadError("La URL del documento no es válida.")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise DocumentLoadError("No se pudo resolver el host del documento.") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            raise DocumentLoadError("La URL apunta a una red privada o local; no está permitida.")


async def download(
    url: str, max_bytes: int, timeout: float, allow_private: bool = False
) -> LoadedDocument:
    if not url.lower().startswith(("http://", "https://")):
        raise DocumentLoadError("Solo se permiten URLs http/https.")

    async def check_request(request: httpx.Request) -> None:
        # Runs for the first request AND for every redirect hop
        if not allow_private:
            await asyncio.to_thread(assert_public_host, str(request.url))

    logger.info("Downloading document from %s", mask_url(url))
    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            max_redirects=5,
            event_hooks={"request": [check_request]},
        ) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise DocumentLoadError(
                        f"No se pudo descargar el documento (HTTP {resp.status_code})."
                    )
                chunks: list[bytes] = []
                size = 0
                # Stream and abort early once the size limit is exceeded
                async for chunk in resp.aiter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise DocumentLoadError(
                            "El documento excede el tamaño máximo permitido "
                            f"({max_bytes // (1024 * 1024)} MB)."
                        )
                    chunks.append(chunk)
    except httpx.TimeoutException as exc:
        raise DocumentLoadError("Tiempo de espera agotado al descargar el documento.") from exc
    except httpx.HTTPError as exc:
        raise DocumentLoadError(f"Error de red al descargar el documento: {type(exc).__name__}") from exc

    data = b"".join(chunks)
    mime = _validate(data, max_bytes)
    return LoadedDocument(content=data, mime_type=mime, source="url")
