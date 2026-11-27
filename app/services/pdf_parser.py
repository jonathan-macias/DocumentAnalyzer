"""PyMuPDF parsing: per-page text + page metadata (as returned in the response).

- width/height: PyMuPDF reports typographic points (1 inch = 72 pt) → converted to inches.
- angle: declared page rotation in degrees (0, 90, 180, 270).
- Images (PNG/JPG/...) are converted into a one-page PDF so there is a single code path.
- needs_ocr: the page has almost no text AND is mostly covered by images (a typical scan).
  A page that only holds a small QR code does NOT trigger OCR.
"""

import re
from dataclasses import dataclass, field

import pymupdf

from app.schemas.analyze import PageInfo

POINTS_PER_INCH = 72.0
IMAGE_COVERAGE_FOR_OCR = 0.30


@dataclass
class ParsedPage:
    page_number: int
    text: str
    info: PageInfo
    needs_ocr: bool = False
    ocr_applied: bool = False


@dataclass
class ParsedDocument:
    pages: list[ParsedPage] = field(default_factory=list)
    pdf_bytes: bytes = b""

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"--- Página {p.page_number} ---\n{p.text}" for p in self.pages)


class PdfParseError(Exception):
    pass


def normalize_text(text: str) -> str:
    """Compact layout whitespace (saves tokens) while keeping the line structure."""
    text = re.sub(r"[ \t]{3,}", "   ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def to_pdf_bytes(content: bytes, mime_type: str) -> bytes:
    if mime_type == "application/pdf":
        return content
    try:
        with pymupdf.open(stream=content, filetype=mime_type.split("/")[-1]) as img:
            return img.convert_to_pdf()
    except Exception as exc:  # noqa: BLE001
        raise PdfParseError("No se pudo leer la imagen.") from exc


def _image_coverage(page: pymupdf.Page) -> float:
    """Fraction of the page area covered by images (0..1)."""
    page_area = abs(page.rect) or 1.0
    covered = 0.0
    for info in page.get_image_info():
        bbox = pymupdf.Rect(info["bbox"]) & page.rect
        covered += abs(bbox)
    return min(covered / page_area, 1.0)


def parse_document(content: bytes, mime_type: str, min_chars_per_page: int = 20) -> ParsedDocument:
    pdf_bytes = to_pdf_bytes(content, mime_type)
    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise PdfParseError("El archivo PDF está dañado o no se puede abrir.") from exc

    if doc.needs_pass:
        doc.close()
        raise PdfParseError("El PDF está protegido con contraseña.")

    parsed = ParsedDocument(pdf_bytes=pdf_bytes)
    with doc:
        for index, page in enumerate(doc, start=1):
            text = normalize_text(page.get_text("text", sort=True))
            # page.rect already accounts for rotation → it is the size "as displayed"
            width_in = round(page.rect.width / POINTS_PER_INCH, 4)
            height_in = round(page.rect.height / POINTS_PER_INCH, 4)
            info = PageInfo(
                page_number=index,
                angle=float(page.rotation),
                width=width_in,
                height=height_in,
                unit="inch",
                orientation="Landscape" if width_in > height_in else "Portrait",
            )
            needs_ocr = (
                len(text) < min_chars_per_page
                and _image_coverage(page) >= IMAGE_COVERAGE_FOR_OCR
            )
            parsed.pages.append(
                ParsedPage(page_number=index, text=text, info=info, needs_ocr=needs_ocr)
            )

    if not parsed.pages:
        raise PdfParseError("El documento no contiene páginas.")
    return parsed


def render_page_png(pdf_bytes: bytes, page_number: int, dpi: int = 150) -> bytes:
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        pix = doc[page_number - 1].get_pixmap(dpi=dpi)
        return pix.tobytes("png")
