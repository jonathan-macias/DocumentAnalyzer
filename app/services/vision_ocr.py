"""OCR through an OpenAI vision model for pages without a text layer (scans, photos).

Each page is rendered to PNG and sent as a multimodal message (content blocks).
Pages are processed concurrently with asyncio.gather.
"""

import asyncio
import base64
import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.llm.prompts import VISION_OCR_PROMPT
from app.services.pdf_parser import ParsedDocument, render_page_png

logger = logging.getLogger(__name__)


async def _ocr_page(model: BaseChatModel, png: bytes) -> str:
    b64 = base64.b64encode(png).decode("ascii")
    message = HumanMessage(
        content=[
            {"type": "text", "text": VISION_OCR_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
    )
    response = await model.ainvoke([message])
    return response.text.strip() if hasattr(response, "text") else str(response.content)


async def ocr_pages_without_text(
    parsed: ParsedDocument, model: BaseChatModel, dpi: int = 150
) -> list[int]:
    """Fill `text` for pages flagged with `needs_ocr`. Returns the processed page numbers."""
    targets = [p for p in parsed.pages if p.needs_ocr]
    if not targets:
        return []
    logger.info("Running Vision OCR on %d page(s)", len(targets))
    pngs = [render_page_png(parsed.pdf_bytes, p.page_number, dpi) for p in targets]
    texts = await asyncio.gather(*(_ocr_page(model, png) for png in pngs))
    for page, text in zip(targets, texts, strict=True):
        page.text = text
        page.ocr_applied = True
    return [p.page_number for p in targets]
