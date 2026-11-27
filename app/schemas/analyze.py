"""HTTP contract for POST /Document/Analyze.

API models use camelCase in JSON (aliases) and snake_case in Python.
The `result` block uses PascalCase because it comes from the catalog extraction models.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class AnalyzeRequest(CamelModel):
    document_url: str | None = Field(
        default=None,
        description="http/https URL of the document (e.g. Azure Blob with a SAS token).",
    )
    document_base64: str | None = Field(
        default=None,
        description="Document content as base64. A data:<mime>;base64, prefix is accepted.",
    )
    file_name: str | None = Field(default=None, description="Optional file name.")
    document_category: str | None = Field(
        default=None,
        description="Expected category (e.g. DOCUMENTO_FISCAL). Validated against the prediction.",
    )
    threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Minimum score. null → DEFAULT_THRESHOLD."
    )

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "AnalyzeRequest":
        has_url = bool(self.document_url and self.document_url.strip())
        has_b64 = bool(self.document_base64 and self.document_base64.strip())
        if has_url == has_b64:
            raise ValueError("Debes enviar exactamente uno de 'documentUrl' o 'documentBase64'.")
        if has_url and not self.document_url.lower().startswith(("http://", "https://")):
            raise ValueError("'documentUrl' debe usar http o https.")
        return self

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "documentUrl": "https://example.blob.core.windows.net/container/CSF.pdf?sv=...",
                    "documentCategory": "DOCUMENTO_FISCAL",
                    "threshold": None,
                }
            ]
        },
    )


class AnalyzeInput(CamelModel):
    """Echo of the request (without the full base64) with the effective threshold."""

    document_url: str | None = None
    document_base64: str | None = None
    file_name: str | None = None
    document_category: str | None = None
    threshold: float


class Prediction(CamelModel):
    document_category: str | None = None
    document_type: str | None = None
    category_prediction_score: float | None = None
    type_prediction_score: float | None = None
    category_reasoning: str | None = None
    type_reasoning: str | None = None


class PageInfo(CamelModel):
    page_number: int
    angle: float
    width: float
    height: float
    unit: Literal["inch"] = "inch"
    orientation: Literal["Portrait", "Landscape"]


class AnalyzeResponse(CamelModel):
    success: bool
    document_task_id: str
    input: AnalyzeInput
    prediction: Prediction | None = None
    is_valid_category: bool = False
    is_valid_type: bool = False
    result: dict[str, Any] | None = None
    pages: list[PageInfo] = Field(default_factory=list)
    total_pages: int = 0
    error_message: str | None = None
