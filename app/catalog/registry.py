"""Document catalog registry (loads the YAML files in `definitions/`).

Adding a new document type:
1. Create its Pydantic model in app/catalog/models/ and register it in EXTRACTION_MODELS.
2. Add the entry to a YAML file in definitions/ (or create a new file for a new category).
3. Restart the API: the catalog semantic index is rebuilt only if the YAML hash changed.
"""

import hashlib
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

from app.catalog.models import EXTRACTION_MODELS

DEFINITIONS_DIR = Path(__file__).parent / "definitions"

OTHER_CATEGORY = "OTRO"
UNKNOWN_TYPE = "DESCONOCIDO"


class DocumentTypeDef(BaseModel):
    type: str
    category: str = ""
    description: str
    keywords: list[str] = Field(default_factory=list)
    examples: list[str] = Field(default_factory=list)
    extraction_model: str | None = None
    postprocessor: str | None = None
    extra_instructions: str = ""
    retrieval_queries: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _model_exists(self) -> "DocumentTypeDef":
        if self.extraction_model and self.extraction_model not in EXTRACTION_MODELS:
            raise ValueError(
                f"extraction_model '{self.extraction_model}' for {self.type} is not registered."
            )
        return self

    @property
    def model(self) -> type[BaseModel] | None:
        return EXTRACTION_MODELS.get(self.extraction_model) if self.extraction_model else None

    def embedding_text(self) -> str:
        """Text embedded into the catalog semantic index."""
        return (
            f"Tipo de documento: {self.type} (categoría {self.category}).\n"
            f"{self.description.strip()}\n"
            f"Palabras clave: {', '.join(self.keywords)}\n"
            f"Fragmentos típicos: {' | '.join(self.examples)}"
        )


class CategoryDef(BaseModel):
    category: str
    description: str
    types: list[DocumentTypeDef]


class CatalogRegistry:
    def __init__(self, categories: list[CategoryDef], fingerprint: str) -> None:
        self.categories: dict[str, CategoryDef] = {c.category: c for c in categories}
        self.types: dict[str, DocumentTypeDef] = {}
        for cat in categories:
            for t in cat.types:
                t.category = cat.category
                if t.type in self.types:
                    raise ValueError(f"Duplicated document type in catalog: {t.type}")
                self.types[t.type] = t
        self.fingerprint = fingerprint

    @classmethod
    def load(cls, directory: Path = DEFINITIONS_DIR) -> "CatalogRegistry":
        files = sorted(directory.glob("*.yaml"))
        if not files:
            raise RuntimeError(f"No catalog definitions found in {directory}")
        digest = hashlib.sha256()
        categories: list[CategoryDef] = []
        for f in files:
            raw = f.read_bytes()
            digest.update(raw)
            categories.append(CategoryDef.model_validate(yaml.safe_load(raw)))
        return cls(categories, digest.hexdigest()[:16])

    # ---- helpers for prompts / dynamic Literals ----
    def category_codes(self) -> list[str]:
        return [*self.categories.keys(), OTHER_CATEGORY]

    def type_codes(self, category: str) -> list[str]:
        cat = self.categories.get(category)
        return [*(t.type for t in cat.types), UNKNOWN_TYPE] if cat else [UNKNOWN_TYPE]

    def describe_categories(self) -> str:
        lines = [f"- {c.category}: {c.description.strip()}" for c in self.categories.values()]
        lines.append(f"- {OTHER_CATEGORY}: el documento no pertenece a ninguna categoría anterior.")
        return "\n".join(lines)

    def describe_types(self, category: str) -> str:
        cat = self.categories.get(category)
        lines = [f"- {t.type}: {t.description.strip()}" for t in cat.types] if cat else []
        lines.append(f"- {UNKNOWN_TYPE}: ninguno de los anteriores.")
        return "\n".join(lines)
