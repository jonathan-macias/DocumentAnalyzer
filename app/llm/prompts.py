"""Classification and extraction prompts (ChatPromptTemplate).

Prompt text is intentionally in Spanish: source documents are Mexican KYC documents and the
model's reasoning is returned to API clients in Spanish.
"""

from langchain_core.prompts import ChatPromptTemplate

CATEGORY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un analista experto en documentos KYC de México. Clasifica el documento en "
            "UNA categoría del catálogo.\n\n"
            "Categorías permitidas:\n{categories}\n\n"
            "Tipos candidatos encontrados por búsqueda semántica en el catálogo "
            "(úsalos como pista, no como verdad):\n{candidates}\n\n"
            "Reglas:\n"
            "- Si el documento no corresponde a ninguna categoría, usa OTRO.\n"
            "- `score` es tu confianza entre 0 y 1 (1 = certeza total).\n"
            "- `reasoning` en español, 1-3 oraciones, citando evidencias del texto.",
        ),
        ("human", "Contenido del documento:\n<<<\n{document}\n>>>"),
    ]
)

TYPE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un analista experto en documentos KYC de México. El documento ya fue "
            "clasificado en la categoría {category}. Determina su TIPO específico.\n\n"
            "Tipos permitidos para esta categoría:\n{types}\n\n"
            "Reglas:\n"
            "- Si ninguno aplica, usa DESCONOCIDO.\n"
            "- `score` es tu confianza entre 0 y 1.\n"
            "- `reasoning` en español, 1-3 oraciones.",
        ),
        ("human", "Contenido del documento:\n<<<\n{document}\n>>>"),
    ]
)

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un sistema de extracción de datos para documentos de tipo {document_type}.\n"
            "Descripción del tipo: {type_description}\n\n"
            "Reglas de extracción:\n"
            "- Extrae SOLO lo que aparece en el texto. Si un dato no aparece, usa cadena vacía \"\".\n"
            "- No inventes ni completes datos.\n"
            "- Fechas en formato DD/MM/AAAA (convierte '21 DE ABRIL DE 2026' → '21/04/2026').\n"
            "- Respeta mayúsculas, acentos y la Ñ tal como aparecen.\n"
            "- Porcentajes y órdenes como números enteros.\n"
            "{extra_instructions}",
        ),
        (
            "human",
            "Texto del documento ({context_mode}):\n<<<\n{context}\n>>>",
        ),
    ]
)

VISION_OCR_PROMPT = (
    "Transcribe de forma literal TODO el texto visible de esta página de documento, "
    "en orden de lectura. Conserva etiquetas y valores en líneas 'Etiqueta: valor' y las "
    "tablas como filas separadas por ' | '. No resumas ni agregues comentarios."
)
