"""Deterministic post-processing of the extraction result.

The LLM extracts; the code normalizes. Anything that can be computed with rules is NOT asked
of the LLM (cheaper, deterministic and testable): dates, whitespace, derived fields.
"""

import re
from collections.abc import Callable
from typing import Any

MONTHS = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4, "MAYO": 5, "JUNIO": 6,
    "JULIO": 7, "AGOSTO": 8, "SEPTIEMBRE": 9, "SETIEMBRE": 9, "OCTUBRE": 10,
    "NOVIEMBRE": 11, "DICIEMBRE": 12,
}  # fmt: skip

_LONG_DATE_RE = re.compile(
    r"^\s*(\d{1,2})\s+DE\s+([A-ZÁÉÍÓÚ]+)\s+(?:DE|DEL)\s+(\d{4})\s*$", re.IGNORECASE
)
_ISO_DATE_RE = re.compile(r"^\s*(\d{4})-(\d{2})-(\d{2})\s*$")


def normalize_date(value: str) -> str:
    """'21 DE ABRIL DE 2026' → '21/04/2026'; '2026-04-21' → '21/04/2026'. Anything else → as is."""
    if m := _LONG_DATE_RE.match(value):
        day, month_name, year = m.groups()
        month = MONTHS.get(month_name.upper())
        if month:
            return f"{int(day):02d}/{month:02d}/{year}"
    if m := _ISO_DATE_RE.match(value):
        year, month, day = m.groups()
        return f"{day}/{month}/{year}"
    return value


def clean_string(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    value = re.sub(r"\s+,", ",", value)  # "TEPIC , NAYARIT" → "TEPIC, NAYARIT"
    return normalize_date(value)


def normalize_values(data: Any) -> Any:
    """Walk dicts/lists recursively and clean every string."""
    if isinstance(data, dict):
        return {k: normalize_values(v) for k, v in data.items()}
    if isinstance(data, list):
        return [normalize_values(v) for v in data]
    if isinstance(data, str):
        return clean_string(data)
    return data


# --------------------------------------------------------------------------- per document type


def postprocess_csf(result: dict[str, Any]) -> dict[str, Any]:
    """Derived fields for the Constancia de Situación Fiscal."""
    actividades = sorted(
        result.get("ActividadesEconomicas") or [], key=lambda a: a.get("Orden") or 0
    )
    result["ActividadesEconomicas"] = actividades
    # Highest percentage wins; on a tie, the lowest "Orden" wins
    result["ActividadEconomicaMayorPorcentaje"] = (
        max(actividades, key=lambda a: (a.get("Porcentaje") or 0, -(a.get("Orden") or 0)))
        if actividades
        else None
    )
    regimenes = result.get("Regimenes") or []
    result["PrimerRegimen"] = regimenes[0] if regimenes else None
    return result


POSTPROCESSORS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "csf": postprocess_csf,
}


def run_postprocess(result: dict[str, Any], postprocessor: str | None) -> dict[str, Any]:
    result = normalize_values(result)
    if postprocessor and postprocessor in POSTPROCESSORS:
        result = POSTPROCESSORS[postprocessor](result)
    return result
