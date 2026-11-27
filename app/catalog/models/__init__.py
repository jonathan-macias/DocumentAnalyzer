"""Extraction model registry. Catalog YAML files reference these models by name."""

from pydantic import BaseModel

from app.catalog.models.bancario import EstadoCuenta
from app.catalog.models.domicilio import ComprobanteDomicilio
from app.catalog.models.fiscal import ConstanciaSituacionFiscal
from app.catalog.models.identificacion import CredencialINE, Pasaporte

EXTRACTION_MODELS: dict[str, type[BaseModel]] = {
    "ConstanciaSituacionFiscal": ConstanciaSituacionFiscal,
    "CredencialINE": CredencialINE,
    "Pasaporte": Pasaporte,
    "ComprobanteDomicilio": ComprobanteDomicilio,
    "EstadoCuenta": EstadoCuenta,
}
