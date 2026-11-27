"""Extraction schema: proof of address (electricity, water, phone/internet bills)."""

from pydantic import BaseModel, Field


class DomicilioServicio(BaseModel):
    DireccionCompleta: str = Field(description="Dirección del servicio tal como aparece.")
    Calle: str
    NumeroExterior: str
    NumeroInterior: str
    Colonia: str
    CodigoPostal: str
    Municipio: str
    Estado: str


class ComprobanteDomicilio(BaseModel):
    """Utility bill used as proof of address."""

    Emisor: str = Field(description="Empresa que emite el recibo (CFE, Telmex, organismo de agua...).")
    Titular: str = Field(description="Nombre del titular del servicio.")
    NumeroServicio: str = Field(description="Número de servicio, cuenta, contrato o teléfono.")
    Domicilio: DomicilioServicio
    FechaEmision: str = Field(description="DD/MM/AAAA o vacío.")
    PeriodoFacturado: str = Field(description="Periodo facturado tal como aparece.")
    FechaLimitePago: str = Field(description="DD/MM/AAAA o vacío.")
    TotalAPagar: str = Field(description="Importe total a pagar con formato original, p. ej. '$1,234.00'.")
