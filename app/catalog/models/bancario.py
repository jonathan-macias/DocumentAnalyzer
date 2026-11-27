"""Extraction schema: bank account statement."""

from pydantic import BaseModel, Field


class PeriodoEstadoCuenta(BaseModel):
    FechaInicio: str = Field(description="DD/MM/AAAA")
    FechaFin: str = Field(description="DD/MM/AAAA")


class EstadoCuenta(BaseModel):
    """Bank account statement (individual or company)."""

    Banco: str
    Titular: str
    RFC: str = Field(description="RFC del titular si aparece.")
    NumeroCuenta: str
    CLABE: str = Field(description="CLABE interbancaria (18 dígitos) si aparece.")
    Domicilio: str = Field(description="Domicilio del titular tal como aparece.")
    Periodo: PeriodoEstadoCuenta
    FechaCorte: str = Field(description="DD/MM/AAAA o vacío.")
    SaldoInicial: str = Field(description="Importe con formato original.")
    SaldoFinal: str = Field(description="Importe con formato original.")
    TotalDepositos: str = Field(description="Importe con formato original.")
    TotalRetiros: str = Field(description="Importe con formato original.")
