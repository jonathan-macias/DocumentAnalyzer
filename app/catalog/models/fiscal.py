"""Extraction schema: Constancia de Situación Fiscal (SAT tax status certificate).

PascalCase field names are the output contract (the `result` block of the response).
Each Field `description` is part of the prompt: the LLM reads it through structured output,
so descriptions are kept in Spanish to match the source documents.
`ActividadEconomicaMayorPorcentaje` and `PrimerRegimen` are NOT extracted: they are computed
in postprocess.
"""

from pydantic import BaseModel, Field


class DatosEmision(BaseModel):
    Lugar: str = Field(description="Municipio y entidad de emisión, p. ej. 'TEPIC, NAYARIT'.")
    Fecha: str = Field(description="Fecha de emisión DD/MM/AAAA.")


class DatosIdentificacion(BaseModel):
    RFC: str = Field(description="Registro Federal de Contribuyentes (12 o 13 caracteres).")
    CURP: str = Field(description="CURP (18 caracteres). Vacío para personas morales.")
    Nombre: str = Field(description="Nombre(s) de pila. Para personas morales, la razón social.")
    ApellidoPaterno: str = Field(description="Primer apellido.")
    ApellidoMaterno: str = Field(description="Segundo apellido.")
    FechaInicioOperaciones: str = Field(description="Fecha inicio de operaciones DD/MM/AAAA.")
    EstatusPadron: str = Field(description="Estatus en el padrón, p. ej. ACTIVO.")
    FechaUltimoCambioEstado: str = Field(description="Fecha de último cambio de estado DD/MM/AAAA.")
    NombreComercial: str = Field(description="Nombre comercial; vacío si no aparece.")


class DatosDomicilio(BaseModel):
    Direccion: str = Field(description="Dirección completa en una línea SOLO si el documento la muestra así; si no, vacío.")
    TipoVialidad: str = Field(description="Tipo de vialidad (CALLE, AVENIDA...).")
    Calle: str = Field(description="Nombre de vialidad.")
    NumeroExterior: str
    NumeroInterior: str
    Cruzamientos: str = Field(description="Cruzamientos, si aparecen.")
    EntreCalle: str
    YCalle: str
    Colonia: str = Field(description="Nombre de la colonia.")
    CodigoPostal: str
    Localidad: str = Field(description="Nombre de la localidad.")
    Municipio: str = Field(description="Municipio o demarcación territorial.")
    Estado: str = Field(description="Entidad federativa.")


class ActividadEconomica(BaseModel):
    Orden: int
    ActividadEconomica: str
    Porcentaje: int
    FechaInicio: str = Field(description="DD/MM/AAAA")
    FechaFin: str = Field(description="DD/MM/AAAA o vacío")


class Regimen(BaseModel):
    Regimen: str
    FechaInicio: str = Field(description="DD/MM/AAAA")
    FechaFin: str = Field(description="DD/MM/AAAA o vacío")


class ConstanciaSituacionFiscal(BaseModel):
    """Data from a SAT Constancia de Situación Fiscal."""

    DatosEmision: DatosEmision
    DatosIdentificacion: DatosIdentificacion
    DatosDomicilio: DatosDomicilio
    ActividadesEconomicas: list[ActividadEconomica] = Field(
        description="Tabla 'Actividades Económicas', en el orden del documento."
    )
    Regimenes: list[Regimen] = Field(description="Tabla 'Regímenes', en el orden del documento.")
