"""Extraction schemas: official IDs (INE voter card, passport)."""

from pydantic import BaseModel, Field


class DomicilioIdentificacion(BaseModel):
    DireccionCompleta: str = Field(description="Domicilio tal como aparece en la credencial.")
    CodigoPostal: str
    Municipio: str
    Estado: str


class CredencialINE(BaseModel):
    """Mexican voter ID card (INE/IFE)."""

    Nombre: str = Field(description="Nombre(s) de pila.")
    ApellidoPaterno: str
    ApellidoMaterno: str
    Sexo: str = Field(description="H o M.")
    FechaNacimiento: str = Field(description="DD/MM/AAAA")
    CURP: str
    ClaveElector: str = Field(description="Clave de elector (18 caracteres).")
    AnioRegistro: str = Field(description="Año de registro, p. ej. '2010 01'.")
    Seccion: str
    Emision: str = Field(description="Año de emisión.")
    Vigencia: str = Field(description="Año o rango de vigencia.")
    Domicilio: DomicilioIdentificacion


class Pasaporte(BaseModel):
    """Mexican passport."""

    Tipo: str = Field(description="Tipo de pasaporte, p. ej. 'P'.")
    ClavePais: str = Field(description="Clave del país emisor, p. ej. 'MEX'.")
    NumeroPasaporte: str
    Apellidos: str
    Nombres: str
    Nacionalidad: str
    FechaNacimiento: str = Field(description="DD/MM/AAAA")
    CURP: str
    Sexo: str
    LugarNacimiento: str
    FechaExpedicion: str = Field(description="DD/MM/AAAA")
    FechaCaducidad: str = Field(description="DD/MM/AAAA")
    Autoridad: str
