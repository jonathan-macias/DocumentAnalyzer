"""Generate fictitious sample documents for every catalog document type.

Usage:
    uv run python scripts/generate_samples.py

Output (overwritten on every run):
    samples/*.pdf              one PDF per catalog type + edge cases (scan, long doc, "OTRO")
    samples/expected/*.json    expected classification and key fields (golden files)

Rules:
- Every person, company, address, ID number, account and amount is FICTITIOUS.
- No logos, QR codes, barcodes, photos, seals or digital signatures are reproduced.
- Every page carries the footer watermark "DOCUMENTO DE PRUEBA - DATOS FICTICIOS".
- Labels and layout mimic the real documents so classification/extraction behave the same.
- Generation is deterministic (fixed random seeds and PDF metadata).
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pymupdf

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "samples"
EXPECTED_DIR = OUT_DIR / "expected"

LETTER = (612.0, 792.0)
WATERMARK = "DOCUMENTO DE PRUEBA - DATOS FICTICIOS"

FONT = "helv"
BOLD = "hebo"
MONO = "cour"

BLACK = (0, 0, 0)
GRAY_TEXT = (0.45, 0.45, 0.45)
GRAY_FILL = (0.88, 0.88, 0.88)
LIGHT_FILL = (0.95, 0.95, 0.95)
LINE = (0.55, 0.55, 0.55)
DARK = (0.19, 0.21, 0.23)

# The long statement must exceed FULL_CONTEXT_MAX_TOKENS (12k) so "auto" uses hybrid retrieval
LONG_STATEMENT_MOVEMENTS = 650


# =============================================================================== layout helpers


_FONTS: dict[str, pymupdf.Font] = {}


def text_width(text: str, size: float, font: str = FONT) -> float:
    # pymupdf.Font measures accented characters correctly (get_text_length does not)
    if font not in _FONTS:
        _FONTS[font] = pymupdf.Font(font)
    return _FONTS[font].text_length(text, fontsize=size)


def wrap(text: str, width: float, size: float, font: str = FONT) -> list[str]:
    """Greedy word wrap using real font metrics."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split(" "):
            candidate = f"{current} {word}".strip()
            if text_width(candidate, size, font) <= width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


class PdfBuilder:
    """Tiny flow-layout helper on top of PyMuPDF (cursor `y` moves down the page)."""

    def __init__(self, size: tuple[float, float] = LETTER, margin: float = 40.0) -> None:
        self.doc = pymupdf.open()
        self.w, self.h = size
        self.m = margin
        self.page: pymupdf.Page
        self.y = 0.0
        self.on_new_page: Callable[[], None] | None = None
        self.new_page()

    # ---- page flow ----
    @property
    def width(self) -> float:
        return self.w - 2 * self.m

    @property
    def bottom(self) -> float:
        return self.h - self.m - 24  # room for the footer

    def new_page(self) -> None:
        self.page = self.doc.new_page(width=self.w, height=self.h)
        self.y = self.m
        if self.on_new_page:
            self.on_new_page()

    def ensure(self, height: float) -> None:
        if self.y + height > self.bottom:
            self.new_page()

    def space(self, height: float = 8) -> None:
        self.y += height

    # ---- primitives ----
    def write(self, x: float, baseline: float, text: str, size: float = 9, font: str = FONT,
              color: tuple = BLACK) -> None:  # fmt: skip
        self.page.insert_text((x, baseline), text, fontsize=size, fontname=font, color=color)

    def cell_text(self, rect: pymupdf.Rect, text: str, size: float = 8.5, font: str = FONT,
                  align: str = "left", color: tuple = BLACK, pad: float = 5) -> None:  # fmt: skip
        """Single-line text vertically centred in a cell; fails loudly if it does not fit."""
        tw = text_width(text, size, font)
        if tw > rect.width - 2 * pad + 0.5:
            raise ValueError(f"Text does not fit in cell ({tw:.0f} > {rect.width:.0f}): {text!r}")
        if align == "center":
            x = rect.x0 + (rect.width - tw) / 2
        elif align == "right":
            x = rect.x1 - pad - tw
        else:
            x = rect.x0 + pad
        baseline = rect.y0 + rect.height / 2 + size * 0.35
        self.write(x, baseline, text, size, font, color)

    def lines(self, x: float, items: list[tuple[str, float, str]], leading: float = 1.3,
              color: tuple = BLACK) -> None:  # fmt: skip
        """Write (text, size, font) lines starting at the cursor."""
        for text, size, font in items:
            self.y += size * leading
            self.write(x, self.y, text, size, font, color)
        self.y += 4

    def paragraph(self, text: str, size: float = 8.5, font: str = FONT, x: float | None = None,
                  width: float | None = None, color: tuple = BLACK) -> None:  # fmt: skip
        x = self.m if x is None else x
        width = self.width if width is None else width
        for line in wrap(text, width, size, font):
            self.ensure(size * 1.4)
            self.y += size * 1.35
            self.write(x, self.y, line, size, font, color)
        self.y += 4

    # ---- blocks ----
    def dark_band(self, height: float = 34) -> None:
        self.page.draw_rect(pymupdf.Rect(self.m, self.y, self.w - self.m, self.y + height),
                            color=DARK, fill=DARK)  # fmt: skip
        self.y += height + 12

    def heading(self, text: str, size: float = 10.5) -> None:
        self.ensure(40)
        rect = pymupdf.Rect(self.m, self.y, self.w - self.m, self.y + 20)
        self.page.draw_rect(rect, color=LINE, fill=GRAY_FILL, width=0.5)
        self.cell_text(rect, text, size=size, font=BOLD)
        self.y += 20

    def kv_rows(self, rows: list[tuple[str, str]], label_ratio: float = 0.42,
                row_h: float = 19, size: float = 8.5) -> None:  # fmt: skip
        """Two-cell rows: | Label: | value |  (like the SAT identification table)."""
        x0, x1 = self.m, self.w - self.m
        xm = x0 + (x1 - x0) * label_ratio
        for label, value in rows:
            self.ensure(row_h)
            left = pymupdf.Rect(x0, self.y, xm, self.y + row_h)
            right = pymupdf.Rect(xm, self.y, x1, self.y + row_h)
            for rect in (left, right):
                self.page.draw_rect(rect, color=LINE, width=0.5)
            self.cell_text(left, label, size=size, font=BOLD)
            self.cell_text(right, value, size=size)
            self.y += row_h

    def kv_grid(self, rows: list[tuple[str, str]], cols: int = 2, row_h: float = 19,
                size: float = 8.5) -> None:  # fmt: skip
        """Cells with inline 'Label: value', `cols` cells per row."""
        cw = self.width / cols
        for i in range(0, len(rows), cols):
            self.ensure(row_h)
            for j, (label, value) in enumerate(rows[i : i + cols]):
                rect = pymupdf.Rect(self.m + j * cw, self.y, self.m + (j + 1) * cw, self.y + row_h)
                self.page.draw_rect(rect, color=LINE, width=0.5)
                baseline = rect.y0 + row_h / 2 + size * 0.35
                self.write(rect.x0 + 5, baseline, label, size, BOLD)
                lx = rect.x0 + 5 + text_width(label, size, BOLD) + 5
                if lx + text_width(value, size) > rect.x1 - 3:
                    raise ValueError(f"Grid cell overflow: {label} {value}")
                self.write(lx, baseline, value, size, FONT)
            self.y += row_h

    def table(self, headers: list[str], rows: list[list[str]], widths: list[float],
              aligns: list[str] | None = None, size: float = 8, repeat_header: bool = True) -> None:  # fmt: skip
        """Bordered table; cells wrap; header is repeated on page breaks."""
        aligns = aligns or ["left"] * len(headers)
        xs = [self.m]
        for frac in widths:
            xs.append(xs[-1] + self.width * frac)

        def draw_header() -> None:
            self.ensure(20)
            for k, head in enumerate(headers):
                rect = pymupdf.Rect(xs[k], self.y, xs[k + 1], self.y + 18)
                self.page.draw_rect(rect, color=LINE, fill=LIGHT_FILL, width=0.5)
                self.cell_text(rect, head, size=size + 0.5, font=BOLD, align="center", pad=3)
            self.y += 18

        draw_header()
        for row in rows:
            wrapped = [wrap(cell, xs[k + 1] - xs[k] - 8, size) for k, cell in enumerate(row)]
            height = max(len(w) for w in wrapped) * size * 1.25 + 7
            if self.y + height > self.bottom:
                self.new_page()
                if repeat_header:
                    draw_header()
            for k, cell_lines in enumerate(wrapped):
                rect = pymupdf.Rect(xs[k], self.y, xs[k + 1], self.y + height)
                self.page.draw_rect(rect, color=LINE, width=0.5)
                for n, line in enumerate(cell_lines):
                    baseline = rect.y0 + 4 + size + n * size * 1.25
                    tw = text_width(line, size)
                    if aligns[k] == "right":
                        x = rect.x1 - 4 - tw
                    elif aligns[k] == "center":
                        x = rect.x0 + (rect.width - tw) / 2
                    else:
                        x = rect.x0 + 4
                    self.write(x, baseline, line, size)
            self.y += height

    def placeholder(self, rect: pymupdf.Rect, label: str) -> None:
        """Gray box standing in for a photo/signature (never reproduce real ones)."""
        self.page.draw_rect(rect, color=LINE, fill=LIGHT_FILL, width=0.6)
        self.cell_text(rect, label, size=7, color=GRAY_TEXT, align="center", pad=2)

    # ---- output ----
    def finish(self, page_labels: bool = False) -> pymupdf.Document:
        total = self.doc.page_count
        for i, page in enumerate(self.doc, start=1):
            mark_w = text_width(WATERMARK, 7)
            page.insert_text(((self.w - mark_w) / 2, self.h - 16), WATERMARK, fontsize=7,
                             fontname=FONT, color=(0.6, 0.6, 0.6))  # fmt: skip
            if page_labels:
                label = f"Página [{i}] de [{total}]"
                lw = text_width(label, 8, BOLD)
                page.insert_text((self.w - self.m - lw, self.h - 30), label, fontsize=8, fontname=BOLD)
        return self.doc


def save(doc: pymupdf.Document, name: str) -> Path:
    doc.set_metadata(
        {
            "title": f"Documento de prueba - {name}",
            "author": "Document Analyze - sample generator",
            "subject": WATERMARK,
            "creator": "scripts/generate_samples.py",
            "producer": "PyMuPDF",
            "creationDate": "D:20261101000000",
            "modDate": "D:20261101000000",
        }
    )
    path = OUT_DIR / f"{name}.pdf"
    doc.save(path, garbage=4, deflate=True, no_new_id=True)
    doc.close()
    return path


def money(cents: int) -> str:
    return f"${cents / 100:,.2f}"


# =============================================================================== fictitious people

PERSONA_A = {  # CSF, CFE, short statement
    "nombre": "MARÍA FERNANDA", "paterno": "PRUEBA", "materno": "EJEMPLO",
    "rfc": "PUEM900315AB1", "curp": "PUEM900315MJCRJR09",
    "calle": "DE LAS PRUEBAS", "tipo_vialidad": "AVENIDA", "ext": "742", "int": "3B",
    "colonia": "AMERICANA", "cp": "44160", "municipio": "GUADALAJARA", "estado": "JALISCO",
}  # fmt: skip

PERSONA_B = {  # INE, water bill, long statement
    "nombre": "JOSÉ LUIS", "paterno": "MUESTRA", "materno": "PRUEBA",
    "rfc": "MUPJ850722K12", "curp": "MUPJ850722HDFSRS02", "clave_elector": "MSPRJS85072209H100",
}  # fmt: skip

PERSONA_C = {  # passport, phone bill
    "nombre": "SOFÍA ISABEL", "paterno": "PRUEBA", "materno": "CASOS",
    "curp": "PUCS921105MJCRSF08",
}  # fmt: skip


def full_name(p: dict[str, str]) -> str:
    return f"{p['nombre']} {p['paterno']} {p['materno']}"


# =============================================================================== documents


@dataclass
class Sample:
    name: str
    category: str
    doc_type: str
    extraction_path: str = "full"
    fields: dict[str, Any] = field(default_factory=dict)  # exact match (normalized)
    contains: dict[str, str] = field(default_factory=dict)  # substring match (normalized)
    counts: dict[str, int] = field(default_factory=dict)  # list lengths
    description: str = ""


def build_csf() -> tuple[pymupdf.Document, Sample]:
    p = PERSONA_A
    b = PdfBuilder()
    b.dark_band()

    # Header: left identification card + right title boxes (logos replaced by plain text)
    top = b.y
    left = pymupdf.Rect(b.m, top, b.m + 250, top + 190)
    b.page.draw_rect(left, color=LINE, width=0.6)
    head = pymupdf.Rect(left.x0, left.y0, left.x1, left.y0 + 22)
    b.page.draw_rect(head, color=LINE, fill=GRAY_FILL, width=0.6)
    b.cell_text(head, "CÉDULA DE IDENTIFICACIÓN FISCAL", size=10, font=BOLD, align="center")
    b.y = head.y1 + 4
    for text, size, font, color in [
        ("SERVICIO DE ADMINISTRACIÓN TRIBUTARIA", 8, BOLD, BLACK),
        (p["rfc"], 9, FONT, GRAY_TEXT),
        ("Registro Federal de Contribuyentes", 8, FONT, GRAY_TEXT),
        (f"{p['nombre']} {p['paterno']}", 9, FONT, GRAY_TEXT),
        (p["materno"], 9, FONT, GRAY_TEXT),
        ("Nombre, denominación o razón social", 8, FONT, GRAY_TEXT),
        ("idCIF: 00000000000", 8.5, FONT, GRAY_TEXT),
        ("VALIDA TU INFORMACIÓN FISCAL", 8, FONT, GRAY_TEXT),
    ]:
        b.y += size * 1.9
        b.write(left.x0 + (left.width - text_width(text, size, font)) / 2, b.y, text, size, font, color)

    right_x = b.m + 290
    title = pymupdf.Rect(right_x, top + 30, b.w - b.m, top + 72)
    b.page.draw_rect(title, color=LINE, width=0.6)
    b.cell_text(title, "CONSTANCIA DE SITUACIÓN FISCAL", size=12.5, font=BOLD, align="center")
    place = pymupdf.Rect(right_x, top + 88, b.w - b.m, top + 132)
    b.page.draw_rect(place, color=LINE, fill=LIGHT_FILL, width=0.6)
    b.write(place.x0 + (place.width - text_width("Lugar y Fecha de Emisión", 9)) / 2,
            place.y0 + 16, "Lugar y Fecha de Emisión", 9)  # fmt: skip
    emision = "GUADALAJARA , JALISCO A 15 DE OCTUBRE DE 2026"
    b.write(place.x0 + (place.width - text_width(emision, 9, BOLD)) / 2, place.y0 + 32, emision, 9, BOLD)
    b.y = left.y1 + 18

    b.heading("Datos de Identificación del Contribuyente:")
    b.kv_rows(
        [
            ("RFC:", p["rfc"]),
            ("CURP:", p["curp"]),
            ("Nombre (s):", p["nombre"]),
            ("Primer Apellido:", p["paterno"]),
            ("Segundo Apellido:", p["materno"]),
            ("Fecha inicio de operaciones:", "01 DE MARZO DE 2015"),
            ("Estatus en el padrón:", "ACTIVO"),
            ("Fecha de último cambio de estado:", "01 DE MARZO DE 2015"),
            ("Nombre Comercial:", "CONSULTORÍA DEMO"),
        ]
    )
    b.space(14)
    b.heading("Datos del domicilio registrado")
    b.kv_grid(
        [
            ("Código Postal:", p["cp"]), ("Tipo de Vialidad:", p["tipo_vialidad"]),
            ("Nombre de Vialidad:", p["calle"]), ("Número Exterior:", p["ext"]),
            ("Número Interior:", p["int"]), ("Nombre de la Colonia:", p["colonia"]),
            ("Nombre de la Localidad:", p["municipio"]), ("Nombre del Municipio:", p["municipio"]),
            ("Nombre de la Entidad Federativa:", p["estado"]), ("Entre Calle:", "CALLE UNO"),
        ]
    )  # fmt: skip

    b.new_page()
    b.dark_band()
    b.kv_grid([("Y Calle:", "CALLE DOS")], cols=1)
    b.space(14)
    b.heading("Actividades Económicas:")
    b.table(
        ["Orden", "Actividad Económica", "Porcentaje", "Fecha Inicio", "Fecha Fin"],
        [
            ["1", "Servicios de consultoría en computación", "60", "01/03/2015", ""],
            ["2", "Otros servicios profesionales, científicos y técnicos", "40", "01/03/2015", ""],
        ],
        [0.08, 0.52, 0.12, 0.14, 0.14],
        ["center", "left", "center", "center", "center"],
    )
    b.space(14)
    b.heading("Regímenes:")
    b.table(
        ["Régimen", "Fecha Inicio", "Fecha Fin"],
        [
            ["Régimen de Sueldos y Salarios e Ingresos Asimilados a Salarios", "01/03/2015", ""],
            ["Régimen Simplificado de Confianza", "01/01/2022", ""],
        ],
        [0.66, 0.17, 0.17],
        ["left", "center", "center"],
    )
    b.space(14)
    b.heading("Obligaciones:")
    b.table(
        ["Descripción de la Obligación", "Descripción Vencimiento", "Fecha Inicio", "Fecha Fin"],
        [
            ["Declaración anual de ISR. Personas Físicas.",
             "A más tardar el 30 de abril del ejercicio siguiente.", "01/03/2015", ""],
            ["Pago definitivo mensual de IVA.",
             "A más tardar el día 17 del mes inmediato posterior al periodo que corresponda.",
             "01/01/2022", ""],
            ["Pago provisional mensual de ISR. Régimen Simplificado de Confianza.",
             "A más tardar el día 17 del mes inmediato posterior al periodo que corresponda.",
             "01/01/2022", ""],
        ],
        [0.38, 0.38, 0.12, 0.12],
        ["left", "left", "center", "center"],
    )  # fmt: skip
    b.space(14)
    b.paragraph(
        "Este documento es una muestra generada para pruebas de software. No tiene validez "
        "oficial y todos los datos que contiene son ficticios.",
        font=BOLD,
    )
    b.kv_grid(
        [
            ("Cadena Original Sello:", "(omitida en documento de prueba)"),
            ("Sello Digital:", "(omitido en documento de prueba)"),
        ],
        cols=1,
    )

    b.new_page()
    b.dark_band()
    b.heading("Información adicional")
    b.paragraph(
        "Para cualquier aclaración sobre esta constancia de prueba consulte la documentación "
        "del proyecto Document Analyze."
    )

    sample = Sample(
        name="01_csf_constancia_situacion_fiscal",
        category="DOCUMENTO_FISCAL",
        doc_type="CONSTANCIA_SITUACION_FISCAL",
        description="Constancia de Situación Fiscal (SAT), 3 páginas",
        fields={
            "DatosEmision.Lugar": "GUADALAJARA, JALISCO",
            "DatosEmision.Fecha": "15/10/2026",
            "DatosIdentificacion.RFC": p["rfc"],
            "DatosIdentificacion.CURP": p["curp"],
            "DatosIdentificacion.Nombre": p["nombre"],
            "DatosIdentificacion.ApellidoPaterno": p["paterno"],
            "DatosIdentificacion.ApellidoMaterno": p["materno"],
            "DatosIdentificacion.FechaInicioOperaciones": "01/03/2015",
            "DatosIdentificacion.EstatusPadron": "ACTIVO",
            "DatosDomicilio.CodigoPostal": p["cp"],
            "DatosDomicilio.NumeroExterior": p["ext"],
            "DatosDomicilio.Colonia": p["colonia"],
            "DatosDomicilio.Estado": p["estado"],
            "ActividadEconomicaMayorPorcentaje.Porcentaje": 60,
        },
        contains={"DatosDomicilio.Calle": "PRUEBAS", "PrimerRegimen.Regimen": "SUELDOS"},
        counts={"ActividadesEconomicas": 2, "Regimenes": 2},
    )
    return b.finish(page_labels=True), sample


def build_ine() -> tuple[pymupdf.Document, Sample]:
    """INE front and back, enlarged on a letter page (like a KYC photocopy)."""
    p = PERSONA_B
    b = PdfBuilder()
    b.lines(b.m, [("COPIA DE IDENTIFICACIÓN OFICIAL (ANVERSO Y REVERSO)", 9, BOLD)], color=GRAY_TEXT)

    front = pymupdf.Rect(63, 80, 549, 386)
    b.page.draw_rect(front, color=LINE, fill=(0.97, 0.97, 0.98), width=1, radius=0.05)
    x = front.x0 + 18
    b.write(x, front.y0 + 26, "MÉXICO", 12, BOLD)
    b.write(x + 70, front.y0 + 22, "INSTITUTO NACIONAL ELECTORAL", 11, BOLD)
    b.write(x + 70, front.y0 + 36, "CREDENCIAL PARA VOTAR", 9.5, FONT)
    b.placeholder(pymupdf.Rect(x, front.y0 + 56, x + 120, front.y0 + 216), "FOTOGRAFÍA (omitida)")

    cx = x + 140
    y = front.y0 + 62
    entries: list[tuple[str, str, float]] = [
        ("NOMBRE", "", 7),
        ("", p["paterno"], 10), ("", p["materno"], 10), ("", p["nombre"], 10),
        ("DOMICILIO", "", 7),
        ("", "C CALLE FICTICIA 123", 9), ("", "COL CENTRO 06000", 9), ("", "CUAUHTÉMOC, CDMX.", 9),
    ]  # fmt: skip
    for label, value, size in entries:
        y += size * 1.45
        if label:
            b.write(cx, y, label, size, FONT, GRAY_TEXT)
        else:
            b.write(cx, y, value, size, BOLD)
    b.write(front.x1 - 70, front.y0 + 80, "SEXO", 7, FONT, GRAY_TEXT)
    b.write(front.x1 - 70, front.y0 + 93, "H", 10, BOLD)

    rows = [
        [("CLAVE DE ELECTOR", p["clave_elector"])],
        [("CURP", p["curp"]), ("AÑO DE REGISTRO", "2003 01")],
        [("FECHA DE NACIMIENTO", "22/07/1985"), ("SECCIÓN", "1234")],
        [("EMISIÓN", "2020"), ("VIGENCIA", "2020 - 2030")],
    ]
    y = front.y0 + 232
    for row in rows:
        col_x = x
        for label, value in row:
            b.write(col_x, y, label, 7, FONT, GRAY_TEXT)
            b.write(col_x + text_width(label, 7) + 6, y, value, 9, BOLD)
            col_x += 250
        y += 17

    back = pymupdf.Rect(63, 410, 549, 716)
    b.page.draw_rect(back, color=LINE, fill=(0.97, 0.97, 0.98), width=1, radius=0.05)
    b.placeholder(pymupdf.Rect(back.x0 + 18, back.y0 + 20, back.x0 + 238, back.y0 + 80), "FIRMA (omitida)")
    b.write(back.x0 + 260, back.y0 + 34, "INSTITUTO NACIONAL ELECTORAL", 8, BOLD)
    b.write(back.x0 + 260, back.y0 + 47, "Credencial de prueba sin validez oficial.", 7.5, FONT, GRAY_TEXT)
    b.write(back.x0 + 260, back.y0 + 60, "Código QR y huella omitidos.", 7.5, FONT, GRAY_TEXT)
    mrz = [
        "IDMEX0000123456<<1234000000000",
        "8507228H3012315MEX<02<<<<<<<<<5",
        "MUESTRA<PRUEBA<<JOSE<LUIS<<<<<<",
    ]
    for n, line in enumerate(mrz):
        b.write(back.x0 + 40, back.y0 + 200 + n * 22, line, 14, MONO)

    sample = Sample(
        name="02_ine_credencial",
        category="IDENTIFICACION",
        doc_type="INE",
        description="Credencial para votar (anverso y reverso), 1 página",
        fields={
            "Nombre": p["nombre"],
            "ApellidoPaterno": p["paterno"],
            "ApellidoMaterno": p["materno"],
            "CURP": p["curp"],
            "ClaveElector": p["clave_elector"],
            "FechaNacimiento": "22/07/1985",
            "Seccion": "1234",
            "Sexo": "H",
            "Domicilio.CodigoPostal": "06000",
        },
        contains={"Vigencia": "2030"},
    )
    return b.finish(), sample


def build_ine_scan(ine_pdf: Path) -> tuple[pymupdf.Document, Sample]:
    """Image-only copy of the INE sample (no text layer) → forces the Vision OCR path."""
    with pymupdf.open(ine_pdf) as src:
        pix = src[0].get_pixmap(dpi=200, colorspace=pymupdf.csGRAY)
    doc = pymupdf.open()
    page = doc.new_page(width=LETTER[0], height=LETTER[1])
    page.insert_image(page.rect, pixmap=pix)  # watermark is baked into the image
    _, ine = build_ine()
    sample = Sample(
        name="03_ine_escaneada",
        category=ine.category,
        doc_type=ine.doc_type,
        description="La misma INE como imagen escaneada (sin capa de texto) → Vision OCR",
        fields={k: v for k, v in ine.fields.items() if k in {"CURP", "ClaveElector", "ApellidoPaterno", "Seccion"}},
    )
    return doc, sample


def build_passport() -> tuple[pymupdf.Document, Sample]:
    p = PERSONA_C
    b = PdfBuilder()
    card = pymupdf.Rect(56, 70, 556, 500)
    b.page.draw_rect(card, color=LINE, fill=(0.97, 0.98, 0.97), width=1, radius=0.03)
    b.write(card.x0 + 20, card.y0 + 26, "ESTADOS UNIDOS MEXICANOS", 11, BOLD)
    b.write(card.x0 + 20, card.y0 + 39, "UNITED MEXICAN STATES", 8, FONT, GRAY_TEXT)
    b.write(card.x1 - 150, card.y0 + 26, "PASAPORTE", 11, BOLD)
    b.write(card.x1 - 150, card.y0 + 39, "PASSPORT", 8, FONT, GRAY_TEXT)
    b.placeholder(pymupdf.Rect(card.x0 + 20, card.y0 + 60, card.x0 + 140, card.y0 + 220), "FOTOGRAFÍA (omitida)")

    fx = card.x0 + 160
    fields = [
        [("Tipo / Type", "P"), ("Clave del país / Code of issuing state", "MEX")],
        [("Pasaporte No. / Passport No.", "X00012345")],
        [("Apellidos / Surname", f"{p['paterno']} {p['materno']}")],
        [("Nombres / Given names", p["nombre"])],
        [("Nacionalidad / Nationality", "MEXICANA"), ("Fecha de nacimiento / Date of birth", "05/11/1992")],
        [("CURP", p["curp"]), ("Sexo / Sex", "F")],
        [("Lugar de nacimiento / Place of birth", "JALISCO")],
        [("Fecha de expedición / Date of issue", "10/01/2024"),
         ("Fecha de caducidad / Date of expiry", "10/01/2030")],
        [("Autoridad / Authority", "SRE GUADALAJARA")],
    ]  # fmt: skip
    y = card.y0 + 66
    for row in fields:
        col_w = (card.x1 - fx - 10) / len(row)
        for k, (label, value) in enumerate(row):
            b.write(fx + k * col_w, y, label, 6.5, FONT, GRAY_TEXT)
            b.write(fx + k * col_w, y + 12, value, 9.5, BOLD)
        y += 32

    mrz = [
        "P<MEXPRUEBA<CASOS<<SOFIA<ISABEL<<<<<<<<<<<<<",
        "X000123450MEX9211052F3001105<<<<<<<<<<<<<<04",
    ]
    for n, line in enumerate(mrz):
        b.write(card.x0 + 20, card.y1 - 40 + n * 18, line, 12.5, MONO)

    b.y = card.y1 + 24
    b.paragraph("Página de datos de un pasaporte de prueba. Sin elementos de seguridad, sin validez oficial.",
                size=8, color=GRAY_TEXT)  # fmt: skip

    sample = Sample(
        name="04_pasaporte",
        category="IDENTIFICACION",
        doc_type="PASAPORTE",
        description="Pasaporte mexicano (página de datos), 1 página",
        fields={
            "Tipo": "P",
            "ClavePais": "MEX",
            "NumeroPasaporte": "X00012345",
            "Apellidos": f"{p['paterno']} {p['materno']}",
            "Nombres": p["nombre"],
            "CURP": p["curp"],
            "FechaNacimiento": "05/11/1992",
            "FechaExpedicion": "10/01/2024",
            "FechaCaducidad": "10/01/2030",
        },
    )
    return b.finish(), sample


@dataclass
class Bill:
    name: str
    doc_type: str
    issuer: str
    issuer_sub: str
    holder: str
    address: list[str]
    service_label: str
    service_number: str
    info: list[tuple[str, str]]
    charges: list[tuple[str, int]]
    due_label: str
    due_value: str
    expected_due: str
    cp: str
    issuer_contains: str
    usage_table: tuple[list[str], list[list[str]]] | None = None
    note: str = ""


def build_bill(bill: Bill) -> tuple[pymupdf.Document, Sample]:
    b = PdfBuilder()
    total = sum(c for _, c in bill.charges)

    top = b.y
    b.write(b.m, top + 18, bill.issuer, 13, BOLD)
    b.write(b.m, top + 32, bill.issuer_sub, 8.5, FONT, GRAY_TEXT)
    box = pymupdf.Rect(b.w - b.m - 170, top, b.w - b.m, top + 58)
    b.page.draw_rect(box, color=DARK, fill=GRAY_FILL, width=0.8)
    b.write(box.x0 + 10, box.y0 + 18, "TOTAL A PAGAR:", 9, BOLD)
    b.write(box.x0 + 10, box.y0 + 42, money(total), 18, BOLD)
    b.y = top + 72

    b.lines(b.m, [(bill.holder, 10, BOLD)] + [(line, 9, FONT) for line in bill.address])
    b.space(6)
    b.kv_grid([(f"{bill.service_label}:", bill.service_number), (f"{bill.due_label}:", bill.due_value)]
              + bill.info, cols=2)  # fmt: skip
    if bill.usage_table:
        b.space(14)
        b.heading("Consumo")
        headers, rows = bill.usage_table
        b.table(headers, rows, [1 / len(headers)] * len(headers), ["center"] * len(headers))
    b.space(14)
    b.heading("Detalle de cargos")
    b.table(
        ["Concepto", "Importe"],
        [[concept, money(c)] for concept, c in bill.charges] + [["TOTAL A PAGAR", money(total)]],
        [0.75, 0.25],
        ["left", "right"],
    )
    if bill.note:
        b.space(14)
        b.paragraph(bill.note, size=8, color=GRAY_TEXT)

    sample = Sample(
        name=bill.name,
        category="COMPROBANTE_DOMICILIO",
        doc_type=bill.doc_type,
        description=f"Recibo {bill.issuer.title()}, 1 página",
        fields={
            "Titular": bill.holder,
            "NumeroServicio": bill.service_number,
            "Domicilio.CodigoPostal": bill.cp,
            "FechaLimitePago": bill.expected_due,
        },
        contains={"Emisor": bill.issuer_contains, "TotalAPagar": f"{total / 100:,.2f}"},
    )
    return b.finish(), sample


CFE_BILL = Bill(
    name="05_recibo_cfe",
    doc_type="RECIBO_CFE",
    issuer="CFE SUMINISTRADOR DE SERVICIOS BÁSICOS",
    issuer_sub="Comisión Federal de Electricidad - Recibo de energía eléctrica (muestra)",
    holder=full_name(PERSONA_A),
    address=["AV DE LAS PRUEBAS 742 INT 3B", "COL AMERICANA C.P. 44160", "GUADALAJARA, JAL."],
    service_label="NO. DE SERVICIO",
    service_number="123456789012",
    info=[("TARIFA:", "1C"), ("NO. MEDIDOR:", "0000AB12"),
          ("PERIODO FACTURADO:", "31 JUL 26 - 30 SEP 26"), ("FECHA DE EMISIÓN:", "05 OCT 26")],  # fmt: skip
    charges=[("Energía (245 kWh)", 43000), ("IVA 16%", 6880), ("Derecho de Alumbrado Público", 1320)],
    due_label="LÍMITE DE PAGO",
    due_value="20 OCT 26",
    expected_due="20/10/2026",
    cp="44160",
    issuer_contains="CFE",
    usage_table=(["Lectura actual", "Lectura anterior", "Consumo kWh"], [["12,345", "12,100", "245"]]),
    note="Recibo de prueba generado para pruebas de software. Sin validez fiscal.",
)

WATER_BILL = Bill(
    name="06_recibo_agua",
    doc_type="RECIBO_AGUA",
    issuer="SIAPA",
    issuer_sub="Sistema Intermunicipal de los Servicios de Agua Potable y Alcantarillado (muestra)",
    holder=full_name(PERSONA_B),
    address=["CALLE DE LOS EJEMPLOS 55", "COL. CIUDAD GRANJA, C.P. 45010", "ZAPOPAN, JALISCO"],
    service_label="NO. DE CUENTA",
    service_number="9876543210",
    info=[("PERIODO:", "SEPTIEMBRE - OCTUBRE 2026"), ("FECHA DE EMISIÓN:", "01/10/2026"),
          ("TIPO DE SERVICIO:", "DOMÉSTICO"), ("CONSUMO:", "18 m3")],  # fmt: skip
    charges=[("Servicio de agua potable", 25000), ("Drenaje y alcantarillado", 6250),
             ("Saneamiento", 3500), ("Infraestructura", 4200)],  # fmt: skip
    due_label="FECHA LÍMITE DE PAGO",
    due_value="25/10/2026",
    expected_due="25/10/2026",
    cp="45010",
    issuer_contains="SIAPA",
    usage_table=(["Lectura actual (m3)", "Lectura anterior (m3)", "Consumo (m3)"], [["1,240", "1,222", "18"]]),
    note="Servicio de agua potable, drenaje y alcantarillado. Documento de prueba sin validez.",
)

PHONE_BILL = Bill(
    name="07_recibo_telefono",
    doc_type="RECIBO_TELEFONO",
    issuer="TELEFONÍA DEMO, S.A. DE C.V.",
    issuer_sub="Factura de servicios de teléfono e internet (muestra)",
    holder=full_name(PERSONA_C),
    address=["CALLE DEMO 1010", "COL. PROVIDENCIA, C.P. 44630", "GUADALAJARA, JAL."],
    service_label="TELÉFONO",
    service_number="33 1234 5678",
    info=[("PERIODO DE FACTURACIÓN:", "01/09/2026 al 30/09/2026"), ("FECHA DE EMISIÓN:", "01/10/2026"),
          ("PAQUETE:", "INTERNET 100 MBPS + LÍNEA"), ("RENTA MENSUAL:", "$516.38")],  # fmt: skip
    charges=[("Renta mensual paquete Internet 100 Mbps + línea telefónica", 51638), ("IVA 16%", 8262)],
    due_label="PAGAR ANTES DE",
    due_value="18/10/2026",
    expected_due="18/10/2026",
    cp="44630",
    issuer_contains="TELEFONIA DEMO",
    note="Factura de prueba. Empresa y datos ficticios.",
)


def movements(seed: int, count: int, opening: int) -> tuple[list[list[str]], int, int, int]:
    """Deterministic bank movements. Returns rows, total deposits, total withdrawals, closing."""
    rng = random.Random(seed)
    deposits_desc = ["DEPÓSITO SPEI RECIBIDO", "PAGO DE NÓMINA", "DEPÓSITO EN EFECTIVO", "TRANSFERENCIA RECIBIDA"]
    withdraw_desc = ["COMPRA EN COMERCIO DEMO", "RETIRO CAJERO AUTOMÁTICO", "TRANSFERENCIA SPEI ENVIADA",
                     "PAGO DE SERVICIOS", "PAGO TARJETA DE CRÉDITO", "DOMICILIACIÓN SEGURO DEMO"]  # fmt: skip
    balance, deposits, withdrawals = opening, 0, 0
    rows: list[list[str]] = []
    for i in range(count):
        day = 1 + i * 29 // max(count, 1)
        ref = f"{rng.randint(10_000_000, 99_999_999)}"
        if rng.random() < 0.35:
            amount = rng.randint(50_000, 900_000)
            balance += amount
            deposits += amount
            rows.append([f"{day:02d}/09/2026", rng.choice(deposits_desc), ref, "", money(amount), money(balance)])
        else:
            amount = min(rng.randint(5_000, 250_000), max(balance - 100_000, 1_000))
            balance -= amount
            withdrawals += amount
            rows.append([f"{day:02d}/09/2026", rng.choice(withdraw_desc), ref, money(amount), "", money(balance)])
    return rows, deposits, withdrawals, balance


def build_statement(name: str, person: dict[str, str], account: str, clabe: str, address: list[str],
                    n_movements: int, seed: int, opening: int, extraction_path: str
                    ) -> tuple[pymupdf.Document, Sample]:  # fmt: skip
    rows, deposits, withdrawals, closing = movements(seed, n_movements, opening)
    b = PdfBuilder()
    holder = full_name(person)

    def page_header() -> None:
        b.write(b.m, b.y + 14, "BANCO DEMO, S.A., INSTITUCIÓN DE BANCA MÚLTIPLE", 10, BOLD)
        b.write(b.w - b.m - text_width(f"Cuenta {account}", 8), b.y + 14, f"Cuenta {account}", 8, FONT, GRAY_TEXT)
        b.page.draw_line((b.m, b.y + 22), (b.w - b.m, b.y + 22), color=LINE, width=0.6)
        b.y += 32

    b.on_new_page = page_header
    page_header()

    b.lines(b.m, [("ESTADO DE CUENTA", 15, BOLD), ("CUENTA DE DÉBITO DEMO", 9, FONT)])
    b.lines(b.m, [(holder, 10, BOLD), (f"RFC: {person['rfc']}", 9, FONT)] + [(a, 9, FONT) for a in address])
    b.space(4)
    b.kv_grid(
        [
            ("No. de Cuenta:", account), ("CLABE:", clabe),
            ("Periodo:", "DEL 01/09/2026 AL 30/09/2026"), ("Fecha de corte:", "30/09/2026"),
        ]
    )  # fmt: skip
    b.space(14)
    b.heading("Resumen del periodo")
    b.kv_rows(
        [
            ("Saldo inicial:", money(opening)),
            ("(+) Total depósitos / abonos:", money(deposits)),
            ("(-) Total retiros / cargos:", money(withdrawals)),
            ("Saldo final:", money(closing)),
        ],
        label_ratio=0.6,
    )
    b.space(14)
    b.heading("Detalle de movimientos")
    b.table(
        ["Fecha", "Descripción", "Referencia", "Cargos", "Abonos", "Saldo"],
        rows,
        [0.12, 0.34, 0.14, 0.13, 0.13, 0.14],
        ["center", "left", "center", "right", "right", "right"],
    )
    b.space(10)
    b.paragraph("Estado de cuenta de prueba. Banco, titular, cuenta y movimientos son ficticios.",
                size=8, color=GRAY_TEXT)  # fmt: skip

    sample = Sample(
        name=name,
        category="DOCUMENTO_BANCARIO",
        doc_type="ESTADO_CUENTA",
        extraction_path=extraction_path,
        description=f"Estado de cuenta con {n_movements} movimientos",
        fields={
            "Titular": holder,
            "RFC": person["rfc"],
            "NumeroCuenta": account,
            "CLABE": clabe,
            "Periodo.FechaInicio": "01/09/2026",
            "Periodo.FechaFin": "30/09/2026",
        },
        contains={
            "Banco": "BANCO DEMO",
            "SaldoInicial": f"{opening / 100:,.2f}",
            "SaldoFinal": f"{closing / 100:,.2f}",
            "TotalDepositos": f"{deposits / 100:,.2f}",
            "TotalRetiros": f"{withdrawals / 100:,.2f}",
        },
    )
    return b.finish(), sample


def build_letter() -> tuple[pymupdf.Document, Sample]:
    """A document outside the catalog → expected OTRO / DESCONOCIDO, no extraction."""
    b = PdfBuilder(margin=60)
    b.lines(b.m, [("EMPRESA DEMO, S.A. DE C.V.", 12, BOLD), ("Recursos Humanos", 9, FONT)])
    b.space(10)
    b.paragraph("Guadalajara, Jalisco, a 20 de octubre de 2026.", size=10)
    b.space(6)
    b.paragraph("A QUIEN CORRESPONDA:", size=10, font=BOLD)
    b.space(4)
    for text in [
        "Por medio de la presente hacemos constar que el C. CARLOS ALBERTO DEMO PRUEBA colaboró en "
        "esta empresa ficticia del 1 de febrero de 2019 al 30 de septiembre de 2026, desempeñando el "
        "puesto de Coordinador de Proyectos.",
        "Durante su estancia mostró responsabilidad, puntualidad y una excelente disposición para el "
        "trabajo en equipo, por lo que no tenemos inconveniente en recomendarlo ampliamente.",
        "Se extiende la presente carta de recomendación a petición del interesado y para los fines "
        "que a él convengan.",
    ]:
        b.paragraph(text, size=10)
        b.space(4)
    b.space(30)
    b.paragraph("ATENTAMENTE", size=10, font=BOLD)
    b.placeholder(pymupdf.Rect(b.m, b.y + 6, b.m + 180, b.y + 50), "FIRMA (omitida)")
    b.y += 60
    b.paragraph("LIC. ANA DEMO EJEMPLO - Gerente de Recursos Humanos", size=9)

    sample = Sample(
        name="10_otro_carta",
        category="OTRO",
        doc_type="DESCONOCIDO",
        extraction_path="skip",
        description="Carta de recomendación (fuera del catálogo) → OTRO / DESCONOCIDO",
    )
    return b.finish(), sample


# =============================================================================== main


def write_expected(sample: Sample, pdf_path: Path) -> None:
    with pymupdf.open(pdf_path) as doc:
        pages = doc.page_count
    payload = {
        "file": pdf_path.name,
        "description": sample.description,
        "documentCategory": sample.category,
        "documentType": sample.doc_type,
        "extractionPath": sample.extraction_path,
        "pages": pages,
        "fields": sample.fields,
        "contains": sample.contains,
        "counts": sample.counts,
    }
    (EXPECTED_DIR / f"{sample.name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    EXPECTED_DIR.mkdir(exist_ok=True)

    builders: list[Callable[[], tuple[pymupdf.Document, Sample]]] = [
        build_csf,
        build_ine,
        build_passport,
        lambda: build_bill(CFE_BILL),
        lambda: build_bill(WATER_BILL),
        lambda: build_bill(PHONE_BILL),
        lambda: build_statement(
            "08_estado_cuenta", {**PERSONA_A}, "0123456789", "012345678901234567",
            ["AV DE LAS PRUEBAS 742 INT 3B", "COL AMERICANA, C.P. 44160", "GUADALAJARA, JALISCO"],
            n_movements=45, seed=8, opening=2_500_000, extraction_path="full",
        ),
        lambda: build_statement(
            "09_estado_cuenta_largo", {**PERSONA_B}, "9876501234", "012180098765012345",
            ["CALLE DE LOS EJEMPLOS 55", "COL. CIUDAD GRANJA, C.P. 45010", "ZAPOPAN, JALISCO"],
            n_movements=LONG_STATEMENT_MOVEMENTS, seed=9, opening=15_000_000, extraction_path="retrieval",
        ),
        build_letter,
    ]  # fmt: skip

    for build in builders:
        doc, sample = build()
        path = save(doc, sample.name)
        write_expected(sample, path)
        print(f"  {path.relative_to(ROOT)}")
        if sample.name == "02_ine_credencial":
            scan_doc, scan_sample = build_ine_scan(path)
            scan_path = save(scan_doc, scan_sample.name)
            write_expected(scan_sample, scan_path)
            print(f"  {scan_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
