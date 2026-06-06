# =============================================================================
# view/xlsx_writer.py
# Write the Michelin evaluation table to an Excel (.xlsx) file.
#
# Format specified in nuevas_normas.pdf (June 2026).
# One worksheet per image; both Reto 1 and Reto 2 tables on the same sheet.
# =============================================================================

from __future__ import annotations

import os
from typing import Optional

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from model.reto_measurements import Reto1Row, Reto2Row
from config.config import RETO1_X_MM, RETO2_Y_OFFSETS_MM


# ── Colour palette ────────────────────────────────────────────────────────────
_HEADER_FILL   = PatternFill("solid", fgColor="1F4E79")   # dark blue
_RETO1_FILL    = PatternFill("solid", fgColor="BDD7EE")   # light blue
_RETO2_FILL    = PatternFill("solid", fgColor="E2EFDA")   # light green
_LABEL_FILL    = PatternFill("solid", fgColor="D9D9D9")   # grey
_WHITE_FILL    = PatternFill("solid", fgColor="FFFFFF")

_HEADER_FONT   = Font(bold=True, color="FFFFFF", size=11)
_BOLD_FONT     = Font(bold=True, size=10)
_NORMAL_FONT   = Font(size=10)

_THIN_BORDER = Border(
    left=Side(style="thin"),  right=Side(style="thin"),
    top=Side(style="thin"),   bottom=Side(style="thin"),
)
_CENTER = Alignment(horizontal="center", vertical="center")


def _fmt(val: Optional[float], decimals: int = 2) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def _cell(ws, row: int, col: int, value, fill=None, font=None,
          border=True, align=True):
    c = ws.cell(row=row, column=col, value=value)
    if fill:
        c.fill = fill
    if font:
        c.font = font
    if border:
        c.border = _THIN_BORDER
    if align:
        c.alignment = _CENTER
    return c


# ── Public API ────────────────────────────────────────────────────────────────

def write_reto_xlsx(
    path: str,
    results: list[dict],
) -> None:
    """Write one xlsx file with one sheet per processed image.

    ``results`` is a list of dicts, each with keys:
        image_name  str
        reto1_rows  list[Reto1Row]
        reto2_rows  list[Reto2Row]
        qr_ok       bool
        n_edges     int
    """
    wb = openpyxl.Workbook()
    wb.remove(wb.active)  # remove default sheet

    for rec in results:
        name = os.path.splitext(rec["image_name"])[0][:31]  # sheet name ≤31 chars
        ws   = wb.create_sheet(title=name)
        _write_sheet(ws, rec)

    wb.save(path)
    print(f"[XlsxWriter] Saved: {path}")


def _write_sheet(ws, rec: dict) -> None:
    reto1: list[Reto1Row] = rec.get("reto1_rows", [])
    reto2: list[Reto2Row] = rec.get("reto2_rows", [])
    n1 = len(RETO1_X_MM)      # 9
    n2 = len(RETO2_Y_OFFSETS_MM)  # 10

    # ── Column widths ─────────────────────────────────────────────────────────
    ws.column_dimensions["A"].width = 8
    ws.column_dimensions["B"].width = 14
    for i in range(n1 + 1):
        ws.column_dimensions[get_column_letter(3 + i)].width = 10
    for i in range(n2 + 1):
        ws.column_dimensions[get_column_letter(3 + n1 + 2 + i)].width = 10

    row = 1

    # ── Image info ────────────────────────────────────────────────────────────
    _cell(ws, row, 1, "Imagen", fill=_LABEL_FILL, font=_BOLD_FONT)
    _cell(ws, row, 2, rec.get("image_name", ""), font=_NORMAL_FONT)
    _cell(ws, row, 3, "QR OK",   fill=_LABEL_FILL, font=_BOLD_FONT)
    _cell(ws, row, 4, "Sí" if rec.get("qr_ok") else "No", font=_NORMAL_FONT)
    _cell(ws, row, 5, "Cortes", fill=_LABEL_FILL, font=_BOLD_FONT)
    _cell(ws, row, 6, rec.get("n_edges", 0), font=_NORMAL_FONT)
    row += 2

    # ════════════════════════════════════════════════════════════════════════
    # RETO 1 TABLE
    # ════════════════════════════════════════════════════════════════════════
    _cell(ws, row, 1, "RETO 1", fill=_HEADER_FILL, font=_HEADER_FONT)
    _cell(ws, row, 2, "Medida \\ Punto", fill=_HEADER_FILL, font=_HEADER_FONT)
    for j, x in enumerate(RETO1_X_MM):
        _cell(ws, row, 3 + j, f"P{j+1}", fill=_HEADER_FILL, font=_HEADER_FONT)
    row += 1

    # Sub-header: distances
    _cell(ws, row, 1, "", fill=_RETO1_FILL)
    _cell(ws, row, 2, "Distancia a borde (mm)",
          fill=_RETO1_FILL, font=_BOLD_FONT)
    for j, x in enumerate(RETO1_X_MM):
        _cell(ws, row, 3 + j, x, fill=_RETO1_FILL, font=_BOLD_FONT)
    row += 1

    reto1_metrics = [
        ("DA", "Dist. borde mesa → Banda A (mm)"),
        ("DB", "Dist. borde mesa → Banda B (mm)"),
        ("LA", "Anchura Banda A (mm)"),
        ("LB", "Anchura Banda B (mm)"),
    ]
    for attr, label in reto1_metrics:
        _cell(ws, row, 1, attr,  fill=_LABEL_FILL, font=_BOLD_FONT)
        _cell(ws, row, 2, label, fill=_WHITE_FILL,  font=_NORMAL_FONT)
        for j, r1 in enumerate(reto1):
            val = getattr(r1, attr, None)
            _cell(ws, row, 3 + j, _fmt(val), fill=_WHITE_FILL,
                  font=_NORMAL_FONT)
        row += 1

    row += 1  # blank separator

    # ════════════════════════════════════════════════════════════════════════
    # RETO 2 TABLE
    # ════════════════════════════════════════════════════════════════════════
    col0 = 3   # first data column for Reto 2 (same column grid as Reto 1)
    _cell(ws, row, 1, "RETO 2", fill=_HEADER_FILL, font=_HEADER_FONT)
    _cell(ws, row, 2, "Medida \\ Punto", fill=_HEADER_FILL, font=_HEADER_FONT)
    for j in range(n2):
        _cell(ws, row, col0 + j, f"P{j+1}", fill=_HEADER_FILL,
              font=_HEADER_FONT)
    row += 1

    _cell(ws, row, 1, "", fill=_RETO2_FILL)
    _cell(ws, row, 2, "Dist. a borde producto (mm)",
          fill=_RETO2_FILL, font=_BOLD_FONT)
    for j, y in enumerate(RETO2_Y_OFFSETS_MM):
        _cell(ws, row, col0 + j, y, fill=_RETO2_FILL, font=_BOLD_FONT)
    row += 1

    reto2_metrics = [
        ("SA1", "Dist. bordes sup. soldadura Banda A (mm)"),
        ("SA2", "Dist. borde sup.-inf. (entre negros) Banda A (mm)"),
        ("YSA", "Dist. eje Y borde mesa → borde soldadura prox. Banda A (mm)"),
        ("SB1", "Dist. bordes sup. soldadura Banda B (mm)"),
        ("SB2", "Dist. borde sup.-inf. (entre negros) Banda B (mm)"),
        ("YSB", "Dist. eje Y borde mesa → borde soldadura prox. Banda B (mm)"),
    ]
    for attr, label in reto2_metrics:
        _cell(ws, row, 1, attr,  fill=_LABEL_FILL, font=_BOLD_FONT)
        _cell(ws, row, 2, label, fill=_WHITE_FILL,  font=_NORMAL_FONT)
        for j, r2 in enumerate(reto2):
            val = getattr(r2, attr, None)
            _cell(ws, row, col0 + j, _fmt(val), fill=_WHITE_FILL,
                  font=_NORMAL_FONT)
        row += 1
