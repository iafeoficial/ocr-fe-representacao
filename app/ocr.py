"""Tesseract OCR + extração de preço."""

from __future__ import annotations

import re

import numpy as np
import pytesseract

# Optional R$; first plausible xx,xx / xx.xx (avoids bare integers).
PRICE_RE = re.compile(r"(?:R\$\s*)?(\d{1,3}[.,]\d{2})")


def run_ocr(image: np.ndarray, *, psm: int = 6) -> str:
    config = f"--oem 3 --psm {psm}"
    text = pytesseract.image_to_string(image, lang="por+eng", config=config)
    return " ".join(text.split()).strip()


def run_ocr_best(
    image: np.ndarray,
    *,
    psms: tuple[int, ...] = (6, 4, 11),
) -> str:
    """Try several PSM modes; keep the longest non-empty reading."""
    best = ""
    for psm in psms:
        try:
            text = run_ocr(image, psm=psm)
        except Exception:  # noqa: BLE001 — fall through to next PSM
            continue
        if len(text) > len(best):
            best = text
    return best


def extract_price(text: str) -> str | None:
    """Primeiro preço no formato 0,00 / 00.00; normaliza decimal para vírgula BR."""
    match = PRICE_RE.search(text or "")
    if not match:
        return None
    raw = match.group(1)
    if "," in raw:
        return raw
    return raw.replace(".", ",")
