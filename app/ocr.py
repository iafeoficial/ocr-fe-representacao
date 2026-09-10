"""Tesseract OCR + extração de preço."""

from __future__ import annotations

import re

import numpy as np
import pytesseract

PRICE_RE = re.compile(r"(\d{1,3}[.,]\d{2})")


def run_ocr(image: np.ndarray, *, psm: int = 6) -> str:
    config = f"--oem 3 --psm {psm}"
    text = pytesseract.image_to_string(image, lang="por+eng", config=config)
    return " ".join(text.split()).strip()


def extract_price(text: str) -> str | None:
    """Primeiro preço no formato 0,00 / 00.00; normaliza decimal para vírgula BR."""
    match = PRICE_RE.search(text or "")
    if not match:
        return None
    raw = match.group(1)
    if "," in raw:
        return raw
    return raw.replace(".", ",")
