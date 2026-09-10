"""Tesseract OCR + extração de nome/preços da etiqueta."""

from __future__ import annotations

import re

import numpy as np
import pytesseract

PRICE_RE = re.compile(r"(?:R\$\s*)?(\d{1,3}[.,]\d{2})")
PRODUCT_LINE_RE = re.compile(
    r"\b([A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9][A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9\s/\-]{6,}?(?:\s+\d{2,4}\s*G)?)\b"
)
WEIGHT_RE = re.compile(r"\b(\d{2,4})\s*G\b", re.I)


def run_ocr(image: np.ndarray, *, psm: int = 6, whitelist: str | None = None) -> str:
    config = f"--oem 3 --psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    text = pytesseract.image_to_string(image, lang="por+eng", config=config)
    return " ".join(text.split()).strip()


def run_ocr_best(
    image: np.ndarray,
    *,
    psms: tuple[int, ...] = (6, 4, 11),
    whitelist: str | None = None,
) -> str:
    best = ""
    for psm in psms:
        try:
            text = run_ocr(image, psm=psm, whitelist=whitelist)
        except Exception:  # noqa: BLE001
            continue
        if len(text) > len(best):
            best = text
    return best


def extract_price(text: str) -> str | None:
    prices = extract_all_prices(text)
    return prices[0] if prices else None


def extract_all_prices(text: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for match in PRICE_RE.finditer(text or ""):
        raw = match.group(1)
        norm = raw if "," in raw else raw.replace(".", ",")
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def assign_varejo_atacado(prices: list[str]) -> tuple[str | None, str | None]:
    if not prices:
        return None, None
    if len(prices) == 1:
        return prices[0], None

    def to_float(p: str) -> float:
        return float(p.replace(".", "").replace(",", "."))

    ordered = sorted(prices, key=to_float, reverse=True)
    varejo = ordered[0]
    atacado = ordered[1]
    if to_float(varejo) == to_float(atacado):
        return varejo, None
    return varejo, atacado


def clean_product_name(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    upper = raw.upper()
    for a, b in (
        ("Á", "A"),
        ("É", "E"),
        ("Í", "I"),
        ("Ó", "O"),
        ("Ú", "U"),
        ("Ã", "A"),
        ("Õ", "O"),
        ("Ç", "C"),
    ):
        upper = upper.replace(a, b)
    upper = re.sub(r"[|_[\]{}<>]+", " ", upper)
    upper = re.sub(r"\s+", " ", upper).strip()

    scored: list[tuple[int, str]] = []
    for m in PRODUCT_LINE_RE.finditer(upper):
        phrase = re.sub(r"\s+", " ", m.group(1)).strip(" -/")
        if any(
            junk in phrase
            for junk in (
                "ANALISANDO",
                "PROCESSANDO",
                "SEGURA",
                "CELULAR",
                "CAPTURA",
                "CAMERA",
            )
        ):
            continue
        if len(phrase) < 8:
            continue
        score = len(phrase)
        if any(b in phrase for b in ("PREDILECTA", "STELLA", "HARIBO", "NESTLE", "BAUDI")):
            score += 40
        if WEIGHT_RE.search(phrase):
            score += 20
        if phrase.count(" ") >= 2:
            score += 10
        scored.append((score, phrase))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        best = re.sub(r"\b1700\b", "170G", best)
        return best

    cleaned = re.sub(r"[^\wÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç\s,.\-/%]", " ", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:120]
