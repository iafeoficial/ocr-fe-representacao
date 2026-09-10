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


def pick_unit_price(text: str, *, prefer: str = "first") -> str | None:
    """Escolhe preço de unidade; ignora emb. packs altos na mesma metade."""
    prices = extract_all_prices(text)
    if not prices:
        return None

    def to_float(p: str) -> float:
        return float(p.replace(".", "").replace(",", "."))

    unitish = [p for p in prices if 0.5 <= to_float(p) < 80.0]
    pool = unitish or prices
    if prefer == "min":
        return min(pool, key=to_float)
    return pool[0]


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
    """Dois preços: maior=varejo, menor=atacado. Um preço: só atacado (Mateus L)."""
    if not prices:
        return None, None
    if len(prices) == 1:
        # Etiqueta Mateus: dígito único lido costuma ser ATACADO (coluna esquerda).
        return None, prices[0]

    def to_float(p: str) -> float:
        return float(p.replace(".", "").replace(",", "."))

    ordered = sorted(prices, key=to_float, reverse=True)
    varejo = ordered[0]
    atacado = ordered[1]
    if to_float(varejo) == to_float(atacado):
        return None, varejo
    return varejo, atacado


_PRODUCT_NOUNS = (
    "ERVILHA",
    "MILHO",
    "MOLHO",
    "EXTRATO",
    "CATCHUP",
    "KETCHUP",
    "MAIONESE",
    "MOSTARDA",
    "ATUM",
    "SARDINHA",
    "SELETA",
    "AZEITONA",
    "PALMITO",
    "GOIABADA",
    "DOCE",
    "GELEIA",
    "BISCOITO",
    "COOKIE",
    "BALA",
    "GOMAS",
    "REFRIGERANTE",
    "SUCO",
    "NECTAR",
)


def _fold_upper(text: str) -> str:
    upper = (text or "").upper()
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
    return re.sub(r"\s+", " ", upper).strip()


def _strip_noise_tokens(phrase: str) -> str:
    keep: list[str] = []
    for tok in phrase.split():
        if tok in _PRODUCT_NOUNS:
            keep.append(tok)
            continue
        if WEIGHT_RE.fullmatch(tok) or re.fullmatch(r"\d{2,4}G?", tok):
            keep.append(tok if tok.endswith("G") else f"{tok}G" if tok.isdigit() else tok)
            continue
        if len(tok) <= 2 and tok not in {"SH", "KG", "UN", "ML"}:
            continue
        if re.fullmatch(r"[A-Z]{1,3}", tok) and tok not in {"SH", "KG", "UN", "ML", "UND"}:
            # Drop short OCR garbage (YET, AAA, II, AG…) unless known brand piece.
            continue
        keep.append(tok)
    return " ".join(keep).strip(" -/")


def clean_product_name(text: str, *, brand_hint: str | None = None) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    upper = _fold_upper(raw)
    hint = _fold_upper(brand_hint or "")
    brands = [
        "PREDILECTA",
        "STELLA",
        "HARIBO",
        "NESTLE",
        "BAUDI",
        "HEINZ",
        "QUERO",
        "ELEFANTE",
    ]
    if hint and len(hint) >= 4 and hint not in brands:
        brands.insert(0, hint)

    scored: list[tuple[int, str]] = []

    # Compact brand-centric: NOUN + BRAND (+ weight)
    for brand in brands:
        idx = upper.find(brand)
        if idx < 0:
            # Fuzzy: brand with 1 OCR typo near end (PREDITECTA…)
            m_fuzzy = re.search(
                rf"\b{re.escape(brand[:6])}[A-Z]{{2,12}}\b", upper
            ) if len(brand) >= 8 else None
            if not m_fuzzy:
                continue
            brand_tok = m_fuzzy.group(0)
            idx = m_fuzzy.start()
        else:
            brand_tok = brand

        before = upper[max(0, idx - 24) : idx].strip()
        after = upper[idx + len(brand_tok) : idx + len(brand_tok) + 16]
        noun = ""
        for n in _PRODUCT_NOUNS:
            if n in before.split() or before.endswith(n):
                noun = n
                break
        weight_m = WEIGHT_RE.search(after) or WEIGHT_RE.search(upper[idx : idx + 40])
        parts = [p for p in (noun, brand) if p]
        if weight_m:
            parts.append(f"{weight_m.group(1)}G")
        phrase = " ".join(parts)
        if not phrase:
            continue
        score = 80 + (30 if noun else 0) + (15 if weight_m else 0)
        scored.append((score, phrase))

    # Shelf-label style line: ERVILHA PREDILECTA SH 170G
    label_m = re.search(
        r"\b((?:ERVILHA|MILHO|MOLHO|EXTRATO|SELETA|ATUM|SARDINHA|GOIABADA|"
        r"MAIONESE|MOSTARDA|CATCHUP|KETCHUP|AZEITONA|PALMITO|DOCE|GELEIA|"
        r"BISCOITO|BALA|GOMAS|SUCO|NECTAR)\s+"
        r"[A-Z]{4,}(?:\s+SH)?(?:\s+\d{2,4}\s*G)?)\b",
        upper,
    )
    if label_m:
        phrase = _strip_noise_tokens(re.sub(r"\s+", " ", label_m.group(1)))
        phrase = re.sub(r"\bSH\b", "", phrase)
        phrase = re.sub(r"\s+", " ", phrase).strip()
        if phrase:
            scored.append((120, phrase))

    for m in PRODUCT_LINE_RE.finditer(upper):
        phrase = _strip_noise_tokens(re.sub(r"\s+", " ", m.group(1)))
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
        # Prefer shorter clean names over long OCR soup.
        score = 40 - max(0, len(phrase) - 28) // 2
        if any(b in phrase for b in brands):
            score += 40
        if any(n in phrase for n in _PRODUCT_NOUNS):
            score += 25
        if WEIGHT_RE.search(phrase):
            score += 10
        scored.append((score, phrase))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]
        best = re.sub(r"\b1700\b", "170G", best)
        best = re.sub(r"\s+R\$?\s*$", "", best).strip(" -/")
        best = _strip_noise_tokens(best)
        return best

    cleaned = _strip_noise_tokens(
        re.sub(r"[^\wÁÉÍÓÚÂÊÔÃÕÇáéíóúâêôãõç\s,.\-/%]", " ", raw)
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    # Sem marca/noun conhecido: lixo OCR (ex. "IEEE HILL") — vazio para fallback.
    folded = _fold_upper(cleaned)
    has_brand = any(b in folded for b in brands)
    has_noun = any(n in folded for n in _PRODUCT_NOUNS)
    if cleaned and not has_brand and not has_noun:
        return ""
    return cleaned[:80]
