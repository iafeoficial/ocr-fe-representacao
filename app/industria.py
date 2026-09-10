"""Equivalente a toIndustriaPadrao / industriasMatch do front (vendasDomain.ts)."""

from __future__ import annotations

import re
import unicodedata

_SUFFIX_RE = re.compile(
    r"\b(ALIMENTOS|ALIMENTO|LTDA|LTDA\.|S/A|SA|BRASIL|OFICIAL)\b",
    re.IGNORECASE,
)


def normalize_industria_key(nome: str) -> str:
    s = unicodedata.normalize("NFD", nome or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.upper()
    s = re.sub(r"[''`]", "", s)
    s = _SUFFIX_RE.sub("", s)
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_industria_padrao(nome: str) -> str:
    return normalize_industria_key(nome)


def industrias_match(a: str, b: str) -> bool:
    """Match estrito: igualdade, contenção (>=5), ou mesmo primeiro token (>=5)."""
    na = normalize_industria_key(a)
    nb = normalize_industria_key(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    # Contenção só se o menor tiver pelo menos 5 chars (evita "SA" / ruído curto).
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    if len(shorter) >= 5 and shorter in longer:
        return True
    token_a = na.split(" ")[0] if na else ""
    token_b = nb.split(" ")[0] if nb else ""
    return len(token_a) >= 5 and token_a == token_b
