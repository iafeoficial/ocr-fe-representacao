"""Fuzzy match OCR text → catálogo `codigos` (rapidfuzz)."""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz, process


def match_candidatos(
    texto_ocr: str,
    catalogo: list[dict[str, Any]],
    *,
    top_n: int = 3,
    min_score: float = 40.0,
) -> list[dict[str, Any]]:
    query = (texto_ocr or "").strip()
    if not query or not catalogo:
        return []

    choices = {i: item["produto"] for i, item in enumerate(catalogo)}
    results = process.extract(
        query,
        choices,
        scorer=fuzz.WRatio,
        limit=top_n,
    )

    out: list[dict[str, Any]] = []
    for _produto, score, idx in results:
        if score < min_score:
            continue
        item = catalogo[idx]
        out.append(
            {
                "codigo": item["codigo"],
                "produto": item["produto"],
                "industria": item["industria"],
                "score": round(float(score), 1),
            }
        )
    return out
