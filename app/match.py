"""Fuzzy match OCR text → catálogo `codigos` (rapidfuzz)."""

from __future__ import annotations

from typing import Any

from rapidfuzz import fuzz, process

# Include in candidate list (user can still pick manually).
MIN_CANDIDATE_SCORE = 50.0
# Only auto-fill descricao / sugerido at or above this.
AUTO_SUGGEST_MIN_SCORE = 75.0


def match_candidatos(
    texto_ocr: str,
    catalogo: list[dict[str, Any]],
    *,
    top_n: int = 3,
    min_score: float = MIN_CANDIDATE_SCORE,
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


def pick_sugerido(
    candidatos: list[dict[str, Any]],
    *,
    min_score: float = AUTO_SUGGEST_MIN_SCORE,
) -> dict[str, Any] | None:
    """Top candidate only if score is strong enough to auto-apply."""
    if not candidatos:
        return None
    top = candidatos[0]
    score = top.get("score")
    if not isinstance(score, (int, float)) or float(score) < min_score:
        return None
    return top
