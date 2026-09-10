"""Auth via header X-OCR-Secret."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from .config import get_settings


def _normalize_secret(value: str | None) -> str:
    """Strip whitespace and accidental surrounding quotes from env / header."""
    s = (value or "").strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s


async def require_ocr_secret(
    x_ocr_secret: str | None = Header(default=None, alias="X-OCR-Secret"),
) -> None:
    settings = get_settings()
    expected = _normalize_secret(settings.ocr_shared_secret)
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR_SHARED_SECRET não configurado",
        )
    provided = _normalize_secret(x_ocr_secret)
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Secret OCR inválido",
        )
