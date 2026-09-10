"""Auth via header X-OCR-Secret."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from .config import get_settings


async def require_ocr_secret(
    x_ocr_secret: str | None = Header(default=None, alias="X-OCR-Secret"),
) -> None:
    settings = get_settings()
    expected = (settings.ocr_shared_secret or "").strip()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OCR_SHARED_SECRET não configurado",
        )
    provided = (x_ocr_secret or "").strip()
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Secret OCR inválido",
        )
