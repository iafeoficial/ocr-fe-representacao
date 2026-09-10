from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .auth import require_ocr_secret
from .catalog import fetch_codigos_for_industria
from .config import get_settings
from .match import match_candidatos
from .ocr import extract_price, run_ocr
from .preprocess import preprocess_for_ocr

logger = logging.getLogger("pesquisa_ocr")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Pesquisa OCR", version="1.0.0")

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.parsed_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*", "X-OCR-Secret"],
)

TipoPesquisa = Literal["interna", "externa"]


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ocr/pesquisa", dependencies=[Depends(require_ocr_secret)])
async def ocr_pesquisa(
    produto_crop: UploadFile = File(...),
    preco_crop: UploadFile = File(...),
    tipo: str = Form(...),
    industria: str = Form(...),
) -> JSONResponse:
    tipo_norm = (tipo or "").strip().lower()
    if tipo_norm not in ("interna", "externa"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="tipo deve ser 'interna' ou 'externa'",
        )

    produto_bytes = await produto_crop.read()
    preco_bytes = await preco_crop.read()
    if not produto_bytes or not preco_bytes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="produto_crop e preco_crop são obrigatórios",
        )

    try:
        produto_img = preprocess_for_ocr(produto_bytes)
        preco_img = preprocess_for_ocr(preco_bytes)
        texto_ocr = run_ocr(produto_img, psm=6)
        preco_ocr_raw = run_ocr(preco_img, psm=7)
        preco = extract_price(preco_ocr_raw) or extract_price(texto_ocr)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001 — surface OCR runtime errors
        logger.exception("Falha no OCR")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha no OCR: {exc}",
        ) from exc

    candidatos: list[dict[str, Any]] = []
    if tipo_norm == "interna":
        settings = get_settings()
        try:
            catalogo = await fetch_codigos_for_industria(settings, industria)
            candidatos = match_candidatos(texto_ocr, catalogo, top_n=3)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Falha ao buscar codigos")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Falha ao consultar catálogo: {exc}",
            ) from exc

    sugerido = candidatos[0] if candidatos else None
    descricao = (
        sugerido["produto"]
        if sugerido
        else texto_ocr
    )

    body = {
        "tipo": tipo_norm,
        "industria": industria.strip(),
        "texto_ocr": texto_ocr,
        "preco_ocr_raw": preco_ocr_raw,
        "preco": preco,
        "descricao": descricao,
        "sugerido": sugerido,
        "candidatos": candidatos,
    }
    return JSONResponse(body)
