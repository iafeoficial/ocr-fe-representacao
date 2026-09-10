from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .auth import require_ocr_secret
from .config import get_settings
from .ocr import (
    assign_varejo_atacado,
    clean_product_name,
    extract_all_prices,
    run_ocr_best,
)
from .preprocess import prepare_tag_and_full

logger = logging.getLogger("pesquisa_ocr")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Pesquisa OCR", version="1.1.0")

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.parsed_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*", "X-OCR-Secret"],
)

TipoPesquisa = Literal["interna", "externa"]

_DEBUG_LOG = Path(__file__).resolve().parents[2] / "debug-bdee85.log"


def _agent_log(hypothesis_id: str, message: str, data: dict[str, Any]) -> None:
    # #region agent log
    try:
        payload = {
            "sessionId": "bdee85",
            "runId": "ocr-post",
            "hypothesisId": hypothesis_id,
            "location": "pesquisa-ocr/main.py",
            "message": message,
            "data": data,
            "timestamp": int(time.time() * 1000),
        }
        with _DEBUG_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass
    # #endregion


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
    raw = produto_bytes if len(produto_bytes) >= len(preco_bytes) else preco_bytes
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="produto_crop e preco_crop são obrigatórios",
        )

    try:
        full_gray, tag_gray, tag_price_gray = prepare_tag_and_full(raw)

        tag_text = ""
        if tag_gray is not None:
            tag_text = run_ocr_best(tag_gray, psms=(6, 4, 11, 3))

        full_text = run_ocr_best(full_gray, psms=(6, 4, 11))

        price_text = ""
        if tag_price_gray is not None:
            price_text = run_ocr_best(
                tag_price_gray,
                psms=(7, 6, 11),
                whitelist="0123456789R$rs., ",
            )
        if not price_text and tag_gray is not None:
            price_text = run_ocr_best(tag_gray, psms=(7, 6, 11))
        if not price_text:
            price_text = full_text

        source_for_name = tag_text if len(tag_text) >= 8 else full_text
        descricao = clean_product_name(source_for_name) or clean_product_name(full_text)

        prices = (
            extract_all_prices(price_text)
            or extract_all_prices(tag_text)
            or extract_all_prices(full_text)
        )
        preco_varejo, preco_atacado = assign_varejo_atacado(prices)
        preco = preco_varejo

        _agent_log(
            "A-B-C",
            "ocr_pesquisa result",
            {
                "usedYellowTag": tag_gray is not None,
                "tagTextSample": tag_text[:160],
                "fullTextSample": full_text[:120],
                "priceTextSample": price_text[:120],
                "prices": prices,
                "preco_varejo": preco_varejo,
                "preco_atacado": preco_atacado,
                "descricao": (descricao or "")[:120],
            },
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Falha no OCR")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Falha no OCR: {exc}",
        ) from exc

    body = {
        "tipo": tipo_norm,
        "industria": industria.strip(),
        "texto_ocr": descricao or tag_text or full_text,
        "preco_ocr_raw": price_text,
        "preco": preco,
        "preco_varejo": preco_varejo,
        "preco_atacado": preco_atacado,
        "descricao": descricao,
        "sugerido": None,
        "candidatos": [],
    }
    return JSONResponse(body)
