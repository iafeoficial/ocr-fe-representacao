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
    assign_prices_from_labels,
    assign_varejo_atacado,
    clean_product_name,
    is_mateus_dual_column,
    is_unit_emb_layout,
    merge_price_lists,
    pick_unit_price,
    reconcile_spatial_prices,
    run_ocr_best,
)
from .preprocess import prepare_tag_and_full, split_tag_price_halves, to_price_ocr_gray

logger = logging.getLogger("pesquisa_ocr")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Pesquisa OCR", version="1.2.3")
OCR_BUILD = "1.2.3-unit-emb"

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
    return {"status": "ok", "version": app.version, "build": OCR_BUILD}


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

    industria_s = industria.strip()

    try:
        full_gray, tag_gray, tag_price_gray, tag_bgr, tag_kind = prepare_tag_and_full(raw)

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

        # Mateus L/R only when ATACADO+VAREJO labels exist.
        # PRECO POR UNIDADE + EMB tags have centered unit price — L/R bisects "2,79".
        spatial_atacado: str | None = None
        spatial_varejo: str | None = None
        spatial_left_txt = ""
        spatial_right_txt = ""
        use_spatial = (
            tag_bgr is not None
            and is_mateus_dual_column(tag_text)
            and not is_unit_emb_layout(tag_text)
        )
        if use_spatial and tag_bgr is not None:
            left_bgr, right_bgr = split_tag_price_halves(tag_bgr)
            spatial_left_txt = run_ocr_best(
                to_price_ocr_gray(left_bgr),
                psms=(7, 6, 11),
                whitelist="0123456789R$rs., ",
            )
            spatial_right_txt = run_ocr_best(
                to_price_ocr_gray(right_bgr),
                psms=(7, 6, 11),
                whitelist="0123456789R$rs., ",
            )
            # Esquerda: EMB pack (maior). Direita: unitário varejo.
            spatial_atacado = pick_unit_price(spatial_left_txt, prefer="max")
            spatial_varejo = pick_unit_price(spatial_right_txt, prefer="first")

        source_for_name = tag_text if len(tag_text) >= 8 else full_text
        descricao = clean_product_name(
            source_for_name, brand_hint=industria_s
        ) or clean_product_name(full_text, brand_hint=industria_s)

        prices = merge_price_lists(price_text, tag_text, full_text)
        label_varejo, label_atacado, price_strategy = assign_prices_from_labels(
            tag_text, price_text, full_text
        )

        if label_varejo or label_atacado:
            preco_varejo, preco_atacado = label_varejo, label_atacado
            price_strategy = f"label:{price_strategy}"
        elif use_spatial and (spatial_atacado or spatial_varejo):
            preco_atacado = spatial_atacado
            preco_varejo = spatial_varejo
            preco_varejo, preco_atacado = reconcile_spatial_prices(
                preco_varejo, preco_atacado
            )
            if preco_varejo and preco_atacado and preco_varejo == preco_atacado:
                preco_varejo = None
            price_strategy = "spatial-mateus"
        else:
            preco_varejo, preco_atacado = assign_varejo_atacado(prices)
            price_strategy = "assign-fallback"

        preco = preco_varejo or preco_atacado

        _agent_log(
            "H-C",
            "ocr_pesquisa result",
            {
                "build": OCR_BUILD,
                "tagKind": tag_kind,
                "usedTag": tag_gray is not None,
                "tagShape": list(tag_bgr.shape[:2]) if tag_bgr is not None else None,
                "useSpatial": use_spatial,
                "priceStrategy": price_strategy,
                "tagTextSample": tag_text[:160],
                "fullTextSample": full_text[:120],
                "priceTextSample": price_text[:120],
                "spatialLeftSample": spatial_left_txt[:80],
                "spatialRightSample": spatial_right_txt[:80],
                "prices": prices,
                "spatialAtacado": spatial_atacado,
                "spatialVarejo": spatial_varejo,
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
        "industria": industria_s,
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
