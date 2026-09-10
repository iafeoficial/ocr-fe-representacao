"""Leitura de `codigos` via Supabase REST (service_role, só SELECT)."""

from __future__ import annotations

from typing import Any

import httpx

from .config import Settings
from .industria import industrias_match, to_industria_padrao


PAGE_SIZE = 1000


async def fetch_codigos_for_industria(
    settings: Settings,
    industria: str,
) -> list[dict[str, Any]]:
    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise RuntimeError("SUPABASE_URL e SUPABASE_SERVICE_ROLE_KEY são obrigatórios")

    target = to_industria_padrao(industria)
    if not target:
        return []

    base = settings.supabase_url.rstrip("/")
    url = f"{base}/rest/v1/codigos"
    headers = {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
        "Accept": "application/json",
    }
    # Busca ampla pelo primeiro token; filtra com industrias_match no cliente.
    token = target.split(" ")[0]
    params = {
        "select": "codigo,produto,industria",
        "industria": f"ilike.*{token}*",
        "order": "codigo.asc",
    }

    rows: list[dict[str, Any]] = []
    offset = 0
    async with httpx.AsyncClient(timeout=30.0) as client:
        while True:
            range_end = offset + PAGE_SIZE - 1
            resp = await client.get(
                url,
                headers={**headers, "Range": f"{offset}-{range_end}"},
                params=params,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not isinstance(batch, list):
                break
            rows.extend(batch)
            if len(batch) < PAGE_SIZE:
                break
            offset += PAGE_SIZE

    return [
        {
            "codigo": str(r.get("codigo") if r.get("codigo") is not None else ""),
            "produto": str(r.get("produto") or "").strip(),
            "industria": to_industria_padrao(str(r.get("industria") or "")),
        }
        for r in rows
        if industrias_match(target, str(r.get("industria") or ""))
        and str(r.get("produto") or "").strip()
    ]
