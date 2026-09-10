"""Pré-processamento OpenCV antes do Tesseract."""

from __future__ import annotations

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Imagem inválida ou formato não suportado")
    return img


def _crop_padded(
    bgr: np.ndarray, x: int, y: int, bw: int, bh: int, *, pad_frac: float = 0.06
) -> np.ndarray | None:
    h, w = bgr.shape[:2]
    pad = max(8, int(min(bw, bh) * pad_frac))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop


def find_yellow_tag_bgr(bgr: np.ndarray) -> np.ndarray | None:
    """Recorta região amarela de etiqueta (ignora emballage grande tipo sachê verde)."""
    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    img_area = float(h * w)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (15, 70, 70), (42, 255, 255))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)),
        iterations=2,
    )
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    # Prefer smaller yellow rectangles (real price tags), not product packaging.
    candidates: list[tuple[float, int, int, int, int]] = []
    for cnt in cnts:
        area = float(cv2.contourArea(cnt))
        x, y, bw, bh = cv2.boundingRect(cnt)
        box_area = float(bw * bh)
        # Contour can be sparse; also reject huge bounding boxes (green sachets).
        if area < img_area * 0.008 or box_area > img_area * 0.28:
            continue
        if bw < 40 or bh < 24:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 1.2 or aspect > 8.0:
            continue
        # Prefer lower half (shelf edge).
        y_bias = 1.0 + (y + bh / 2) / h
        candidates.append((area * y_bias, x, y, bw, bh))
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, x, y, bw, bh = candidates[0]
    return _crop_padded(bgr, x, y, bw, bh)


def _longest_true_run(flags: np.ndarray) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    i = 0
    n = int(flags.shape[0])
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < n and flags[j]:
            j += 1
        if best is None or (j - i) > (best[1] - best[0]):
            best = (i, j)
        i = j
    return best


def _trim_white_paper_vs_blue_rail(
    bgr: np.ndarray, x: int, y: int, bw: int, bh: int
) -> tuple[int, int, int, int]:
    """Aperta bbox H+V: papel branco vs trilho azul (isola etiqueta no strip full-bleed)."""
    h, w = bgr.shape[:2]
    base_x, base_y = max(0, x), max(0, y)
    x0, y0 = base_x, base_y
    x1, y1 = min(w, x + bw), min(h, y + bh)
    if x1 - x0 < 40 or y1 - y0 < 24:
        return x, y, bw, bh

    sub = bgr[y0:y1, x0:x1]
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 155), (180, 55, 255))
    blue = cv2.inRange(hsv, (95, 70, 40), (130, 255, 255))

    # Horizontal: longest white-dominant column run.
    # Do NOT use bw*0.25 — on full-width shelf strips a real ~396px tag
    # fails when the threshold becomes ~399.
    col_dom = (white.mean(axis=0) > 70) & (blue.mean(axis=0) < 50)
    best_c = _longest_true_run(col_dom)
    min_run_c = max(80, int(min(bw, w) * 0.10))
    if best_c is not None and (best_c[1] - best_c[0]) >= min_run_c:
        x0 = base_x + best_c[0]
        x1 = base_x + best_c[1]

    sub = bgr[y0:y1, x0:x1]
    if sub.size == 0:
        return x, y, bw, bh
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 155), (180, 55, 255))
    blue = cv2.inRange(hsv, (95, 70, 40), (130, 255, 255))
    row_dom = (white.mean(axis=1) > 55) & (blue.mean(axis=1) < 60)
    best_r = _longest_true_run(row_dom)
    min_run_r = max(28, int((y1 - y0) * 0.12))
    if best_r is not None and (best_r[1] - best_r[0]) >= min_run_r:
        y0 = base_y + best_r[0]
        y1 = base_y + best_r[1]

    nw, nh = x1 - x0, y1 - y0
    if nw < 40 or nh < 24:
        return x, y, bw, bh
    return x0, y0, nw, nh


def find_white_shelf_tag_bgr(bgr: np.ndarray) -> np.ndarray | None:
    """Recorta etiqueta branca Mateus (ATACADO|VAREJO) na borda da gôndola."""
    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
    img_area = float(h * w)
    # Focus search on lower 55% (shelf strip); full frame still allowed if needed.
    y0_search = int(h * 0.40)
    roi = bgr[y0_search:, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Low saturation + high value ≈ white label paper.
    mask = cv2.inRange(hsv, (0, 0, 155), (180, 55, 255))
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 9)),
        iterations=2,
    )
    # Break bridges to white UI chrome / letterbox so tag is not full-bleed.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    scored: list[tuple[float, int, int, int, int]] = []
    for cnt in cnts:
        area = float(cv2.contourArea(cnt))
        if area < img_area * 0.008 or area > img_area * 0.28:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        abs_x, abs_y = x, y0_search + y
        if bw < 80 or bh < 36:
            continue
        aspect = bw / max(bh, 1)
        # Mateus labels ~1.3–4; reject near full-frame chrome blobs.
        if aspect < 1.30 or aspect > 6.5:
            continue
        width_frac = bw / max(w, 1)
        touches_both = abs_x <= 4 and (abs_x + bw) >= (w - 4)
        if touches_both or width_frac > 0.88:
            # Still try: trim blue rails then re-check.
            tx, ty, tw, th = _trim_white_paper_vs_blue_rail(bgr, abs_x, abs_y, bw, bh)
            if tw / max(w, 1) > 0.88 or (tx <= 4 and tx + tw >= w - 4):
                continue
            abs_x, abs_y, bw, bh = tx, ty, tw, th
            aspect = bw / max(bh, 1)
            area = float(bw * bh)
            if aspect < 1.30 or bw < 80 or bh < 36:
                continue
        # Prefer inset shelf-edge labels; demote mid-frame packaging whites.
        inset = 1.35 if (abs_x > 8 and abs_x + bw < w - 8) else 0.55
        cy = (abs_y + bh / 2) / max(h, 1)
        shelf_bias = 0.45 if cy < 0.55 else (1.0 + cy)
        score = area * (1.0 + aspect / 4.0) * (1.0 + abs_y / h) * inset * shelf_bias
        scored.append((score, abs_x, abs_y, bw, bh))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    _, x, y, bw, bh = scored[0]
    x, y, bw, bh = _trim_white_paper_vs_blue_rail(bgr, x, y, bw, bh)
    return _crop_padded(bgr, x, y, bw, bh, pad_frac=0.04)


def find_price_tag_bgr(bgr: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Prefere etiqueta branca Mateus; fallback amarela pequena."""
    white = find_white_shelf_tag_bgr(bgr)
    if white is not None:
        return white, "white"
    yellow = find_yellow_tag_bgr(bgr)
    if yellow is not None:
        return yellow, "yellow"
    return None, "none"


def to_ocr_gray(bgr: np.ndarray, *, min_scale: float = 2.0) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    unsharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    h, w = unsharp.shape[:2]
    scale = max(min_scale, 1.0)
    if max(h, w) < 480:
        scale = max(scale, 480 / max(h, w))
    if scale > 1.01:
        unsharp = cv2.resize(
            unsharp,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )
    return unsharp


def to_price_ocr_gray(bgr: np.ndarray) -> np.ndarray:
    """Alto contraste para dígitos de preço na etiqueta."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = max(2.5, 700 / max(h, w))
    gray = cv2.resize(
        gray,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return thr


def preprocess_for_ocr(data: bytes, *, min_scale: float = 2.0) -> np.ndarray:
    """grayscale → CLAHE → unsharp → upscale (compat)."""
    bgr = decode_image(data)
    return to_ocr_gray(bgr, min_scale=min_scale)


def prepare_tag_and_full(
    data: bytes,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
    """Retorna (gray_full, gray_tag, gray_tag_price, tag_bgr_or_None, tag_kind)."""
    bgr = decode_image(data)
    full = to_ocr_gray(bgr)
    tag_bgr, kind = find_price_tag_bgr(bgr)
    if tag_bgr is None:
        return full, None, None, None, kind
    return (
        full,
        to_ocr_gray(tag_bgr, min_scale=2.5),
        to_price_ocr_gray(tag_bgr),
        tag_bgr,
        kind,
    )


def split_tag_price_halves(
    tag_bgr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mateus: esquerda=ATACADO, direita=VAREJO."""
    h, w = tag_bgr.shape[:2]
    # Drop top ~28% (product name strip) so digits dominate.
    y0 = int(h * 0.28)
    body = tag_bgr[y0:, :]
    mid = body.shape[1] // 2
    return body[:, :mid], body[:, mid:]
