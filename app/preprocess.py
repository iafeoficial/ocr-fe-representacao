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


def find_yellow_tag_bgr(bgr: np.ndarray) -> np.ndarray | None:
    """Recorta a maior região amarela (etiqueta de preço Mateus/gondola)."""
    if bgr is None or bgr.size == 0:
        return None
    h, w = bgr.shape[:2]
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
    cnt = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    if area < (h * w) * 0.01:
        return None
    x, y, bw, bh = cv2.boundingRect(cnt)
    pad = max(8, int(min(bw, bh) * 0.06))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop


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
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Retorna (gray_full, gray_tag_or_None, gray_tag_price_or_None)."""
    bgr = decode_image(data)
    full = to_ocr_gray(bgr)
    tag_bgr = find_yellow_tag_bgr(bgr)
    if tag_bgr is None:
        return full, None, None
    return full, to_ocr_gray(tag_bgr, min_scale=2.5), to_price_ocr_gray(tag_bgr)
