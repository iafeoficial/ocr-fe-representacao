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


def preprocess_for_ocr(data: bytes, *, min_scale: float = 2.0) -> np.ndarray:
    """grayscale → CLAHE → unsharp → upscale mínimo ~2×."""
    bgr = decode_image(data)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.0)
    unsharp = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    h, w = unsharp.shape[:2]
    scale = max(min_scale, 1.0)
    if max(h, w) < 400:
        scale = max(scale, 400 / max(h, w))
    if scale > 1.01:
        unsharp = cv2.resize(
            unsharp,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    return unsharp
