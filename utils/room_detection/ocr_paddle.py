"""PaddleOCR wrapper for local sketch OCR.

Supports PaddleOCR v2.x (ocr API) and v3.x (predict API).
Falls back gracefully if PaddleOCR is broken or unavailable.
"""

from __future__ import annotations

import re
from typing import Dict, List

import cv2
import numpy as np

_paddle_reader = None
_paddle_version = 0  # 2 or 3

# Quick junk filter for form area noise (matches ocr.py _FORM_JUNK_RE)
_FORM_JUNK_RE = re.compile(
    r'(controlled\s*document|issue\s*\d|revision|acorn|analytical|'
    r'surveyor|prepared\s*by|limited|services|0129|address|'
    r'january|february|march|april|may|june|july|august|september|'
    r'october|november|december|^\d{2}/\d{2}/\d{4}$|page\s*\d)',
    re.IGNORECASE
)


def _get_reader():
    global _paddle_reader, _paddle_version
    if _paddle_reader is not None:
        return _paddle_reader

    from paddleocr import PaddleOCR

    # Detect PaddleOCR version from API signature
    import paddleocr
    ver = getattr(paddleocr, "__version__", "2.0.0")
    major = int(ver.split(".")[0]) if ver else 2

    if major >= 3:
        _paddle_version = 3
        try:
            _paddle_reader = PaddleOCR(use_textline_orientation=True, lang="en")
        except (TypeError, ValueError):
            _paddle_reader = PaddleOCR(lang="en")
    else:
        _paddle_version = 2
        try:
            _paddle_reader = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        except (TypeError, ValueError):
            _paddle_reader = PaddleOCR(use_angle_cls=True, lang="en")

    return _paddle_reader


def _erase_blue_hatching(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (100, 40, 40), (130, 255, 255))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray[blue_mask > 0] = 255
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _run_v2(reader, cleaned: np.ndarray) -> List[Dict]:
    """Parse PaddleOCR v2.x output format: [[bbox_pts, (text, conf)], ...]"""
    try:
        result = reader.ocr(cleaned, cls=True)
    except TypeError:
        result = reader.ocr(cleaned)
    if not result or not result[0]:
        return []

    out: List[Dict] = []
    for line in result[0]:
        pts = line[0]
        text = str(line[1][0]).strip()
        conf = float(line[1][1])
        if not text or conf < 0.6:
            continue
        if _FORM_JUNK_RE.search(text):
            continue
        xs = [int(p[0]) for p in pts]
        ys = [int(p[1]) for p in pts]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        out.append({
            "text": text.upper(),
            "location": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
            "bbox": [x1, y1, max(1, x2 - x1), max(1, y2 - y1)],
            "confidence": conf,
            "source": "paddleocr",
        })
    return out


def _run_v3(reader, cleaned: np.ndarray) -> List[Dict]:
    """Parse PaddleOCR v3.x output format using predict()."""
    try:
        results = reader.predict(cleaned)
    except Exception:
        # v3 may also support legacy ocr() call
        try:
            results = reader.ocr(cleaned)
        except Exception:
            return []

    if not results:
        return []

    out: List[Dict] = []
    # v3 predict returns list of page results
    for page in (results if isinstance(results, list) else [results]):
        # Each page result may be a dict with 'rec_texts', 'rec_scores', 'dt_polys'
        if isinstance(page, dict):
            texts = page.get("rec_texts") or page.get("rec_text") or []
            scores = page.get("rec_scores") or page.get("rec_score") or []
            polys = page.get("dt_polys") or page.get("dt_poly") or []
            for text, score, poly in zip(texts, scores, polys):
                text = str(text).strip()
                conf = float(score)
                if not text or conf < 0.6:
                    continue
                if _FORM_JUNK_RE.search(text):
                    continue
                poly = np.array(poly)
                xs = poly[:, 0].astype(int)
                ys = poly[:, 1].astype(int)
                x1, y1 = int(xs.min()), int(ys.min())
                x2, y2 = int(xs.max()), int(ys.max())
                out.append({
                    "text": text.upper(),
                    "location": (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    "bbox": [x1, y1, max(1, x2 - x1), max(1, y2 - y1)],
                    "confidence": conf,
                    "source": "paddleocr",
                })
        elif isinstance(page, (list, tuple)):
            # Legacy format: list of [bbox, (text, conf)]
            return _run_v2(reader, cleaned)

    return out


def run_paddle_ocr(image: np.ndarray) -> List[Dict]:
    """Run PaddleOCR and return normalized OCR items."""
    try:
        reader = _get_reader()
    except Exception as e:
        print(f"[OCR-PADDLE] Init failed: {e}")
        return []

    cleaned = _erase_blue_hatching(image)

    try:
        if _paddle_version >= 3:
            items = _run_v3(reader, cleaned)
        else:
            items = _run_v2(reader, cleaned)
    except Exception as e:
        print(f"[OCR-PADDLE] Runtime error: {e}")
        return []

    if items:
        print(f"[OCR-PADDLE] Found {len(items)} items: {[i['text'] for i in items[:10]]}")
    return items
