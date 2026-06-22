"""Image hash caching for sketch-to-plan pipeline.

Caches GPT-4o/Gemini layout and OCR results keyed by SHA-256 hash of the
preprocessed sketch image. Prevents repeated API calls for the same sketch.

Cache dir: output/cache/sketch_ocr/
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Cache directory (project root / output / cache / sketch_ocr)
_CACHE_DIR = Path(__file__).resolve().parents[3] / "output" / "cache" / "sketch_ocr"


def _ensure_cache_dir() -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def get_sketch_hash(sketch: np.ndarray) -> str:
    """SHA-256 hash of the sketch image bytes (after preprocessing)."""
    return hashlib.sha256(sketch.tobytes()).hexdigest()[:16]


def _cache_path(sketch_hash: str, kind: str) -> Path:
    return _ensure_cache_dir() / f"{sketch_hash}_{kind}.json"


def get_cached_layout(sketch_hash: str) -> Optional[List[Dict]]:
    """Return cached layout result if exists, else None."""
    path = _cache_path(sketch_hash, "layout")
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"[CACHE] Layout cache HIT for {sketch_hash}")
        return data.get("layout_result")
    except (json.JSONDecodeError, KeyError):
        return None


def save_layout_cache(sketch_hash: str, layout_result: List[Dict],
                      image_path: str = "", source: str = "") -> None:
    """Save layout result to cache."""
    path = _cache_path(sketch_hash, "layout")
    # Clean numpy/non-serializable types
    clean = _sanitize(layout_result)
    data = {
        "hash": sketch_hash,
        "timestamp": datetime.now().isoformat(),
        "image_path": image_path,
        "source": source,
        "layout_result": clean,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[CACHE] Layout cached -> {path.name}")


def get_cached_ocr(sketch_hash: str) -> Optional[Dict]:
    """Return cached OCR result if exists, else None."""
    path = _cache_path(sketch_hash, "ocr")
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"[CACHE] OCR cache HIT for {sketch_hash}")
        return data.get("ocr_result")
    except (json.JSONDecodeError, KeyError):
        return None


def save_ocr_cache(sketch_hash: str, ocr_result: Dict,
                   image_path: str = "") -> None:
    """Save OCR result to cache."""
    path = _cache_path(sketch_hash, "ocr")
    clean = _sanitize(ocr_result)
    data = {
        "hash": sketch_hash,
        "timestamp": datetime.now().isoformat(),
        "image_path": image_path,
        "ocr_result": clean,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[CACHE] OCR cached -> {path.name}")


def _sanitize(obj):
    """Convert numpy types and tuples to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
