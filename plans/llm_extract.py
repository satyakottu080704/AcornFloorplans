#!/usr/bin/env python3
"""Run the PRODUCTION AI extractor and write a prediction JSON.

Mirrors ``free_local_extract.py`` but uses the real layout extractor
(``utils.layout_extractor.extract_floor_plan_layout`` -> OpenAI/Gemini per
``PLAN_LAYOUT_PROVIDERS``) so its output can be scored by
``ground_truth_eval.py`` on the same footing as the free/local path.

Usage:
    python plans/llm_extract.py --image path/to/sketch.jpg \
        --output-dir evaluation/predictions_llm
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

from utils.layout_extractor import extract_floor_plan_layout


def _project_from_image(image_path: Path) -> str:
    match = re.search(r"N-?\d{5,}", image_path.name, re.IGNORECASE)
    if not match:
        return image_path.stem
    value = match.group(0).upper()
    return value if "-" in value else f"N-{value[1:]}"


def _layout_to_prediction(image_path: Path, layout: Dict[str, Any]) -> Dict[str, Any]:
    """Map the extractor's {rooms:[{name,x,y}], samples:[{id,x,y}]} layout into
    the shared prediction schema (same as free_local_extract.py)."""
    rooms = []
    for r in layout.get("rooms", []) or []:
        try:
            cx, cy = int(float(r.get("x", 0))), int(float(r.get("y", 0)))
        except (TypeError, ValueError):
            cx, cy = 0, 0
        rooms.append({
            "label": str(r.get("name", "")).strip(),
            "room_number": str(r.get("number", "") or "").strip(),
            # Extractor returns a centre point in 1000x1000 space, not a box.
            # A nominal box keeps the scorer happy; IoU is only meaningful once
            # ground truth carries real boxes.
            "bbox": [cx, cy, 0, 0],
            "floor": str(r.get("floor", "") or "").strip(),
            "room_type": "clear",
        })
    samples = [str(s.get("id", "")).strip() for s in (layout.get("samples", []) or [])]
    return {
        "project_number": _project_from_image(image_path),
        "image_name": image_path.name,
        "source_image": str(image_path.resolve()),
        "detection_method": "ai_llm",
        "image_quality": "n/a",
        "rooms": rooms,
        "samples": [s for s in samples if s],
        "sample_details": layout.get("samples", []) or [],
    }


def extract_llm(image_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    layout = extract_floor_plan_layout(str(image_path))
    payload = _layout_to_prediction(image_path, layout)
    out_path = output_dir / f"{payload['project_number']}__{image_path.stem}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the production AI extractor and write a prediction JSON."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="evaluation/predictions_llm")
    args = parser.parse_args()

    path = extract_llm(Path(args.image), Path(args.output_dir))
    print(f"LLM prediction written: {path}")


if __name__ == "__main__":
    main()
