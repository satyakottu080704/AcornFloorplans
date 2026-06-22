#!/usr/bin/env python3
"""Run the free/local detector and write prediction JSON.

This uses OpenCV geometry plus installed local OCR engines only when
``--local-only`` is used. It does not produce final production output; it is
for benchmarking and building the no-AI path.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.room_detection.detector_legacy import detect_rooms_multi_strategy


def _project_from_image(image_path: Path) -> str:
    match = re.search(r"N-?\d{5,}", image_path.name, re.IGNORECASE)
    if not match:
        return image_path.stem
    value = match.group(0).upper()
    return value if "-" in value else f"N-{value[1:]}"


def _analysis_to_prediction(image_path: Path, analysis) -> Dict[str, Any]:
    return {
        "project_number": _project_from_image(image_path),
        "image_name": image_path.name,
        "source_image": str(image_path.resolve()),
        "detection_method": analysis.detection_method,
        "image_quality": analysis.quality_score,
        "rooms": [
            {
                "label": room.label or "",
                "room_number": room.room_number or "",
                "bbox": list(room.bbox),
                "floor": room.floor or "",
                "room_type": room.room_type or "clear",
            }
            for room in analysis.rooms
        ],
        "samples": [sample.get("id", "") for sample in analysis.samples],
        "sample_details": analysis.samples,
    }


def extract_local(image_path: Path, output_dir: Path, local_only: bool = True) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    old_local = os.environ.get("ACORN_LOCAL_OCR_ONLY")
    if local_only:
        os.environ["ACORN_LOCAL_OCR_ONLY"] = "true"
    try:
        analysis = detect_rooms_multi_strategy(str(image_path), ai_fallback=False)
    finally:
        if old_local is None:
            os.environ.pop("ACORN_LOCAL_OCR_ONLY", None)
        else:
            os.environ["ACORN_LOCAL_OCR_ONLY"] = old_local

    payload = _analysis_to_prediction(image_path, analysis)
    out_path = output_dir / f"{payload['project_number']}__{image_path.stem}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run free/local extraction and write prediction JSON.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", default="evaluation/predictions")
    parser.add_argument("--allow-ai", action="store_true")
    args = parser.parse_args()

    path = extract_local(Path(args.image), Path(args.output_dir), local_only=not args.allow_ai)
    print(f"Local prediction written: {path}")


if __name__ == "__main__":
    main()
