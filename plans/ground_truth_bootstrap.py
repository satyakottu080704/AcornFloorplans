#!/usr/bin/env python3
"""Create editable ground-truth starter JSON from local/free detection.

This is not an approval tool. It produces a draft JSON file that a surveyor or
operator corrects before it becomes ground truth.
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


def _room_to_json(room) -> Dict[str, Any]:
    return {
        "label": room.label or "",
        "room_number": room.room_number or "",
        "bbox": list(room.bbox),
        "floor": room.floor or "",
        "room_type": room.room_type or "clear",
        "notes": "review_required",
    }


def bootstrap_ground_truth(image_path: Path, output_dir: Path, local_only: bool = True) -> Path:
    image_path = image_path.resolve()
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

    project = _project_from_image(image_path)
    payload = {
        "project_number": project,
        "image_name": image_path.name,
        "source_image": str(image_path),
        "status": "draft_needs_human_correction",
        "detection_method": analysis.detection_method,
        "image_quality": analysis.quality_score,
        "rooms": [_room_to_json(room) for room in analysis.rooms],
        "samples": [
            {
                "id": sample.get("id", ""),
                "material": sample.get("material", ""),
                "location": list(sample.get("location", [])) if sample.get("location") else [],
                "notes": "review_required",
            }
            for sample in analysis.samples
        ],
        "stairs_present": any(
            "stair" in str(room.label or "").lower() or "step" in str(room.label or "").lower()
            for room in analysis.rooms
        ),
        "review_instructions": [
            "Correct every room label and room_number.",
            "Adjust bbox values to the approved room rectangles.",
            "Add floor_idx/floor for every room.",
            "Add stairs_bbox where stair/step location matters.",
            "Only move this JSON into the approved truth folder after review.",
        ],
    }

    out_path = output_dir / f"{project}__{image_path.stem}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap editable ground-truth JSON from local/free detection.")
    parser.add_argument("--image", required=True, help="Input survey sketch image")
    parser.add_argument("--output-dir", default="evaluation/draft_truth", help="Folder for draft JSON")
    parser.add_argument("--allow-ai", action="store_true", help="Allow existing AI OCR/crop-label fallbacks")
    args = parser.parse_args()

    path = bootstrap_ground_truth(
        Path(args.image),
        Path(args.output_dir),
        local_only=not args.allow_ai,
    )
    print(f"Draft ground truth written: {path}")


if __name__ == "__main__":
    main()
