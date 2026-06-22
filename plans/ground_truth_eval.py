#!/usr/bin/env python3
"""Ground-truth evaluation utilities for plan detection outputs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


STAIR_WORDS = ("STAIR", "STAIRS", "STAIRCASE", "STEP", "STEPS", "STAIRWELL")
DEFAULT_ACCEPTANCE_THRESHOLDS = {
    "overall_score": 0.85,
    "room_f1": 0.85,
    "mean_iou": 0.65,
    "label_match_rate": 0.85,
    "room_number_match_rate": 0.85,
    "floor_match_rate": 0.90,
    "stairs_location_score": 0.80,
}


def _norm_text(value: str) -> str:
    value = str(value or "").upper().strip()
    value = re.sub(r"[^A-Z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _bbox_iou(a: List[int], b: List[int]) -> float:
    if len(a) != 4 or len(b) != 4:
        return 0.0
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _greedy_iou_match(pred_rooms: List[Dict], gt_rooms: List[Dict], threshold: float = 0.5) -> Tuple[int, float]:
    candidates = []
    for pi, pr in enumerate(pred_rooms):
        pb = pr.get("bbox") or [0, 0, 0, 0]
        for gi, gr in enumerate(gt_rooms):
            gb = gr.get("bbox") or [0, 0, 0, 0]
            iou = _bbox_iou(pb, gb)
            if iou >= threshold:
                candidates.append((iou, pi, gi))
    candidates.sort(reverse=True, key=lambda x: x[0])
    matched_pred = set()
    matched_gt = set()
    iou_sum = 0.0
    for iou, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        iou_sum += iou
    return len(matched_gt), iou_sum


def _counter_match_rate(pred_values: List[str], gt_values: List[str]) -> float:
    gt_clean = [_norm_text(v) for v in gt_values if _norm_text(v)]
    pred_clean = [_norm_text(v) for v in pred_values if _norm_text(v)]
    if not gt_clean:
        return 1.0 if not pred_clean else 0.0
    gt_counter = Counter(gt_clean)
    pred_counter = Counter(pred_clean)
    matched = sum(min(gt_counter[k], pred_counter.get(k, 0)) for k in gt_counter)
    return matched / max(sum(gt_counter.values()), 1)


def _stairs_present(rooms: List[Dict]) -> bool:
    for room in rooms:
        label = _norm_text(room.get("label", ""))
        if room.get("has_stairs") or any(word in label for word in STAIR_WORDS):
            return True
    return False


def _floor_value(room: Dict) -> str:
    value = room.get("floor_idx", room.get("floor", ""))
    return _norm_text(str(value))


def _stairs_bbox(room: Dict) -> List[int]:
    bbox = room.get("stairs_bbox") or room.get("stairs") or []
    if isinstance(bbox, dict):
        bbox = [bbox.get("x", 0), bbox.get("y", 0), bbox.get("w", 0), bbox.get("h", 0)]
    return bbox if isinstance(bbox, list) else []


def _stair_rooms(rooms: List[Dict]) -> List[Dict]:
    out = []
    for room in rooms:
        label = _norm_text(room.get("label", room.get("name", "")))
        if room.get("has_stairs") or any(word in label for word in STAIR_WORDS):
            out.append(room)
    return out


def _match_rooms_for_attributes(pred_rooms: List[Dict], gt_rooms: List[Dict], threshold: float = 0.5) -> List[Tuple[Dict, Dict, float]]:
    candidates = []
    for pi, pr in enumerate(pred_rooms):
        for gi, gr in enumerate(gt_rooms):
            iou = _bbox_iou(pr.get("bbox") or [0, 0, 0, 0], gr.get("bbox") or [0, 0, 0, 0])
            if iou >= threshold:
                candidates.append((iou, pi, gi))
    candidates.sort(reverse=True, key=lambda item: item[0])
    matched_pred = set()
    matched_gt = set()
    pairs = []
    for iou, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        pairs.append((pred_rooms[pi], gt_rooms[gi], iou))
    return pairs


def _floor_match_rate(pred_rooms: List[Dict], gt_rooms: List[Dict]) -> float:
    pairs = _match_rooms_for_attributes(pred_rooms, gt_rooms)
    if not pairs:
        return 0.0 if gt_rooms else 1.0
    comparable = [(p, g) for p, g, _ in pairs if _floor_value(g)]
    if not comparable:
        return 1.0
    matched = sum(1 for p, g in comparable if _floor_value(p) == _floor_value(g))
    return matched / len(comparable)


def _stairs_location_score(pred_rooms: List[Dict], gt_rooms: List[Dict]) -> float:
    gt_stairs = _stair_rooms(gt_rooms)
    pred_stairs = _stair_rooms(pred_rooms)
    if not gt_stairs:
        return 1.0 if not pred_stairs else 0.0
    if not pred_stairs:
        return 0.0

    candidates = []
    for pi, pred in enumerate(pred_stairs):
        pred_box = _stairs_bbox(pred) or pred.get("bbox") or [0, 0, 0, 0]
        for gi, truth in enumerate(gt_stairs):
            truth_box = _stairs_bbox(truth) or truth.get("bbox") or [0, 0, 0, 0]
            iou = _bbox_iou(pred_box, truth_box)
            if iou > 0:
                candidates.append((iou, pi, gi))
    candidates.sort(reverse=True, key=lambda item: item[0])
    matched_pred = set()
    matched_gt = set()
    score = 0.0
    for iou, pi, gi in candidates:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        score += iou
    return score / len(gt_stairs)


@dataclass
class EvalResult:
    key: str
    project_number: str
    image_name: str
    status: str
    room_precision: float
    room_recall: float
    room_f1: float
    mean_iou: float
    label_match_rate: float
    room_number_match_rate: float
    sample_match_rate: float
    floor_match_rate: float
    stairs_expected: int
    stairs_predicted: int
    stairs_correct: int
    stairs_location_score: float
    overall_score: float
    notes: str = ""
    accepted: int = 0


def evaluate_prediction(pred: Dict, truth: Dict, key: str) -> EvalResult:
    pred_rooms = pred.get("rooms") or []
    gt_rooms = truth.get("rooms") or []

    matched, iou_sum = _greedy_iou_match(pred_rooms, gt_rooms, threshold=0.5)
    pred_n = len(pred_rooms)
    gt_n = len(gt_rooms)
    precision = matched / pred_n if pred_n else 0.0
    recall = matched / gt_n if gt_n else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    mean_iou = (iou_sum / matched) if matched else 0.0

    label_rate = _counter_match_rate(
        [r.get("label", "") for r in pred_rooms],
        [r.get("label", "") for r in gt_rooms],
    )
    room_num_rate = _counter_match_rate(
        [r.get("room_number", "") for r in pred_rooms],
        [r.get("room_number", "") for r in gt_rooms],
    )
    sample_rate = _counter_match_rate(pred.get("samples") or [], truth.get("samples") or [])
    floor_rate = _floor_match_rate(pred_rooms, gt_rooms)

    expected_stairs = int(bool(truth.get("stairs_present", _stairs_present(gt_rooms))))
    predicted_stairs = int(_stairs_present(pred_rooms))
    stairs_correct = int(expected_stairs == predicted_stairs)
    stairs_location = _stairs_location_score(pred_rooms, gt_rooms)

    overall = (
        0.32 * f1
        + 0.15 * mean_iou
        + 0.18 * label_rate
        + 0.10 * room_num_rate
        + 0.10 * sample_rate
        + 0.08 * floor_rate
        + 0.03 * stairs_correct
        + 0.04 * stairs_location
    )

    notes = []
    if f1 < 0.6:
        notes.append("low_geometry_f1")
    if label_rate < 0.7:
        notes.append("low_label_match")
    if not stairs_correct:
        notes.append("stairs_mismatch")
    if floor_rate < DEFAULT_ACCEPTANCE_THRESHOLDS["floor_match_rate"]:
        notes.append("low_floor_match")
    if stairs_location < DEFAULT_ACCEPTANCE_THRESHOLDS["stairs_location_score"]:
        notes.append("low_stairs_location")
    accepted = int(
        overall >= DEFAULT_ACCEPTANCE_THRESHOLDS["overall_score"]
        and f1 >= DEFAULT_ACCEPTANCE_THRESHOLDS["room_f1"]
        and mean_iou >= DEFAULT_ACCEPTANCE_THRESHOLDS["mean_iou"]
        and label_rate >= DEFAULT_ACCEPTANCE_THRESHOLDS["label_match_rate"]
        and room_num_rate >= DEFAULT_ACCEPTANCE_THRESHOLDS["room_number_match_rate"]
        and floor_rate >= DEFAULT_ACCEPTANCE_THRESHOLDS["floor_match_rate"]
        and stairs_location >= DEFAULT_ACCEPTANCE_THRESHOLDS["stairs_location_score"]
    )

    return EvalResult(
        key=key,
        project_number=str(pred.get("project_number") or truth.get("project_number") or ""),
        image_name=str(pred.get("image_name") or truth.get("image_name") or key),
        status="ok",
        room_precision=round(precision, 4),
        room_recall=round(recall, 4),
        room_f1=round(f1, 4),
        mean_iou=round(mean_iou, 4),
        label_match_rate=round(label_rate, 4),
        room_number_match_rate=round(room_num_rate, 4),
        sample_match_rate=round(sample_rate, 4),
        floor_match_rate=round(floor_rate, 4),
        stairs_expected=expected_stairs,
        stairs_predicted=predicted_stairs,
        stairs_correct=stairs_correct,
        stairs_location_score=round(stairs_location, 4),
        overall_score=round(max(0.0, min(1.0, overall)), 4),
        notes=",".join(notes),
        accepted=accepted,
    )


def _key_for_path(path: Path) -> str:
    name = path.stem
    if "__" in name:
        return name.split("__", 1)[1].lower()
    return name.lower()


def _looks_like_local_path(value: str) -> bool:
    text = str(value or "")
    return bool(
        re.match(r"^[A-Za-z]:[\\/]", text)
        or text.startswith(("/Users/", "/home/", "/mnt/", "/tmp/"))
        or "\\Users\\" in text
    )


def _validate_truth_payload(data: Dict, path: Path) -> List[str]:
    """Reject draft/local-machine artefacts before they become benchmark truth."""
    errors: List[str] = []
    status = _norm_text(str(data.get("status", "")))
    if "DRAFT" in status or "NEEDS HUMAN" in status:
        errors.append("status is still draft")

    source_image = data.get("source_image")
    if source_image and _looks_like_local_path(str(source_image)):
        errors.append("source_image contains a local absolute path")

    for idx, room in enumerate(data.get("rooms") or [], start=1):
        bbox = room.get("bbox") or []
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"room {idx} has invalid bbox")
            continue
        try:
            width = float(bbox[2])
            height = float(bbox[3])
        except (TypeError, ValueError):
            errors.append(f"room {idx} has non-numeric bbox size")
            continue
        if width <= 0 or height <= 0:
            errors.append(f"room {idx} has zero-size bbox")

    return [f"{path.name}: {err}" for err in errors]


def _load_json_map(folder: Path) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for p in sorted(folder.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
            out[_key_for_path(p)] = data
        except Exception:
            continue
    return out


def evaluate_folders(pred_dir: Path, truth_dir: Path, output_dir: Path) -> Path:
    preds = _load_json_map(pred_dir)
    truths = _load_json_map(truth_dir)
    truth_errors = []
    for p in sorted(truth_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            truth_errors.append(f"{p.name}: invalid JSON ({exc})")
            continue
        truth_errors.extend(_validate_truth_payload(data, p))
    if truth_errors:
        raise ValueError(
            "Ground truth folder contains unapproved or invalid files:\n"
            + "\n".join(f"- {err}" for err in truth_errors)
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[EvalResult] = []
    keys = sorted(set(preds.keys()) | set(truths.keys()))
    for key in keys:
        pred = preds.get(key)
        truth = truths.get(key)
        if pred is None:
            rows.append(
                EvalResult(
                    key=key,
                    project_number=str(truth.get("project_number", "")) if truth else "",
                    image_name=str(truth.get("image_name", key)) if truth else key,
                    status="missing_prediction",
                    room_precision=0.0,
                    room_recall=0.0,
                    room_f1=0.0,
                    mean_iou=0.0,
                    label_match_rate=0.0,
                    room_number_match_rate=0.0,
                    sample_match_rate=0.0,
                    floor_match_rate=0.0,
                    stairs_expected=0,
                    stairs_predicted=0,
                    stairs_correct=0,
                    stairs_location_score=0.0,
                    overall_score=0.0,
                    notes="prediction_not_found",
                )
            )
            continue
        if truth is None:
            rows.append(
                EvalResult(
                    key=key,
                    project_number=str(pred.get("project_number", "")),
                    image_name=str(pred.get("image_name", key)),
                    status="missing_ground_truth",
                    room_precision=0.0,
                    room_recall=0.0,
                    room_f1=0.0,
                    mean_iou=0.0,
                    label_match_rate=0.0,
                    room_number_match_rate=0.0,
                    sample_match_rate=0.0,
                    floor_match_rate=0.0,
                    stairs_expected=0,
                    stairs_predicted=0,
                    stairs_correct=0,
                    stairs_location_score=0.0,
                    overall_score=0.0,
                    notes="ground_truth_not_found",
                )
            )
            continue
        rows.append(evaluate_prediction(pred, truth, key))

    summary = {
        "total": len(rows),
        "matched_pairs": sum(1 for r in rows if r.status == "ok"),
        "missing_prediction": sum(1 for r in rows if r.status == "missing_prediction"),
        "missing_ground_truth": sum(1 for r in rows if r.status == "missing_ground_truth"),
        "avg_overall_score": round(
            sum(r.overall_score for r in rows if r.status == "ok") / max(sum(1 for r in rows if r.status == "ok"), 1),
            4,
        ),
        "accepted": sum(1 for r in rows if r.status == "ok" and r.accepted),
        "acceptance_thresholds": DEFAULT_ACCEPTANCE_THRESHOLDS,
    }

    out_json = output_dir / "ground_truth_eval.json"
    out_csv = output_dir / "ground_truth_eval.csv"
    out_json.write_text(
        json.dumps(
            {
                "summary": summary,
                "results": [r.__dict__ for r in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(EvalResult.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    return out_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate plan predictions against ground-truth JSON files.")
    parser.add_argument("--pred-dir", required=True, help="Folder containing prediction JSON files")
    parser.add_argument("--truth-dir", required=True, help="Folder containing human-corrected ground-truth JSON files")
    parser.add_argument("--output-dir", required=True, help="Folder for evaluation reports")
    args = parser.parse_args()

    report_path = evaluate_folders(Path(args.pred_dir), Path(args.truth_dir), Path(args.output_dir))
    print(f"Ground-truth evaluation written: {report_path}")


if __name__ == "__main__":
    main()
