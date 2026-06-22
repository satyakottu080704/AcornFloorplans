"""
Acorn Atlas Floor Plan Pipeline
================================
Pure GPT-4o pipeline (ML model disabled — see config.USE_MODEL).

  Sketch Image
      ↓
  1. Preprocessing (rotate, crop form panel)
      ↓
  2. GPT-4o vision PASS 1 — labels-only (names, numbers, floor,
     ACM, no-access, stairs, samples)
      ↓
  3. GPT-4o vision PASS 2 — full layout (proportional bboxes)
     with uniform-size / coverage / sample retries
      ↓
  4. Merge AI label + layout passes; post-process dedupe
      ↓
  5. Visio COM export — one page per floor (.vsdx)

ML model: the trained ResNet+UNet checkpoint predicts 0% room pixels on
real sketches, so it's disabled. Detector code kept in
utils/room_detection/deep_learning/ for when the model is retrained.
"""

import os
import re
import json
import time
import base64
import subprocess
import hashlib
from difflib import SequenceMatcher
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
from config import DEBUG_MODE

# Load .env
_PROJECT_ROOT = Path(__file__).resolve().parent
_env_file = _PROJECT_ROOT / ".env"
if _env_file.exists():
    for line in _env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Room:
    """A detected room with geometry + labels."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height) in pixels
    area: int = 0
    label: str = ""                  # "Kitchen", "Living Room"
    number: str = ""                 # "001", "02"
    room_type: str = "clear"         # "clear" | "acm" | "no_access"
    has_acm: bool = False
    acm_color: Optional[str] = None
    has_stairs: bool = False
    no_access: bool = False          # Room marked with X or "No Access" text
    floor: str = "Ground Floor"      # Human-readable floor title
    floor_idx: int = 0               # 0=Ground, 1=First, 2=Loft/Second, etc.
    measured_width_m: Optional[float] = None   # Explicit surveyor dimension
    measured_height_m: Optional[float] = None
    dimension_source: str = "estimated"        # "estimated" | "measured"
    contour: Optional[np.ndarray] = None
    label_bbox: Optional[Tuple[int, int, int, int]] = None
    stairs_bbox: Optional[Tuple[int, int, int, int]] = None
    geometry_source: str = "unknown"       # "model" | "ai_bbox" | "unknown"
    detection_confidence: Optional[float] = None
    is_fallback: bool = False



@dataclass
class Sample:
    """A sample annotation from the sketch."""
    id: str = ""                     # "S01", "P002"
    material: str = ""               # "Mastic", "FT", "Putty"
    x_pct: float = 0.0
    y_pct: float = 0.0
    acm_positive: bool = False
    is_ref: bool = False
    target_room_number: Optional[str] = None  # Room the sample arrow points to
    target_floor_idx: int = 0                 # Floor of the target room


@dataclass
class FloorPlan:
    """Complete floor plan result."""
    rooms: List[Room] = field(default_factory=list)
    samples: List[Sample] = field(default_factory=list)
    doors: List[Dict] = field(default_factory=list)
    windows: List[Dict] = field(default_factory=list)
    stairs: List[Dict] = field(default_factory=list)
    floor_title: str = "Ground Floor"
    image_size: Tuple[int, int] = (0, 0)
    project_number: str = ""
    address: str = ""
    detection_time: float = 0.0
    pixel_scale: float = 0.0  # meters per pixel (0 = unknown)
    page: Optional[int] = None          # Page X of the survey
    total_pages: Optional[int] = None   # Page X of TOTAL
    floor_names: Dict[int, str] = field(default_factory=dict)  # {0:'Ground', 1:'First'}


_KNOWN_SAMPLE_MATERIALS = (
    "Bitumen",
    "Felt",
    "Mastic",
    "Putty",
    "Textured Coating",
)


def _normalize_sample_id(value: Any) -> str:
    """Normalize confident handwritten sample-ID OCR substitutions."""
    raw = re.sub(r"[^A-Za-z0-9]", "", str(value or "")).upper()
    if not raw:
        return ""
    prefix = raw[0]
    if prefix not in {"S", "P"}:
        return str(value or "").strip()
    digits = raw[1:].replace("O", "0").replace("I", "1").replace("L", "1")
    if not digits.isdigit() or not 1 <= len(digits) <= 4:
        return str(value or "").strip()
    return prefix + digits.zfill(3)


def _normalize_sample_material(value: Any) -> str:
    """Correct only close OCR matches to known sample materials."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    compact = re.sub(r"[^a-z]", "", raw.lower())
    aliases = {
        "bit": "Bitumen",
        "ft": "Felt",
        "felt": "Felt",
        "mastic": "Mastic",
        "putty": "Putty",
        "tc": "Textured Coating",
        "texturedcoating": "Textured Coating",
    }
    if compact in aliases:
        return aliases[compact]
    candidates = {
        re.sub(r"[^a-z]", "", material.lower()): material
        for material in _KNOWN_SAMPLE_MATERIALS
    }
    best_key = max(
        candidates,
        key=lambda candidate: SequenceMatcher(None, compact, candidate).ratio(),
    )
    score = SequenceMatcher(None, compact, best_key).ratio()
    return candidates[best_key] if score >= 0.65 else raw


# ============================================================================
# Step 1: Preprocessing
# ============================================================================

def preprocess_sketch(image: np.ndarray) -> np.ndarray:
    """Rotate sketch to landscape and crop off the form area."""
    from utils.room_detection.preprocessing import preprocess_sketch as _preprocess
    sketch, _form = _preprocess(image)
    return sketch


# ============================================================================
# Step 2: ML model — YOLOv11s-seg geometry detector.
#
# Re-enabled 2026-05-13. Replaces the old ResNet+UNet path. Detects
# rooms, ACM regions, doors, stairs, loft hatches, etc. as bounding
# boxes (1280px). The Room objects produced here feed into the existing
# merge_results() which combines model geometry with GPT-4o labels.
#
# Falls back gracefully: if YOLO finds fewer than 3 rooms or
# config.USE_MODEL is False, the pipeline runs as the pure GPT-4o path.
# ============================================================================


def _yolo_class_id(name: str) -> int:
    """Return the YOLO class id for a final-class name (case-insensitive)."""
    import config as _cfg
    nm = name.strip().lower()
    for i, c in enumerate(_cfg.CLASSES):
        if c.strip().lower() == nm:
            return i
    return -1


def _bbox_iou_overlap(a, b) -> float:
    """IoU of two [x1,y1,x2,y2] boxes. Used to decide if an ACM/stairs box
    sits inside a room box."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / min(a_area, b_area)   # "contained" measure, not strict IoU


# Module-level cache so we don't reload the YOLO model for every sketch in a batch run.
_yolo_model = None
_yolo_model_path = None


def _minimum_detection_side(sketch_w: int, sketch_h: int) -> int:
    """Minimum useful detection side, relative to the current image size."""
    return max(4, int(round(min(sketch_w, sketch_h) * 0.005)))


def _load_yolo_model(model_path: str):
    """Lazy-load YOLO weights once per process. Returns None on failure."""
    global _yolo_model, _yolo_model_path
    resolved_path = os.path.abspath(model_path)
    if _yolo_model is not None and _yolo_model_path == resolved_path:
        return _yolo_model
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[YOLO] ultralytics not installed — skipping model detection")
        return None
    if not os.path.exists(resolved_path):
        print(f"[YOLO] weights not found at {model_path} — skipping")
        return None
    with open(resolved_path, "rb") as weights_file:
        model_sha = hashlib.sha256(weights_file.read()).hexdigest()
    expected_sha = os.environ.get("ACORN_MODEL_SHA256", "").strip().lower()
    if expected_sha and model_sha.lower() != expected_sha:
        raise RuntimeError(
            f"YOLO model checksum mismatch for {resolved_path}: "
            f"expected {expected_sha}, got {model_sha}"
        )

    print(f"[YOLO] loading {resolved_path}")
    print(f"[YOLO] weights sha256={model_sha}")
    _yolo_model = YOLO(resolved_path)
    _yolo_model_path = resolved_path
    return _yolo_model


def _promote_large_isolated_acm_boxes(
    room_boxes: List[Tuple[np.ndarray, float, bool]],
    acm_boxes: List[Tuple[np.ndarray, float]],
    sketch_w: int,
    sketch_h: int,
) -> List[Tuple[np.ndarray, float, bool]]:
    """Treat large isolated ACM detections as room geometry candidates.

    The trained YOLO model sometimes classifies a whole loft/no-access outline
    as ``acm`` instead of ``room``. Dropping that box removes the detached loft
    from the merge step entirely. Small ACM patches are left as annotations; only
    large boxes that are not already contained in a room are promoted.
    """
    if not acm_boxes:
        return room_boxes

    min_area_ratio = float(os.environ.get("PLAN_PROMOTE_ACM_ROOM_MIN_RATIO", "0.05") or "0.05")
    max_area_ratio = float(os.environ.get("PLAN_PROMOTE_ACM_ROOM_MAX_RATIO", "0.70") or "0.70")
    image_area = max(1.0, float(sketch_w * sketch_h))
    promoted = list(room_boxes)

    for acm_box, acm_conf in acm_boxes:
        x1, y1, x2, y2 = [float(v) for v in acm_box]
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / image_area
        if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
            continue
        if any(_bbox_iou_overlap(room_box, acm_box) >= 0.35 for room_box, _, _ in promoted):
            continue
        promoted.append((acm_box, float(acm_conf), True))
        print(
            "[YOLO] Promoted large isolated ACM box to room candidate "
            f"(area={area_ratio:.1%}, conf={float(acm_conf):.2f})"
        )

    return promoted


def yolo_detect_rooms(
    sketch: np.ndarray,
    original: Optional[np.ndarray] = None,
    model_path: str = None,
    imgsz: int = None,
    conf: float = None,
    project_number: str = "unknown",
) -> List[Room]:
    """Run YOLOv11s-seg and return Room objects in *preprocessed-sketch*
    coordinates.

    Always runs on the preprocessed sketch (lighting normalized, grid suppressed)
    since this matches the training data format and yields highly accurate bounds.
    """
    import config as _cfg
    path = model_path or _cfg.MODEL_PATH
    cf = conf if conf is not None else getattr(_cfg, 'MODEL_CONF_THRESHOLD', 0.15)  # Default to 0.15 from config

    model = _load_yolo_model(path)
    if model is None:
        return []

    sketch_h, sketch_w = sketch.shape[:2]

    # Run multi-scale ensembling if imgsz is not explicitly provided.
    cfg_imgsz = getattr(_cfg, 'MODEL_IMGSZ', 768)
    if imgsz is not None:
        scales = [imgsz]
    else:
        scales = list(set([1024, cfg_imgsz]))
        scales.sort()

    all_rooms_raw = []
    all_acm_boxes = []
    all_stairs_boxes = []

    for current_sz in scales:
        try:
            print(f"[YOLO] Running prediction at imgsz={current_sz}...")
            results = model.predict(sketch, imgsz=current_sz, conf=cf, verbose=False)
            if not results:
                continue
            result = results[0]

            # Auto-Training Active Learning data collection (only run once on first scale)
            if (
                current_sz == scales[0]
                and os.getenv("ACORN_AUTO_TRAINING_ENABLED", "false").strip().lower()
                in ("true", "1", "yes")
            ):
                try:
                    from plans.auto_training import collect_training_sample
                    collect_training_sample(sketch, result, project_number)
                except Exception as ex:
                    print(f"[Auto-Training] Collection skipped or failed: {ex}")

            if result.boxes is None or len(result.boxes) == 0:
                continue

            names = model.names
            room_cid   = next((i for i, name in names.items() if name.lower() == 'room'), -1)
            acm_cid    = next((i for i, name in names.items() if name.lower() == 'acm'), -1)
            stairs_cid = next((i for i, name in names.items() if name.lower() == 'stairs'), -1)

            if room_cid == -1 and any('class' in str(name).lower() for name in names.values()):
                room_cid = 3
                acm_cid = 0
                stairs_cid = 4

            cls_ids = result.boxes.cls.int().tolist()
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for idx in range(len(cls_ids)):
                cid = cls_ids[idx]
                box = boxes_xyxy[idx]
                conf_val = confs[idx]

                if cid == acm_cid:
                    all_acm_boxes.append((box, float(conf_val)))
                elif cid == stairs_cid:
                    all_stairs_boxes.append(box)
                elif cid == room_cid:
                    all_rooms_raw.append((box, float(conf_val), False))
        except Exception as e:
            print(f"[YOLO] predict failed at imgsz={current_sz}: {e!r}")

    all_rooms_raw = _promote_large_isolated_acm_boxes(
        all_rooms_raw, all_acm_boxes, sketch_w, sketch_h
    )

    if not all_rooms_raw:
        return []

    # Sort by confidence descending so highest confidence rooms are kept by NMS
    all_rooms_raw.sort(key=lambda item: -item[1])

    rooms: List[Room] = []
    min_detection_side = _minimum_detection_side(sketch_w, sketch_h)
    for box, conf_val, promoted_acm in all_rooms_raw:
        x1, y1, x2, y2 = box
        # Clip to sketch bounds
        x1c = max(0, min(sketch_w, x1))
        y1c = max(0, min(sketch_h, y1))
        x2c = max(0, min(sketch_w, x2))
        y2c = max(0, min(sketch_h, y2))

        bw = int(round(x2c - x1c))
        bh = int(round(y2c - y1c))
        if bw < min_detection_side or bh < min_detection_side:
            continue
        room = Room(
            bbox=(int(round(x1c)), int(round(y1c)), bw, bh),
            area=bw * bh,
            geometry_source="model",
            detection_confidence=float(conf_val),
            has_acm=bool(promoted_acm),
            acm_color="blue" if promoted_acm else None,
            room_type="acm" if promoted_acm else "clear",
        )
        # Flag ACM if any acm box mostly sits inside this room.
        for ab, _ab_conf in all_acm_boxes:
            if _bbox_iou_overlap(box, ab) >= 0.5:
                room.has_acm = True
                room.acm_color = room.acm_color or "blue"
                room.room_type = "acm"
                break
        # Flag stairs similarly
        for sb in all_stairs_boxes:
            if _bbox_iou_overlap(box, sb) >= 0.5:
                room.has_stairs = True
                break
        rooms.append(room)

    # Run Non-Maximum Suppression to filter out overlapping room boxes
    rooms = _run_nms(rooms, iou_threshold=0.45)

    n_acm = sum(1 for r in rooms if r.has_acm)
    n_stairs = sum(1 for r in rooms if r.has_stairs)
    print(f"[YOLO] Combined multi-scale NMS kept {len(rooms)} room boxes (ACM-flagged: {n_acm}, stairs: {n_stairs})")
    return rooms


def _run_nms(rooms: List[Room], iou_threshold: float = 0.45) -> List[Room]:
    """Filter out overlapping room boxes using NMS (highest area/confidence first)."""
    if not rooms:
        return []
    keep = []
    # Since rooms are already sorted by confidence or area, we can keep the order.
    for r in rooms:
        rx1, ry1, rw, rh = r.bbox
        rx2, ry2 = rx1 + rw, ry1 + rh

        overlap = False
        for k in keep:
            kx1, ky1, kw, kh = k.bbox
            kx2, ky2 = kx1 + kw, ky1 + kh

            # Intersection
            ix1 = max(rx1, kx1)
            iy1 = max(ry1, ky1)
            ix2 = min(rx2, kx2)
            iy2 = min(ry2, ky2)

            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = (rw * rh) + (kw * kh) - inter
                iou = inter / max(1.0, union)
                if iou > iou_threshold:
                    overlap = True
                    break
        if not overlap:
            keep.append(r)
    return keep
# ============================================================================


# ============================================================================
# Step 3: AI Vision — Room Labels (GPT-4o only)
# ============================================================================

# Labels-only prompt — used when model successfully detected room geometry.
# AI only needs to read text, not estimate positions.
_LABELS_ONLY_PROMPT = """You are reading a hand-drawn asbestos survey floor plan sketch.

Your job: READ ALL TEXT, DETECT ACM HATCHING, and READ EVERY RED-PEN SAMPLE.
The computer vision model has already detected room boundaries — you just
need to read the labels and annotations.

For each room provide:
1. NAME exactly as the surveyor wrote it ("Bed 1", "Loft", "Bathroom", etc.)
2. Circled ROOM NUMBER drawn INSIDE a circle on the sketch (BLACK pen).
   Read what the surveyor actually wrote — DO NOT invent a sequential
   number based on reading order. If the circle shows "004", return "004"
   not "001". Only invent a number if no circled number is visible.
3. has_acm: Identify any rooms with diagonal hatching (red OR black lines
    drawn diagonally across the room interior). These are ACM rooms. Mark
    them with has_acm=true. This is CRITICAL — we have NO other ACM
    detector, so if you miss hatching here, the final plan will misreport
    asbestos-containing materials.
    - Even 1 or 2 diagonal lines count as ACM.
    - RED pen diagonals, red dots, red markings, or red sample markers inside a room are ALWAYS ACM: set has_acm=true, acm_color="red".
    - GREEN markings, green lines, or green sample markers inside a room are clear/negative: do NOT set has_acm=true (has_acm=false).
    - Black pen diagonals are ACM when inside a room boundary.
    - An X (two crossed diagonals through the whole room) is NOT hatching,
      that is no_access (rule 6).
4. acm_color: "red", "blue", or "green" if has_acm, else null
5. has_stairs: TRUE if a staircase symbol is drawn inside this room.
   has_stairs and has_acm are INDEPENDENT — a Landing can be BOTH. Do not
   let stair treads cause you to miss ACM hatching in the same room.
   If has_stairs=true, also return stairs_x_pct, stairs_y_pct, stairs_w_pct,
   stairs_h_pct for the visible staircase/steps symbol itself, not the whole
   room. If unclear, return null for all four.
6. no_access: TRUE if an X (two crossed lines) is drawn through the entire
   room, or text like "No Access", "N/A", "Locked", "Inaccessible" is
   written inside. A LOFT with an X is no_access=true, has_acm=false.
7. floor: 0 = Ground Floor, 1 = First Floor, 2 = Loft/Second Floor.
   Lofts are always floor=2 unless labelled otherwise.
8. position: "top-left", "top-center", "top-right", "center-left", "center",
   "center-right", "bottom-left", "bottom-center", "bottom-right".
9. measured_width_m / measured_height_m: if the surveyor wrote dimensions
   inside the room (e.g. "3m x 4m"), return in METRES. Else null.

Also read EVERY RED PEN annotation as a SAMPLE — this is CRITICAL, do not
skip. Samples look like: "S01 FT", "S02 Mastic", "S003 TC", "P001 TC",
"Ref S004". Return the samples array even if empty.

Also read the form panel: floor_name, page, total_pages.

Return ONLY valid JSON:
{
  "floor_title": "Ground Floor",
  "floor_name": "Ground Floor",
  "page": 1,
  "total_pages": 1,
  "room_count": 7,
  "rooms": [
    {"name": "Kitchen", "number": "03", "has_acm": false, "acm_color": null, "has_stairs": false, "no_access": false, "floor": 0, "position": "bottom-left", "measured_width_m": null, "measured_height_m": null},
    {"name": "Living Room", "number": "02", "has_acm": true, "acm_color": "red", "has_stairs": false, "no_access": false, "floor": 0, "position": "top-left", "measured_width_m": null, "measured_height_m": null}
  ],
  "samples": [
    {"id": "S01", "material": "Felt", "acm_positive": false, "is_ref": false, "x_pct": 30, "y_pct": 50, "target_room_number": "001", "target_floor": 2}
  ]
}

CRITICAL RULES:
- Read room names EXACTLY as the surveyor wrote them. "Podiatry" not "Room 44".
- If you can see handwriting but can't read it clearly, give your best guess.
- Room numbers are in CIRCLED BUBBLES in BLACK pen (01, 02, 41, 42 etc.)
- Samples are written in RED pen or GREEN pen: "S01 FT", "S02 Mastic", "P001 TC", "Ref S003"
- "+" after sample, OR if the sample label/drawing is written/drawn in RED pen, means ACM positive: set "acm_positive": true
- If the sample label/drawing is written/drawn in GREEN pen, it is ACM negative: set "acm_positive": false
- "Ref" prefix means cross-reference: set "is_ref": true
- Position field: describe where this room is on the sketch relative to other rooms.

COMPLETENESS — DO NOT MISS ROOMS (most common failure):
- Include EVERY enclosed space, no exceptions: corridors, hallways,
  landings, WCs, bathrooms, cupboards (CPD), airing cupboards, store
  rooms, porches, en-suites — even tiny ones tucked between larger rooms.
- A small cupboard (CPD) drawn as a little box off a hallway IS a room.
  A Landing at the top of stairs IS a room. Bathrooms are rooms.
- Before finishing, COUNT every enclosed box on the sketch and make sure
  "rooms" has exactly that many entries. "room_count" MUST equal the
  length of the "rooms" array — if they disagree, you missed a room: go
  back and find it.

MULTIPLE FLOOR SKETCHES ON ONE PAGE:
- A single page often contains TWO OR MORE separate floor sketches drawn
  side by side (e.g. the upstairs plan beside the downstairs plan).
- Read EVERY sketch on the page. Do not stop after the first one.
- Assign each room the correct "floor" index (0/1/2) for the sketch it
  belongs to. Bedrooms + Landing are usually floor=1; Lounge, Kitchen,
  Hall are usually floor=0.

ROOM NUMBERING (strict):
- Numbers are UNIQUE WITHIN A FLOOR, not across the whole page. The
  ground floor and the first floor may BOTH legitimately contain a room
  "001" — that is correct, keep both, do NOT renumber or drop either.
- Only treat numbers as a true duplicate if they collide on the SAME
  floor; then keep the larger room's number and give the other the next
  unused number on that floor.
- If a room has no circled number visible, still give it a number
  following that floor's sequence.

STAIRS AND LANDINGS (read carefully):
- A LANDING is a room (rectangular walkway at the top of stairs) — it has a number and name, typically "Landing".
- STAIRS are a SYMBOL drawn INSIDE a room (usually the landing or a hallway), not a room of their own. Treat stairs as an attribute of their containing room via "has_stairs": true.
- Do NOT create a separate room called "Stairs" unless the surveyor actually numbered and labelled it as its own room. Merge stairs into the landing/hallway that contains them.
- If you see the word "STAIR ACCESS" or an UP arrow with hatched steps, that's the stair symbol inside the landing — set has_stairs=true on that room only.

COMMERCIAL SKETCHES & EQUIPMENT SAFEGUARD (STRICT):
- Do NOT parse labels of equipment, appliances, or wall fixtures as rooms. Specifically, do NOT return rooms for terms like "Boiler", "Fuse Box", "Distribution Board", "ELECTRICAL DIS BOARD", "DB", "ELEC", "Meters", "Cylinder", or "ATM". Those are wall fixtures, not enclosed rooms.
- Do NOT hallucinate residential room names (like "Kitchen", "Living Room", "Bedroom") on commercial storefront or shop sketches. On a shop/storefront, the main area is "SHOP FLOOR" or "Shop Floor". If there is no bedroom/kitchen drawn, do not invent them.
- Do NOT treat text written outside the walls (like "FRONT OF SHOP" or sample labels like "S01 F.T.") as rooms."""

# Full prompt with bounding boxes — used when model FAILED to detect rooms
# and we need AI to provide both labels AND positions.
_FULL_LAYOUT_PROMPT = """You are reading a hand-drawn asbestos survey floor plan sketch.

Your job: READ THE TEXT, DETECT ACM HATCHING, READ SAMPLE ANNOTATIONS, and
estimate proportional room positions on the sketch.

Look carefully at ALL areas of the sketch including the top-left, top-right,
corners, edges, insets, and areas with hatching, X marks, or unusual symbols.
Do not miss any enclosed room just because it has hatching, an X through it,
or a dense set of annotations inside.

For each room provide:
1. NAME written inside (e.g. "Kitchen", "Bed 1", "Loft", "WC", "Bathroom")
2. Circled ROOM NUMBER drawn INSIDE a circle bubble on the sketch, in BLACK
   pen (e.g. "001", "002", "41"). Read what the surveyor actually wrote —
   do NOT invent a sequential number based on reading order. If a room
   clearly shows "004" in its circle, return "004" even if it's the first
   room you describe. Only invent a number if there is NO circled number
   visible inside the room.
3. has_acm: Identify any rooms with diagonal hatching (red OR black lines
   drawn diagonally across the room interior). These are ACM rooms. Mark
   them with has_acm=true. This is CRITICAL — we have NO other ACM
   detector, so if you miss hatching here the final plan will misreport
   asbestos-containing materials.
   - Even 1 or 2 diagonal lines count as ACM.
   - RED pen diagonals, red dots, red markings, or red sample markers inside a room are ALWAYS ACM: set has_acm=true, acm_color="red".
   - GREEN markings, green lines, or green sample markers inside a room are clear/negative: do NOT set has_acm=true (has_acm=false).
   - BLACK pen diagonals are ACM if inside a room boundary (not the
     building outline).
   - An X (two crossed diagonals through the whole room) is NOT hatching,
     that is no-access (rule 6).
   - A small patch of hatching in one corner of a room is still ACM.
4. acm_color: "red", "blue", or "green" if has_acm, else null
5. has_stairs: TRUE if a staircase symbol (hatched treads, UP arrow, or the
   text "STAIR ACCESS") is drawn INSIDE this room. Usually only the landing
   or hallway has stairs — not a separate room.
   IMPORTANT: has_stairs and has_acm are INDEPENDENT. A Landing with a
   stair symbol AND diagonal hatching lines across the walkway must have
   BOTH has_stairs=true AND has_acm=true. Do not let the stair treads
   cause you to miss ACM hatching in the same room.
   If has_stairs=true, also return stairs_x_pct, stairs_y_pct, stairs_w_pct,
   stairs_h_pct for the visible staircase/steps symbol itself, not the whole
   room. If unclear, return null for all four.
6. no_access: TRUE if an X (two crossed lines) is drawn through the ENTIRE
   room, or the text "No Access", "N/A", "Locked", "Inaccessible" is
   written inside. A LOFT with an X through it is always no_access=true,
   has_acm=false — the X means the surveyor could not access it. Do not
   set both has_acm and no_access on the same room unless the sketch
   clearly shows BOTH diagonal hatching AND a separate "No Access" label.
7. floor: 0 = Ground Floor, 1 = First Floor, 2 = Loft / Second Floor.
   A "Loft" room is always floor=2 unless the sketch clearly labels it
   otherwise. Landing + upstairs bedrooms are usually floor=1.
8. BOUNDING BOX as percentage of the FULL sketch (0-100):
   x_pct, y_pct = top-left corner. w_pct, h_pct = size.
   Bounding boxes MUST accurately reflect the RELATIVE SIZE of each room as
   drawn. A large bedroom must have a larger bbox than a small bathroom. Do
   NOT return all rooms as similar sizes — the surveyor's drawing shows
   genuinely different sizes and you must reproduce that variation.
   Rooms should NOT overlap; adjacent rooms share edges.
9. measured_width_m / measured_height_m: if the surveyor wrote explicit
   dimensions inside or next to the room (e.g. "3m x 4m", "10ft x 12ft"),
   return them in METRES. Convert feet to metres (1 ft = 0.3048 m). If no
   dimensions are written, return null for both.

Also read EVERY RED PEN or GREEN PEN annotation as a SAMPLE. These are CRITICAL — do not
skip them. Samples look like: "S01 FT", "S02 Mastic", "S003 TC",
"P001 TC", "Ref S004". You must return the samples array even if you think
there are none (return [] in that case).
- "id" is the label (S01, S03, P001, etc).
- "material" is the text after the id (FT, Mastic, TC, Felt, Putty, etc).
- "Ref" prefix means cross-reference: set is_ref=true.
- "+" after the sample label, OR if the sample label/drawing is written/drawn in RED pen, means ACM positive: set acm_positive=true.
- If the sample label/drawing is written/drawn in GREEN pen, it is ACM negative: set acm_positive=false.
- x_pct / y_pct: position on the sketch where the sample label is written.
- target_room_number: the room number the sample's arrow/line points to
  (e.g. "001" for a sample pointing into the Loft). If the sample has no
  arrow or can't be linked to a specific room, return null.
- target_floor: 0 / 1 / 2 — the floor of the target room. If target_room
  is null, infer from the material text (e.g. "Loft Felt" → floor=2) or
  return 0.

Also read the FORM PANEL on the left/top of the sketch:
- floor_name: e.g. "Ground Floor", "First Floor", "Loft"
- page: page number if shown (e.g. 1)
- total_pages: total pages if shown (e.g. 2)
If not shown, return null for those fields.

Return ONLY valid JSON (no markdown, no explanation):
{
  "floor_title": "Ground Floor",
  "floor_name": "Ground Floor",
  "page": 1,
  "total_pages": 1,
  "room_count": 7,
  "rooms": [
    {"name": "Kitchen", "number": "03", "has_acm": false, "acm_color": null, "has_stairs": false, "no_access": false, "floor": 0, "x_pct": 5, "y_pct": 60, "w_pct": 25, "h_pct": 35, "measured_width_m": null, "measured_height_m": null},
    {"name": "Living Room", "number": "02", "has_acm": true, "acm_color": "red", "has_stairs": false, "no_access": false, "floor": 0, "x_pct": 5, "y_pct": 5, "w_pct": 40, "h_pct": 50, "measured_width_m": 4.5, "measured_height_m": 5.0}
  ],
  "samples": [
    {"id": "S01", "material": "Felt", "acm_positive": false, "is_ref": false, "x_pct": 30, "y_pct": 50, "target_room_number": "001", "target_floor": 2}
  ]
}

CRITICAL RULES:
- Read room names EXACTLY as the surveyor wrote them ("Bed 1" not "Room 4").
- Include EVERY enclosed space: corridors, hallways, landings, WCs,
  bathrooms, cupboards (CPD), airing cupboards, store rooms, lofts — even
  tiny ones squeezed between larger rooms. Do not merge two rooms into one.
- Before finishing, COUNT every enclosed box drawn on the sketch and make
  sure "rooms" has exactly that many entries. "room_count" MUST equal the
  length of the "rooms" array; if they disagree you missed a room.
- One page may hold TWO OR MORE separate floor sketches side by side
  (e.g. upstairs drawn beside downstairs). Read EVERY sketch on the page
  and give each room the correct "floor" index for the sketch it is in.
- Room numbers are UNIQUE WITHIN A FLOOR, not across the whole page. The
  ground floor and the first floor may BOTH have a room "001" — keep
  both, do NOT renumber or drop either. Only resolve a duplicate when two
  rooms collide on the SAME floor.
- If a room has no visible circled number, assign one following that
  floor's sequence.
- STAIRS: the landing/hallway that contains the stair symbol gets
  has_stairs=true. Do NOT emit a separate room called "Stairs".
- ACM detection is the most important job after reading names. If you are
  unsure whether a room has diagonal lines, look closer — do not skip it.

COMMERCIAL SKETCHES & EQUIPMENT SAFEGUARD (STRICT):
- Do NOT parse labels of equipment, appliances, or wall fixtures as rooms. Specifically, do NOT return rooms for terms like "Boiler", "Fuse Box", "Distribution Board", "ELECTRICAL DIS BOARD", "DB", "ELEC", "Meters", "Cylinder", or "ATM". Those are wall fixtures, not enclosed rooms.
- Do NOT hallucinate residential room names (like "Kitchen", "Living Room", "Bedroom") on commercial storefront or shop sketches. On a shop/storefront, the main area is "SHOP FLOOR" or "Shop Floor". If there is no bedroom/kitchen drawn, do not invent them.
- Do NOT treat text written outside the walls (like "FRONT OF SHOP" or sample labels like "S01 F.T.") as rooms."""


# Room name normalization — common abbreviations on Acorn sketches
_ROOM_NAME_MAP = {
    'k': 'Kitchen', 'kit': 'Kitchen', 'kitch': 'Kitchen',
    'lr': 'Lounge', 'lounge': 'Lounge', 'living': 'Lounge',
    'living room': 'Lounge', 'livingroom': 'Lounge',
    'lobby': 'Lobby', 'loby': 'Lobby', 'lob': 'Lobby',
    'br1': 'Bedroom 1', 'br2': 'Bedroom 2', 'br3': 'Bedroom 3',
    'bed1': 'Bedroom 1', 'bed2': 'Bedroom 2', 'bed3': 'Bedroom 3',
    'bsd': 'Bed', 'bsp': 'Bed',
    # Intentionally NO standalone 'bed' or 'br' entry — if GPT-4o returns
    # a bare "Bed" we keep it as-is rather than normalizing to "Bedroom",
    # so the surveyor can see the AI couldn't read the number.
    'bath': 'Bathroom', 'bathrm': 'Bathroom',
    'wc': 'WC', 'toilet': 'WC',
    'corr': 'Corridor', 'hall': 'Hall', 'hallway': 'Hall',
    'landing': 'Landing', 'land': 'Landing',
    'cup': 'Cupboard', 'cpd': 'Cupboard', "cup'd": 'Cupboard', 'cupd': 'Cupboard',
    'ac': 'Airing Cupboard',
    'gar': 'Garage', 'util': 'Utility Room', 'ut': 'Utility Room',
    'con': 'Conservatory', 'porch': 'Porch', 'ent': 'Entrance',
    'st': 'Store Room', 'store': 'Store Room',
    'off': 'Office', 'rec': 'Reception',
    'din': 'Dining Room', 'dr': 'Dining Room',
    'loft': 'Loft', 'attic': 'Loft',
    'stairs': 'Stairs', 'staircase': 'Stairs',
    'en': 'En-Suite', 'ensuite': 'En-Suite', 'en-suite': 'En-Suite',
    'esu suite': 'En-Suite', 'es suite': 'En-Suite', 'en suite': 'En-Suite',
    'cloak': 'Cloakroom', 'cloakrm': 'Cloakroom',
    'boiler': 'Boiler Room', 'plant': 'Plant Room',
    'wait': 'Waiting Area', 'waiting': 'Waiting Area',
    # External-survey sketches label the area only by orientation ("REAR",
    # "FRONT"); these are not room names — the area is "External". Exact-match
    # only, so real names like "Front Entrance"/"Front Room" are untouched.
    'external': 'External', 'ext': 'External',
    'rear': 'External', 'front': 'External', 'side': 'External', 'gable': 'External',
}


def _normalize_room_name(name: str) -> str:
    """Normalize abbreviated room names to full names."""
    if not name:
        return name
    key = name.lower().strip().rstrip('.')
    key = re.sub(r"\s+", " ", key)
    if "candidate" in key:
        return ""
    # Exact match
    if key in _ROOM_NAME_MAP:
        return _ROOM_NAME_MAP[key]
    # Check if already a full name (e.g. "Boiler Room" — don't re-expand)
    full_names = set(_ROOM_NAME_MAP.values())
    for fn in full_names:
        if key == fn.lower() or name.strip() == fn:
            return name.strip()
    # Check partial matches (e.g. "bed 1" → "Bedroom 1")
    prefix_map = dict(_ROOM_NAME_MAP)
    prefix_map['bed'] = 'Bedroom'
    prefix_map['br'] = 'Bedroom'
    for abbr, full in prefix_map.items():
        if key.startswith(abbr + ' ') and len(abbr) >= 2:
            suffix = name[len(abbr):].strip()
            # Don't add suffix if it would duplicate (e.g. "Room" already in full)
            if suffix.lower() in full.lower():
                return full
            return f"{full} {suffix}" if suffix else full
    return name


def compute_normalized_bbox(x_pct, y_pct, w_pct, h_pct, sketch_w, sketch_h):
    """Convert percentage bbox to pixel coordinates with validation.
    Parameters are expected in 0-100 range; width/height have minimum 1%.
    Returns (x, y, w, h) as ints clamped within sketch dimensions.
    """
    # Default values if missing
    x_pct = float(x_pct) if x_pct is not None else 0.0
    y_pct = float(y_pct) if y_pct is not None else 0.0
    w_pct = float(w_pct) if w_pct is not None else 20.0
    h_pct = float(h_pct) if h_pct is not None else 20.0
    # Clamp percentages
    def clamp(v, min_v=0.0, max_v=100.0):
        return max(min_v, min(max_v, v))
    x_pct = clamp(x_pct)
    y_pct = clamp(y_pct)
    w_pct = max(1.0, clamp(w_pct))
    h_pct = max(1.0, clamp(h_pct))
    # Compute pixel values
    x = int(x_pct / 100 * sketch_w)
    y = int(y_pct / 100 * sketch_h)
    w = max(1, int(w_pct / 100 * sketch_w))
    h = max(1, int(h_pct / 100 * sketch_h))
    # Ensure bbox fits within image
    x = max(0, min(x, sketch_w - w))
    y = max(0, min(y, sketch_h - h))
    if DEBUG_MODE:
        if any(v < 0 or v > 100 for v in [x_pct, y_pct, w_pct, h_pct]):
            print('[WARNING] BBox percentage out of range, clamped')
    return x, y, w, h


def _optional_pct_bbox(data: Dict[str, Any], prefix: str, sketch_w: int, sketch_h: int) -> Optional[Tuple[int, int, int, int]]:
    """Read an optional percentage bbox such as stairs_x_pct/... safely."""
    keys = (f"{prefix}_x_pct", f"{prefix}_y_pct", f"{prefix}_w_pct", f"{prefix}_h_pct")
    if not all(data.get(key) is not None for key in keys):
        return None
    try:
        bbox = compute_normalized_bbox(
            data.get(keys[0]),
            data.get(keys[1]),
            data.get(keys[2]),
            data.get(keys[3]),
            sketch_w,
            sketch_h,
        )
    except Exception:
        return None
    if bbox[2] <= 1 or bbox[3] <= 1:
        return None
    return bbox


def _clean_number(val, fallback: str = "") -> str:
    """
    Coerce a room/sample number from GPT-4o JSON into a clean string.

    GPT-4o frequently returns a JSON `null` for an unreadable number.
    `dict.get("number", default)` only yields `default` when the KEY is
    absent — a present-but-null value sails straight through and
    `str(None)` produces the literal string "None", which downstream
    dedup then treats as a real, shared room number. This normalises all
    of those (None, "", "null", "none", "n/a") to `fallback`.
    """
    s = "" if val is None else str(val).strip()
    if s.lower() in ("", "none", "null", "n/a", "na", "?"):
        return fallback
    # Survey room numbers are circled numeric identifiers. Vision models
    # sometimes copy the room name into this field ("Office", "Bath"), which
    # must not be displayed as a room number.
    if not re.fullmatch(r"\d{1,4}", s):
        return fallback
    return s


def _normalize_number_for_comparison(val: str) -> str:
    """Normalize room numbers for comparison (e.g. 004 -> 4, 04 -> 4) to merge duplicates."""
    if not val:
        return ""
    s_clean = val.lstrip('0')
    if not s_clean:
        return "0"
    return s_clean


# Form-panel "Floor:" values that map to a single floor index.
_PANEL_FLOOR_IDX = {
    "ground": 0, "ground floor": 0, "gf": 0, "g": 0,
    "first": 1, "first floor": 1, "1st": 1, "1st floor": 1, "ff": 1,
    "second": 2, "second floor": 2, "2nd": 2, "2nd floor": 2,
    "loft": 2, "attic": 2,
}
# Form-panel values meaning "this page covers more than one floor".
_PANEL_FLOOR_MULTI = ("all", "various", "multiple", "all floors", "mixed")


# Room types that occur at most ONCE per floor in a domestic survey — used
# to collapse the same room re-read under inconsistent numbers by the tiled
# pass. Bedrooms, cupboards, WCs etc. legitimately repeat, so are excluded.
_SINGLETON_ROOM_TYPES = {
    "kitchen", "living room", "loft", "lobby", "hallway",
    "landing", "dining room", "conservatory", "utility room", "garage",
    "shop floor", "shop",
}
# Asbestos sample material codes — a red-pen sample label ("S003 BIT",
# "002 TC") sometimes gets mistaken for a room name.
_SAMPLE_MATERIAL_WORDS = {
    "bit", "tc", "ft", "felt", "putty", "mastic", "gasket", "rope",
    "cement", "txt", "debris", "ais", "acm",
}


def _looks_like_sample(name: str) -> bool:
    """True if a 'room' name is really a red-pen asbestos sample label.

    Catches material-suffixed labels ("003 BIT", "S02 TC"), sample ids
    anywhere in the text ("S003", "P001"), and cross-reference labels
    ("Ref S002").
    """
    toks = re.sub(r'[^\w\s]', ' ', name.strip().lower()).split()
    if not toks:
        return False
    if toks[-1] in _SAMPLE_MATERIAL_WORDS:
        return True
    if toks[0] == "ref":                                   # "Ref S002"
        return True
    if any(re.fullmatch(r'[sp]\d{1,4}', t) for t in toks):  # S003, P001
        return True
    return False


def _safe_int_floor(val, default: int = 0) -> int:
    """Safely parse a floor value to an integer index (0=Ground, 1=First, 2=Loft/Second)."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip().lower()
    if not s:
        return default
    # Try mapping textual names
    if "ground" in s or "gf" == s or s.rstrip('.').endswith("g") or s.startswith("g "):
        return 0
    if "first" in s or "ff" == s or "1st" in s or s.rstrip('.').endswith("f") or s.startswith("f "):
        return 1
    if "second" in s or "2nd" in s or "loft" in s or "attic" in s:
        return 2
    # Try direct integer parsing
    try:
        # Catch case where it is "0 = Ground Floor"
        m = re.match(r'^(\d+)', s)
        if m:
            return int(m.group(1))
        return int(float(s))
    except ValueError:
        pass
    return default


def _dedup_room_list(ai: Optional[Dict]) -> None:
    """
    Clean the room list from the tiled labels-only read, in place.

    Tiling boosts recall but is noisy: the same room surfaces in several
    overlapping crops (often with an inconsistent or missing number), GPT-4o
    sometimes mashes the number into the name ("005 lounge"), and red-pen
    sample labels occasionally get read as rooms. This collapses those
    without losing genuinely distinct rooms (two different bedrooms, two
    different cupboards on the same floor are kept).
    """
    if not ai or not ai.get("rooms"):
        return
    rooms = ai["rooms"]

    # Pass 1: tidy names, split off mashed-in numbers, drop sample labels.
    cleaned = []
    for r in rooms:
        raw = str(r.get("name") or "").strip()
        m = re.match(r'^(\d{1,4})\s+(.+)$', raw)
        if m and not _clean_number(r.get("number")):
            r["number"] = m.group(1)
            raw = m.group(2).strip()
        else:
            # Also check for digits at the end of the name (e.g. "Bathroom 009", "Kitchen 08")
            m2 = re.match(r'^(.+?)\s+(\d{1,4})$', raw)
            if m2 and not _clean_number(r.get("number")):
                r["number"] = m2.group(2)
                raw = m2.group(1).strip()
        if _looks_like_sample(raw):
            print(f"[DEDUP] Dropped '{raw}' — reads as a sample, not a room")
            continue
        r["name"] = _normalize_room_name(raw)
        if not r["name"]:
            print(f"[DEDUP] Dropped '{raw}' - not a usable room label")
            continue
        cleaned_num = _clean_number(r.get("number"))
        r["number"] = cleaned_num if cleaned_num else ""
        cleaned.append(r)

    # Pass 2: merge duplicates by (floor, normalized name). A singleton room
    # type collapses to one; other types collapse only when they share a
    # number, or one copy has no number (a numberless re-read).
    out = []
    for r in cleaned:
        nm = str(r.get("name") or "").strip()
        fl = r.get("floor")
        fl = _safe_int_floor(fl) if fl is not None else None
        num = _clean_number(r.get("number"))
        match = None
        for o in out:
            if str(o.get("name") or "").strip().lower() != nm.lower() or not nm:
                continue
            ofl = o.get("floor")
            ofl = _safe_int_floor(ofl) if ofl is not None else None
            if fl is not None and ofl is not None and fl != ofl:
                continue
            onum = _clean_number(o.get("number"))
            if nm.lower() in _SINGLETON_ROOM_TYPES or any(c.isdigit() for c in nm):
                match = o
                break
            if not num or not onum or _normalize_number_for_comparison(num) == _normalize_number_for_comparison(onum):
                match = o
                break
        if match is None:
            out.append(r)
            continue
        if not _clean_number(match.get("number")) and num:
            match["number"] = num
        if match.get("floor") is None and r.get("floor") is not None:
            match["floor"] = r.get("floor")
        for flag in ("has_acm", "no_access", "has_stairs"):
            if r.get(flag):
                match[flag] = True
        print(f"[DEDUP] Merged duplicate room '{nm}' #{num or '?'}")

    if len(out) != len(rooms):
        print(f"[DEDUP] Tiled room list cleaned: {len(rooms)} -> {len(out)} rooms")
    ai["rooms"] = out
    ai["room_count"] = len(out)


def _is_valid_single_floor(val) -> bool:
    if not val:
        return False
    s = str(val).strip().lower()
    if s in _PANEL_FLOOR_IDX:
        return True
    if s.isdigit():
        return True
    if any(x in s for x in ["ground", "first", "second", "loft", "attic", "floor"]):
        return True
    return False


def _apply_panel_floor(ai: Optional[Dict]) -> None:
    """
    Stamp every room with a consistent floor index from the survey form
    panel's "Floor:" field, in place.

    GPT-4o's per-room "floor" guesses are noisy and — critically — differ
    between the labels-only and full-layout passes. Reconciliation matches
    rooms by (floor, name), so a room tagged floor 0 in one pass and floor 1
    in another looks like two different rooms and spawns a phantom
    duplicate. The form panel's "Floor:" field is authoritative: when it
    names ONE floor, every room on the page is on that floor. A room
    explicitly named "Loft"/"Attic" still goes to the loft index. When the
    panel says "All"/"Various" (a genuine multi-floor page) or is missing,
    the per-room guesses are left untouched.
    """
    if not ai or not ai.get("rooms"):
        return
    
    # Always apply the panel floor stamp if the panel designates a specific floor
    # to clean up any floor hallucinations (e.g., Ground Floor rooms on a 1st Floor sheet).


    panel = str(ai.get("floor_name") or "").strip().lower()
    if not panel or panel in _PANEL_FLOOR_MULTI or not _is_valid_single_floor(panel):
        return
    
    if panel in _PANEL_FLOOR_IDX:
        page_idx = _PANEL_FLOOR_IDX[panel]
    else:
        page_idx = _safe_int_floor(panel, default=0)

    explicit_loft_rooms = []
    non_loft_rooms = []
    for room in ai["rooms"]:
        label = str(room.get("name") or "").strip().lower()
        if "loft" in label or "attic" in label:
            explicit_loft_rooms.append(room)
        else:
            non_loft_rooms.append(room)

    # A detached Loft plus unanimous room-level floor evidence is stronger
    # than the small survey-panel OCR field.
    non_loft_floors = {
        _safe_int_floor(room.get("floor"))
        for room in non_loft_rooms
        if room.get("floor") is not None
    }
    preserve_room_floor = (
        bool(explicit_loft_rooms)
        and bool(non_loft_rooms)
        and len(non_loft_floors) == 1
        and next(iter(non_loft_floors)) != page_idx
    )
    if preserve_room_floor:
        room_floor = next(iter(non_loft_floors))
        for room in explicit_loft_rooms:
            room["floor"] = 2
        print(
            f"[FLOOR] Preserving unanimous room floor index {room_floor}; "
            f"panel OCR '{panel}' conflicts and an explicit Loft is present"
        )
        return

    changed = 0
    for r in ai["rooms"]:
        lbl = str(r.get("name") or "").strip().lower()
        target = 2 if ("loft" in lbl or "attic" in lbl) else page_idx
        if _safe_int_floor(r.get("floor", 0)) != target:
            r["floor"] = target
            changed += 1
    if changed:
        print(f"[FLOOR] Form panel says '{panel}' — stamped {changed} room(s) "
              f"to consistent floor index {page_idx} (loft to 2)")


def _normalize_explicit_loft_access(ai: Optional[Dict]) -> None:
    """Do not treat a small loft hatch/access marker as room-wide no access."""
    if not ai:
        return
    for room in ai.get("rooms") or []:
        label = str(room.get("name") or "").strip().lower()
        if "loft" not in label and "attic" not in label:
            continue
        access_text = " ".join(
            str(room.get(key) or "").strip().lower()
            for key in ("name", "access", "access_status", "notes", "status")
        )
        explicitly_no_access = (
            "no access" in access_text
            or "not accessed" in access_text
            or "inaccessible" in access_text
        )
        if room.get("no_access") and not explicitly_no_access:
            room["no_access"] = False
            print("[ACCESS] Cleared Loft no-access flag without explicit no-access text")


def _propagate_explicit_multifloor_evidence(
    labels_data: Optional[Dict],
    layout_data: Optional[Dict],
) -> None:
    """Copy reliable main-floor plus Loft separation into cached layout data."""
    if not labels_data or not layout_data:
        return
    label_rooms = labels_data.get("rooms") or []
    layout_rooms = layout_data.get("rooms") or []
    label_lofts = [
        room for room in label_rooms
        if "loft" in str(room.get("name") or "").lower()
        or "attic" in str(room.get("name") or "").lower()
    ]
    label_main = [room for room in label_rooms if room not in label_lofts]
    main_floors = {
        _safe_int_floor(room.get("floor"))
        for room in label_main
        if room.get("floor") is not None
    }
    if not label_lofts or not label_main or len(main_floors) != 1:
        return

    main_floor = next(iter(main_floors))
    changed = 0
    for room in layout_rooms:
        label = str(room.get("name") or "").lower()
        target = 2 if ("loft" in label or "attic" in label) else main_floor
        if _safe_int_floor(room.get("floor")) != target:
            room["floor"] = target
            changed += 1
    if changed:
        print(
            f"[FLOOR] Propagated explicit multi-floor evidence to "
            f"{changed} full-layout room(s)"
        )


def _call_gpt4o(api_key: str, b64: str, prompt: str, max_tokens: int = 4000) -> Optional[str]:
    """Make a GPT-4o API call with retries. Returns raw response text or None.

    Retry policy: 3 attempts total. Waits 0s / 5s / 15s before each.
    Retries on SSL errors, network errors, timeouts, and transient HTTP
    status codes (408, 429, 5xx). Non-transient HTTP errors (4xx except
    408/429) fail fast — no point retrying a bad request.
    """
    import httpx
    model = os.environ.get("OPENAI_VISION_MODEL", "gpt-4o").strip() or "gpt-4o"
    waits = (0, 5, 15)

    for attempt_idx, wait in enumerate(waits, start=1):
        if wait:
            time.sleep(wait)
        _t0 = time.time()
        if attempt_idx == 1:
            print(f"[GPT4O] -> POST {model} (image ~{len(b64)//1024}KB)")
        else:
            print(f"[RETRY {attempt_idx}/3] POST {model} (after {wait}s wait)")
        try:
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            dt = time.time() - _t0
            print(f"[GPT4O] <- 200 in {dt:.1f}s")
            return resp.json()["choices"][0]["message"]["content"].strip()

        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            body = e.response.text[:200]
            print(f"[GPT4O] <- HTTP {code} in {time.time()-_t0:.1f}s: {body}")
            # 4xx (except 408/429) = bad request, won't get better with retry.
            if 400 <= code < 500 and code not in (408, 429):
                return None
            # 5xx / 408 / 429 => fall through and retry.
        except httpx.TimeoutException:
            print(f"[GPT4O] <- TIMEOUT after {time.time()-_t0:.1f}s")
        except Exception as e:
            # Network / SSL / ReadError / ConnectError all land here.
            print(f"[GPT4O] <- ERROR in {time.time()-_t0:.1f}s: {type(e).__name__}: {str(e)[:200]}")

        if attempt_idx < len(waits):
            print(f"[RETRY {attempt_idx+1}/3] will wait {waits[attempt_idx]}s and retry")

    print("[GPT4O] FAILED after 3 attempts")
    return None


def _call_gemini(api_key: str, b64: str, prompt: str, max_tokens: int = 4000) -> Optional[str]:
    """Make a Gemini 2.5 Flash Vision API call with retries. Returns raw response text or None.

    Retry policy: 3 attempts total. Waits 0s / 5s / 15s before each.
    """
    import httpx
    model = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash")
    waits = (0, 5, 15)

    for attempt_idx, wait in enumerate(waits, start=1):
        if wait:
            time.sleep(wait)
        _t0 = time.time()
        if attempt_idx == 1:
            print(f"[GEMINI] -> POST {model} (image ~{len(b64)//1024}KB)")
        else:
            print(f"[RETRY {attempt_idx}/3] POST {model} (after {wait}s wait)")
        try:
            resp = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": b64
                                }
                            }
                        ]
                    }],
                    "generationConfig": {
                        "temperature": 0,
                        "maxOutputTokens": max_tokens,
                        "responseMimeType": "application/json"
                    },
                    "safetySettings": [
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}
                    ]
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            dt = time.time() - _t0
            print(f"[GEMINI] <- 200 in {dt:.1f}s")
            
            result = resp.json()
            candidates = result.get("candidates", [])
            if not candidates:
                return None
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts if "text" in part)
            return text.strip()

        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            body = e.response.text[:200]
            print(f"[GEMINI] <- HTTP {code} in {time.time()-_t0:.1f}s: {body}")
            # 4xx (except 408/429) = bad request, won't get better with retry.
            if 400 <= code < 500 and code not in (408, 429):
                return None
        except httpx.TimeoutException:
            print(f"[GEMINI] <- TIMEOUT after {time.time()-_t0:.1f}s")
        except Exception as e:
            print(f"[GEMINI] <- ERROR in {time.time()-_t0:.1f}s: {type(e).__name__}: {str(e)[:200]}")

        if attempt_idx < len(waits):
            print(f"[RETRY {attempt_idx+1}/3] will wait {waits[attempt_idx]}s and retry")

    print("[GEMINI] FAILED after 3 attempts")
    return None



def _parse_json(raw: str) -> Optional[Dict]:
    """Parse GPT-4o / Gemini JSON response with cleanup and list-wrapping safeguard."""
    if not raw:
        return None
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE).strip()
    raw = re.sub(r',\s*}', '}', raw)
    raw = re.sub(r',\s*]', ']', raw)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            parsed = {"rooms": parsed}
        return parsed
    except json.JSONDecodeError:
        try:
            parsed = json.loads(raw.replace("'", '"'))
            if isinstance(parsed, list):
                parsed = {"rooms": parsed}
            return parsed
        except json.JSONDecodeError as e:
            print(f"[GPT4O] JSON parse error: {e}")
            return None


def _encode_sketch(sketch: np.ndarray, max_dim: int = 2000) -> str:
    """Encode sketch as base64 JPEG for GPT-4o."""
    h, w = sketch.shape[:2]
    img = sketch.copy()
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return base64.b64encode(buf.tobytes()).decode()


# Quadrant prompt for zoomed-in reading
_QUADRANT_PROMPT = """You are reading a ZOOMED-IN SECTION of a hand-drawn asbestos
survey floor plan. Because this is a magnified crop you can see detail the
full-page reader misses — read EVERY room in this section.

Small rooms are the ones most often missed: cupboards (CPD), WCs, stores,
lobbies, landings, porches. Do NOT skip them. Include every enclosed box,
even tiny ones, and rooms only partly visible at the crop edges.

For each room provide:
- name: room name EXACTLY as written ("Kitchen", "Store", "Loby", "CPD").
- number: the circled room number in black pen, or null if none is visible.
- has_acm: true if diagonal hatching (red or black) crosses the room.
- no_access: true if an X is drawn across the whole room.
- floor: 0 ground, 1 first, 2 loft — or null if not determinable.

Return ONLY valid JSON:
{"rooms": [{"name": "Kitchen", "number": "008", "has_acm": false, "no_access": false, "floor": 0}]}

Rules:
- Read names EXACTLY as written — never return just a bare number as a name.
- Room numbers are in circled bubbles in black pen.
- Do NOT return asbestos SAMPLE annotations as rooms. Samples are red-pen
  labels like "S001", "S003 BIT", "S02 TC", "Ref S002", "P001 Putty", or a
  bare material code ("TC", "BIT", "FT", "Putty", "Mastic"). Those are NOT
  rooms — only return enclosed rooms that have an actual room name.

COMMERCIAL SKETCHES & EQUIPMENT SAFEGUARD (STRICT):
- Do NOT parse labels of equipment, appliances, or wall fixtures as rooms. Specifically, do NOT return rooms for terms like "Boiler", "Fuse Box", "Distribution Board", "ELECTRICAL DIS BOARD", "DB", "ELEC", "Meters", "Cylinder", or "ATM". Those are wall fixtures, not enclosed rooms.
- Do NOT hallucinate residential room names (like "Kitchen", "Living Room", "Bedroom") on commercial storefront or shop sketches. On a shop/storefront, the main area is "SHOP FLOOR" or "Shop Floor". If there is no bedroom/kitchen drawn, do not invent them.
- Do NOT treat text written outside the walls (like "FRONT OF SHOP" or sample labels like "S01 F.T.") as rooms."""


def get_room_labels_gpt4o(sketch: np.ndarray, prompt: str = None) -> Optional[Dict]:
    """
    Get room labels from the configured OpenAI vision model.
    """
    gemini_keys = []
    for i in range(1, 11):
        k = os.environ.get(f"GEMINI_API_KEY_{i}" if i > 1 else "GEMINI_API_KEY", "")
        if k:
            gemini_keys.append(k)

    gemini_key = gemini_keys[0] if gemini_keys else ""
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    use_gemini = False

    try:
        import httpx
    except ImportError:
        print("[AI] httpx not installed. Run: pip install httpx")
        return None

    b64 = _encode_sketch(sketch)
    use_prompt = prompt or _LABELS_ONLY_PROMPT

    # Decide primary and secondary provider
    primary_provider = "GEMINI" if use_gemini else "GPT4O"
    primary_key = gemini_key if use_gemini else openai_key
    primary_fn = _call_gemini if use_gemini else _call_gpt4o

    secondary_provider = "GPT4O" if use_gemini else None
    secondary_key = openai_key if use_gemini else None
    secondary_fn = _call_gpt4o if use_gemini else None

    # ---- Pass 1: Full sketch ----
    data = None
    provider = primary_provider
    api_key = primary_key
    call_fn = primary_fn
    gemini_key_idx = 0

    for attempt in range(1, 4):
        print(f"[{provider}] Attempt {attempt}/3 — reading sketch labels...")
        raw = call_fn(api_key, b64, use_prompt, max_tokens=4000)
        parsed_data = _parse_json(raw) if raw else None

        # If primary (Gemini) failed (empty, failed to parse, or missing rooms list),
        # try rotating keys or check if we should fall back to OpenAI
        is_invalid = not raw or not parsed_data or not parsed_data.get("rooms")
        if is_invalid and provider == "GEMINI":
            if gemini_key_idx + 1 < len(gemini_keys):
                gemini_key_idx += 1
                api_key = gemini_keys[gemini_key_idx]
                print(f"[GEMINI] Key #{gemini_key_idx} rate-limited or failed. Rotating to Key #{gemini_key_idx + 1}...")
                raw = call_fn(api_key, b64, use_prompt, max_tokens=4000)
                parsed_data = _parse_json(raw) if raw else None
                is_invalid = not raw or not parsed_data or not parsed_data.get("rooms")

            if is_invalid and secondary_key:
                print("=" * 80)
                print(" [QUOTA ALERT/TRUNCATION] GEMINI API RETURNED TRUNCATED OR INVALID DATA!")
                print(" FALLING BACK AUTOMATICALLY TO OPENAI CHATGPT (GPT-4o) AS SECONDARY...")
                print("=" * 80)
                provider = secondary_provider
                api_key = secondary_key
                call_fn = secondary_fn
                # Retry immediately with secondary
                print(f"[{provider}] Retry Attempt 1/3 — reading sketch labels via fallback...")
                raw = call_fn(api_key, b64, use_prompt, max_tokens=4000)
                parsed_data = _parse_json(raw) if raw else None

        if parsed_data and parsed_data.get("rooms"):
            data = parsed_data
            break

        time.sleep(2 * attempt)


    if not data or not data.get("rooms"):
        print(f"[{provider}] All attempts failed")
        return None

    rooms = data["rooms"]
    named_count = sum(1 for r in rooms
                      if not re.fullmatch(r'\d+', str(r.get('name', '')).strip()))
    total = len(rooms)
    print(f"[{provider}] Found {total} rooms ({named_count} with names): "
          f"{[r.get('name', '?') for r in rooms[:8]]}{'...' if total > 8 else ''}")

    # ---- Pass 1.5: Completeness recovery ----
    # The single most common pipeline failure is a room silently dropped
    # from the rooms array — usually a small cupboard, landing, or bathroom
    # squeezed between larger rooms. We run a dedicated second-opinion pass
    # that shows AI what was found and asks only for the omissions.
    try:
        stated = int(data.get("room_count") or 0)
    except (TypeError, ValueError):
        stated = 0
    is_primary_labels = use_prompt is _LABELS_ONLY_PROMPT or use_prompt is None
    single_pass_only = os.environ.get("PLAN_SINGLE_PASS_ONLY", "false").strip().lower() in ("true", "1", "yes")

    if (is_primary_labels or stated > total) and not single_pass_only:
        if stated > total:
            print(f"[{provider}] room_count={stated} but only {total} rooms listed "
                  f"— running completeness pass")
        else:
            print(f"[{provider}] Running completeness pass ({total} rooms found)")
        with_bbox = use_prompt is _FULL_LAYOUT_PROMPT
        recovered = _find_omitted_rooms(api_key, b64, rooms, with_bbox=with_bbox, call_fn=call_fn, provider_name=provider)
        if recovered:
            rooms.extend(recovered)
            data["rooms"] = rooms
            total = len(rooms)
            data["room_count"] = total
            print(f"[{provider}] Completeness pass recovered {len(recovered)} "
                  f"room(s): {[r.get('name', '?') for r in recovered]} — "
                  f"total now {total}")
            # Recovered rooms changed the counts — refresh named_count so the
            # quadrant trigger below isn't skewed by stale values.
            named_count = sum(1 for r in rooms
                              if not re.fullmatch(r'\d+', str(r.get('name', '')).strip()))

    # ---- Pass 2: Quadrant mode if names are mostly numbers or many rooms ----
    # The full-page read systematically misses small rooms because the
    # sketch is downscaled to fit the model's input. The quadrant pass
    # crops the sketch into 4 overlapping pieces and reads each at higher
    # effective resolution — the only reliable way to recover the small
    # rooms. We ALWAYS tile the primary labels-only pass (it is the room-
    # count anchor for the whole pipeline); other passes tile only when
    # names are unreadable or the sketch is already known to be complex.
    need_quadrants = (
        not single_pass_only and (
            (is_primary_labels and total >= 12)  # tile the anchor pass only if we have at least 12 rooms to prevent hallucinations on standard sketches
            or (named_count < total * 0.4 and total > 5)  # can't read names
            or total >= 15  # complex sketch, AI likely missed rooms at edges
            or (is_primary_labels and sketch.shape[0] / sketch.shape[1] >= 1.4)  # tall portrait sheet with stacked layouts
        )
    )
    if need_quadrants:
        if is_primary_labels:
            reason = f"primary read found {total} room(s) — tiling for recall"
        elif named_count < total * 0.4:
            reason = "unreadable names"
        else:
            reason = f"complex sketch ({total} rooms)"
        # Tiling grid is 2x2 by default. A finer 3x3 grid was measured on dense
        # plans (N-105325/N-105005) and did NOT help — it slightly hurt (rooms
        # fragment across more tiles), confirming dense-plan drift is a vision-
        # reading limit, not magnification. Left configurable via DENSE_TILE_GRID
        # for future experiments, default 2x2.
        g = int(os.getenv("DENSE_TILE_GRID", "2").strip() or "2")
        grid = (g, g) if (g > 2 and total >= 14) else (2, 2)
        print(f"[{provider}] Quadrant mode: {reason} (grid {grid[0]}x{grid[1]})")
        quadrant_names = _read_quadrants(api_key, sketch, call_fn=call_fn, provider_name=provider, grid=grid)
        if quadrant_names:
            # Upgrade numeric main-room names with real names from quadrants.
            _merge_quadrant_names(rooms, quadrant_names)
            # Add NEW rooms the full-page read missed. Dedup by room number
            # when present, otherwise by normalized name — this keeps a
            # numberless "Store" or "Loby" (which the old number-only check
            # silently discarded) while still collapsing the same room seen
            # in two overlapping quadrants.
            def _room_key(r):
                num = _clean_number(r.get("number"))
                if num:
                    return f"#{_normalize_number_for_comparison(num)}"
                nm = _normalize_room_name(str(r.get("name", "")).strip())
                return nm.lower() if nm else ""

            seen_keys = {k for k in (_room_key(r) for r in rooms) if k}
            added = 0
            for qr in quadrant_names:
                qname = str(qr.get("name", "")).strip()
                if not qname or re.fullmatch(r'\d+', qname):
                    continue  # need a real name, not a bare number
                key = _room_key(qr)
                if not key or key in seen_keys:
                    continue
                seen_keys.add(key)
                rooms.append(qr)
                added += 1
            if added:
                data["rooms"] = rooms
                data["room_count"] = len(rooms)
                print(f"[{provider}] Added {added} new room(s) from quadrants, "
                      f"total now {len(rooms)}: "
                      f"{[r.get('name', '?') for r in rooms[-added:]]}")
            named_after = sum(1 for r in rooms
                              if not re.fullmatch(r'\d+', str(r.get('name', '')).strip()))
            print(f"[{provider}] After quadrants: {named_after}/{len(rooms)} names readable")

    # ---- Normalize room names ----
    for room in rooms:
        name = str(room.get("name", "")).strip()
        room["name"] = _normalize_room_name(name)

    # ---- Clean the (noisy, higher-recall) tiled list ----
    # Tiling recovers missed rooms but re-reads some rooms several times and
    # can mistake sample labels for rooms. Collapse that before the result
    # is cached and handed to reconciliation.
    _apply_panel_floor(data)
    _dedup_room_list(data)

    return data



def _find_omitted_rooms(
    api_key: str,
    b64: str,
    found_rooms: List[Dict],
    with_bbox: bool = False,
    call_fn: Any = None,
    provider_name: str = "GPT4O",
) -> List[Dict]:
    """
    Second-opinion pass: given the rooms a first reading found, ask the AI
    to name only the rooms it MISSED.

    This catches the most common pipeline failure — a small cupboard,
    landing, or bathroom silently dropped from the rooms array. Returns the
    list of newly-found room dicts (may be empty). Rooms whose number or
    name already appears in `found_rooms` are filtered out.
    """
    if call_fn is None:
        call_fn = _call_gpt4o

    listed = []
    for r in found_rooms:
        num = str(r.get("number", "")).strip()
        nm = str(r.get("name", "")).strip()
        fl = r.get("floor", 0)
        listed.append(f"  - #{num or '?'} {nm or '?'} (floor {fl})")
    listed_str = "\n".join(listed) if listed else "  (none)"

    bbox_fields = (
        ', "x_pct": 0, "y_pct": 0, "w_pct": 20, "h_pct": 20'
        if with_bbox else ""
    )
    prompt = (
        "This is a hand-drawn asbestos survey floor plan sketch. A previous "
        "reading already identified these rooms:\n"
        f"{listed_str}\n\n"
        "Look at the WHOLE page again — every separate floor sketch drawn on "
        "it, every corner, and every small box. List ONLY the enclosed rooms "
        "that ARE drawn on the sketch but are NOT already in the list above. "
        "Commonly missed: cupboards (CPD), airing cupboards, landings, "
        "bathrooms, WCs, corridors, store rooms, porches — especially small "
        "ones squeezed between larger rooms.\n"
        "If every room on the sketch is already in the list, return an empty "
        "array.\n\n"
        "Return ONLY valid JSON, no markdown:\n"
        '{"rooms": [{"name": "Cupboard", "number": "005", "has_acm": false, '
        '"acm_color": null, "has_stairs": false, "no_access": false, '
        f'"floor": 0, "measured_width_m": null, "measured_height_m": null'
        f'{bbox_fields}}}]}}'
    )

    raw = call_fn(api_key, b64, prompt, max_tokens=2000)
    parsed_data = _parse_json(raw) if raw else None
    
    # If using Gemini and it fails or returns truncated/invalid JSON, fall back to OpenAI GPT-4o
    if False:
        openai_key = os.environ.get("OPENAI_API_KEY", "")
        gemini_disabled = True
        if openai_key and not gemini_disabled:
            print("[OMITTED] Gemini failed or returned truncated JSON — falling back to GPT-4o...")
            raw = _call_gpt4o(openai_key, b64, prompt, max_tokens=2000)
            parsed_data = _parse_json(raw) if raw else None

    if not parsed_data or not parsed_data.get("rooms"):
        return []
    data = parsed_data

    existing_nums = {str(r.get("number", "")).strip() for r in found_rooms
                     if str(r.get("number", "")).strip()}
    existing_names = {str(r.get("name", "")).strip().lower() for r in found_rooms
                      if str(r.get("name", "")).strip()}
    new_rooms = []
    for r in data["rooms"]:
        num = str(r.get("number", "")).strip()
        nm = str(r.get("name", "")).strip().lower()
        # Skip anything the first pass already had (GPT often re-lists them).
        if num and num in existing_nums:
            continue
        if nm and nm in existing_names:
            continue
        if not nm and not num:
            continue
        new_rooms.append(r)
    return new_rooms



def _read_quadrants(
    api_key: str,
    sketch: np.ndarray,
    call_fn: Any = None,
    provider_name: str = "GPT4O",
    grid: Tuple[int, int] = (2, 2),
) -> List[Dict]:
    """Split sketch into a grid of overlapping tiles and read each at higher res.

    Default 2x2. For very dense plans a finer grid (e.g. 3x3) gives more
    magnification per tile so cramped numbers/names become legible.
    """
    if call_fn is None:
        call_fn = _call_gpt4o

    h, w = sketch.shape[:2]
    rows, cols = grid
    overlap_x, overlap_y = w // 10, h // 10  # 10% overlap to catch boundary rooms
    quadrants = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, (w * c) // cols - overlap_x)
            x2 = min(w, (w * (c + 1)) // cols + overlap_x)
            y1 = max(0, (h * r) // rows - overlap_y)
            y2 = min(h, (h * (r + 1)) // rows + overlap_y)
            quadrants.append((f"r{r}c{c}", sketch[y1:y2, x1:x2]))

    all_rooms = []
    for q_name, q_img in quadrants:
        # Each quadrant is ~half the sketch; encoding it at 2048 gives the AI
        # roughly double the effective resolution of the full-page read, so
        # small room labels (CPD, Store, Loby) become legible.
        b64 = _encode_sketch(q_img, max_dim=2048)
        raw = call_fn(api_key, b64, _QUADRANT_PROMPT, max_tokens=2000)
        q_data = _parse_json(raw) if raw else None
        
        # If using Gemini and it fails or returns truncated/invalid JSON, fall back to OpenAI GPT-4o
        if False:
            openai_key = os.environ.get("OPENAI_API_KEY", "")
            gemini_disabled = True
            if openai_key and not gemini_disabled:
                print(f"[QUADRANT] Gemini failed or returned truncated JSON in {q_name} — falling back to GPT-4o...")
                raw = _call_gpt4o(openai_key, b64, _QUADRANT_PROMPT, max_tokens=2000)
                q_data = _parse_json(raw) if raw else None

        if q_data and q_data.get("rooms"):
            for r in q_data["rooms"]:
                r["_quadrant"] = q_name
            all_rooms.extend(q_data["rooms"])
            print(f"[{provider_name}] Quadrant {q_name}: {len(q_data['rooms'])} rooms")
    return all_rooms



def _merge_quadrant_names(main_rooms: list, quadrant_rooms: list):
    """Merge room names from quadrants into main room list by matching numbers."""
    # Build number→name map from quadrants
    num_to_name = {}
    for qr in quadrant_rooms:
        num = str(qr.get("number", "")).strip()
        name = str(qr.get("name", "")).strip()
        if num and name and not re.fullmatch(r'\d+', name):
            num_to_name[num] = name

    # Apply to main rooms
    for room in main_rooms:
        num = str(room.get("number", "")).strip()
        current_name = str(room.get("name", "")).strip()
        # Only override if current name is just a number and we have a real name
        if num in num_to_name and re.fullmatch(r'\d+', current_name):
            room["name"] = num_to_name[num]


def _find_empty_region(rooms: list, min_side: int = 12) -> Tuple[int, int, int, int]:
    """Find a reasonably large empty rectangle in 0-100 pct space.

    Rasterises existing room bboxes on a 100x100 grid and returns
    (x_pct, y_pct, w_pct, h_pct) of the largest axis-aligned empty rectangle.
    Falls back to a centre placement if the grid is fully covered.
    """
    grid = np.zeros((100, 100), dtype=bool)
    for r in rooms:
        try:
            x = int(max(0.0, float(r.get("x_pct", 0) or 0)))
            y = int(max(0.0, float(r.get("y_pct", 0) or 0)))
            w = int(max(0.0, float(r.get("w_pct", 0) or 0)))
            h = int(max(0.0, float(r.get("h_pct", 0) or 0)))
        except (TypeError, ValueError):
            continue
        x1 = min(100, x + w)
        y1 = min(100, y + h)
        if x1 > x and y1 > y:
            grid[y:y1, x:x1] = True

    best = (0, 0, 0, 0)  # (area, x, y, side_best)
    best_area = 0
    # Grid scan: at every empty cell, try growing a square/rectangle down-right.
    for y0 in range(0, 100, 3):
        for x0 in range(0, 100, 3):
            if grid[y0, x0]:
                continue
            # Find max width starting at (x0, y0)
            x_end = x0
            while x_end < 100 and not grid[y0, x_end]:
                x_end += 1
            w = x_end - x0
            if w < min_side:
                continue
            # Grow downward while full w is clear.
            y_end = y0
            while y_end < 100 and not grid[y_end:y_end + 1, x0:x_end].any():
                y_end += 1
            h = y_end - y0
            if h < min_side:
                continue
            area = w * h
            if area > best_area:
                best_area = area
                best = (x0, y0, w, h)

    if best_area == 0:
        # Everything covered — use centre fallback.
        return 40, 40, 20, 20
    x, y, w, h = best
    # Shrink slightly so the new room doesn't sit flush against neighbours.
    pad = 1
    return (x + pad, y + pad, max(min_side, w - 2 * pad), max(min_side, h - 2 * pad))


def _bbox_size_variance(rooms: list) -> float:
    """Coefficient of variation of bbox areas. 0 = all identical, >0.3 = healthy spread."""
    areas = []
    for r in rooms:
        w = r.get("w_pct")
        h = r.get("h_pct")
        if w is None or h is None:
            continue
        try:
            areas.append(float(w) * float(h))
        except (TypeError, ValueError):
            continue
    if len(areas) < 2:
        return 1.0
    mean = sum(areas) / len(areas)
    if mean <= 0:
        return 0.0
    var = sum((a - mean) ** 2 for a in areas) / len(areas)
    return (var ** 0.5) / mean  # CV


def _bbox_coverage(rooms: list) -> float:
    """Fraction of sketch area covered by room bboxes (0.0-1.0). Overlaps counted once."""
    boxes = []
    for r in rooms:
        try:
            x = max(0.0, float(r.get("x_pct", 0))) / 100
            y = max(0.0, float(r.get("y_pct", 0))) / 100
            w = max(0.0, float(r.get("w_pct", 0))) / 100
            h = max(0.0, float(r.get("h_pct", 0))) / 100
        except (TypeError, ValueError):
            continue
        if w > 0 and h > 0:
            boxes.append((x, y, min(w, 1.0 - x), min(h, 1.0 - y)))
    if not boxes:
        return 0.0
    # Rasterise onto a 100x100 grid and count filled cells.
    grid = np.zeros((100, 100), dtype=bool)
    for x, y, w, h in boxes:
        x0, y0 = int(x * 100), int(y * 100)
        x1, y1 = int(min(100, (x + w) * 100)), int(min(100, (y + h) * 100))
        if x1 > x0 and y1 > y0:
            grid[y0:y1, x0:x1] = True
    return float(grid.sum()) / grid.size


def _full_layout_geometry_is_usable(data: Optional[Dict]) -> bool:
    """Reject missing or obviously fabricated AI layout geometry."""
    if not data or not data.get("rooms"):
        return False

    rooms = data["rooms"]
    valid = []
    valid_rooms = []
    expected_geometry_count = sum(1 for room in rooms if not room.get("is_fallback"))
    for room in rooms:
        if room.get("is_fallback"):
            continue
        try:
            x = float(room["x_pct"])
            y = float(room["y_pct"])
            w = float(room["w_pct"])
            h = float(room["h_pct"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            0 <= x < 100
            and 0 <= y < 100
            and 0 < w <= 100
            and 0 < h <= 100
            and x + w <= 102
            and y + h <= 102
        ):
            valid.append((round(x, 1), round(y, 1), round(w, 1), round(h, 1)))
            valid_rooms.append({
                "x_pct": x, "y_pct": y, "w_pct": w, "h_pct": h,
            })

    if expected_geometry_count == 0 or len(valid) != expected_geometry_count:
        return False

    if len(valid) >= 4:
        size_counts: Dict[Tuple[float, float], int] = {}
        for _, _, w, h in valid:
            size_counts[(w, h)] = size_counts.get((w, h), 0) + 1
        repeated_size_ratio = max(size_counts.values()) / len(valid)
        unique_x = len({box[0] for box in valid})
        unique_y = len({box[1] for box in valid})
        grid_axis_limit = max(2, int(np.ceil(len(valid) ** 0.5)))
        if repeated_size_ratio >= 0.60 and (
            unique_x <= grid_axis_limit or unique_y <= grid_axis_limit
        ):
            return False

    coverage = _bbox_coverage(valid_rooms)
    total_area = sum(w * h for _, _, w, h in valid) / 10000
    overlap_ratio = total_area / max(coverage, 0.001)
    excessive_overlap = len(valid) >= 4 and overlap_ratio > 1.65
    return 0.03 <= coverage <= 0.95 and not excessive_overlap


def _rooms_are_fragmented(plan: FloorPlan, gap_ratio: float = 0.30,
                          min_coverage: float = 0.45) -> bool:
    """True when a floor's rooms render scattered, so an overlay is more faithful.

    Two signals (either trips it):
    - GAP: a large dead-space gap (> gap_ratio of the span) on one axis — two
      clusters with whitespace between (e.g. N-105005, floor detection collapsed).
    - COVERAGE: a floor's rooms fill < min_coverage of their own bounding box —
      i.e. boxes float with dead space around them, no single clean gap
      (e.g. N-105325 floor 1: 6 rooms, 36% coverage). Real plans tile their
      footprint (~0.7-1.0), so low coverage = a scattered reconstruction.
    """
    by_floor: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for r in plan.rooms:
        try:
            x, y, w, h = (float(v) for v in r.bbox)
        except (TypeError, ValueError):
            continue
        by_floor.setdefault(int(r.floor_idx or 0), []).append((x, y, w, h))

    def _max_gap_ratio(intervals: List[Tuple[float, float]]) -> float:
        ivs = sorted(intervals)
        span = max(e for _, e in ivs) - min(s for s, _ in ivs)
        if span <= 0:
            return 0.0
        cur_end, gap = ivs[0][1], 0.0
        for s, e in ivs[1:]:
            if s > cur_end:
                gap = max(gap, s - cur_end)
            cur_end = max(cur_end, e)
        return gap / span

    for boxes in by_floor.values():
        if len(boxes) < 3:
            continue
        x_gap = _max_gap_ratio([(b[0], b[0] + b[2]) for b in boxes])
        y_gap = _max_gap_ratio([(b[1], b[1] + b[3]) for b in boxes])
        if max(x_gap, y_gap) > gap_ratio:
            return True
        # COVERAGE: with >=4 rooms, if they fill little of their own footprint
        # they're scattered/floating (no clean gap to catch above).
        if len(boxes) >= 4:
            fx1 = min(b[0] for b in boxes); fy1 = min(b[1] for b in boxes)
            fx2 = max(b[0] + b[2] for b in boxes); fy2 = max(b[1] + b[3] for b in boxes)
            bbox_area = max(1.0, (fx2 - fx1) * (fy2 - fy1))
            room_area = sum(b[2] * b[3] for b in boxes)
            if room_area / bbox_area < min_coverage:
                return True
    return False


def _vector_plan_geometry_is_usable(plan: FloorPlan) -> bool:
    """Reject final reconstructed geometry that cannot represent a floor plan."""
    sketch_w, sketch_h = plan.image_size
    if sketch_w <= 0 or sketch_h <= 0 or not plan.rooms:
        return False

    rooms_by_floor: Dict[int, List[Room]] = {}
    for room in plan.rooms:
        try:
            x, y, w, h = (float(v) for v in room.bbox)
        except (TypeError, ValueError):
            return False
        if (
            w <= 0 or h <= 0 or x < 0 or y < 0
            or x + w > sketch_w * 1.02
            or y + h > sketch_h * 1.02
        ):
            return False
        rooms_by_floor.setdefault(int(room.floor_idx or 0), []).append(room)

    _dbg = os.environ.get("PLAN_GATE_DEBUG", "").strip().lower() in ("1", "true", "yes")
    for fidx, rooms in rooms_by_floor.items():
        if len(rooms) >= 5:
            areas = [float(room.bbox[2]) * float(room.bbox[3]) for room in rooms]
            largest_area = max(areas)
            tiny_fragments = sum(area < largest_area * 0.03 for area in areas)
            if tiny_fragments > len(rooms) / 2:
                if _dbg:
                    print(f"[GATE] floor {fidx}: REJECT tiny_fragments={tiny_fragments}/{len(rooms)}")
                return False
        # Normalise each floor's rooms to THAT floor's own bounding region, not
        # the whole sketch — on a multi-floor sheet a single floor only occupies
        # part of the image, so whole-sketch coverage wrongly rejected it.
        fx1 = min(float(r.bbox[0]) for r in rooms)
        fy1 = min(float(r.bbox[1]) for r in rooms)
        fx2 = max(float(r.bbox[0]) + float(r.bbox[2]) for r in rooms)
        fy2 = max(float(r.bbox[1]) + float(r.bbox[3]) for r in rooms)
        fw = max(fx2 - fx1, 1.0)
        fh = max(fy2 - fy1, 1.0)
        geometry = {
            "rooms": [
                {
                    "x_pct": (float(room.bbox[0]) - fx1) / fw * 100,
                    "y_pct": (float(room.bbox[1]) - fy1) / fh * 100,
                    "w_pct": float(room.bbox[2]) / fw * 100,
                    "h_pct": float(room.bbox[3]) / fh * 100,
                }
                for room in rooms
            ],
        }
        # A floor with only 1-2 rooms (e.g. a loft) can't be assessed for
        # layout plausibility — the bounds check above is enough; skip the
        # multi-room fabrication/coverage heuristics that need >=3 rooms.
        if len(rooms) < 3:
            # Can't assess layout of 1-2 rooms, but still reject microscopic
            # detections (a real loft/external area covers a meaningful share).
            sketch_cov = sum(float(r.bbox[2]) * float(r.bbox[3]) for r in rooms) / (sketch_w * sketch_h)
            if sketch_cov < 0.005:
                if _dbg:
                    print(f"[GATE] floor {fidx}: REJECT — {len(rooms)} room(s) cover {sketch_cov:.4f} of sketch")
                return False
            if _dbg:
                print(f"[GATE] floor {fidx}: rooms={len(rooms)} sketch_cov={sketch_cov:.3f} -> skip sublayout check")
            continue
        ok = _floor_sublayout_is_usable(geometry["rooms"])
        if _dbg:
            cov = _bbox_coverage(geometry["rooms"])
            print(f"[GATE] floor {fidx}: rooms={len(rooms)} coverage={cov:.3f} ok={ok}")
        if not ok:
            return False
    return True


def _floor_sublayout_is_usable(rooms: List[Dict]) -> bool:
    """Plausibility of ONE floor's rooms, normalised to that floor's region.

    Rooms are already expressed as percentages of the floor's own bounding box
    (so they fill ~0-100%). We therefore only reject a fabricated grid (many
    identical sizes on a single axis) and microscopic coverage — NOT high
    coverage, because a real tightly-packed floor legitimately fills its own
    footprint.
    """
    valid = [(round(r["x_pct"], 1), round(r["y_pct"], 1), round(r["w_pct"], 1), round(r["h_pct"], 1))
             for r in rooms]
    if len(valid) >= 4:
        size_counts: Dict[Tuple[float, float], int] = {}
        for _, _, w, h in valid:
            size_counts[(w, h)] = size_counts.get((w, h), 0) + 1
        repeated_size_ratio = max(size_counts.values()) / len(valid)
        unique_x = len({b[0] for b in valid})
        unique_y = len({b[1] for b in valid})
        grid_axis_limit = max(2, int(np.ceil(len(valid) ** 0.5)))
        if repeated_size_ratio >= 0.70 and (unique_x <= grid_axis_limit and unique_y <= grid_axis_limit):
            return False
    coverage = _bbox_coverage(rooms)
    return coverage >= 0.03


def _append_geometry_safe_rooms(data: Dict, candidates: List[Dict]) -> int:
    """Append candidates only while the complete layout remains trustworthy."""
    rooms = data.setdefault("rooms", [])
    added = 0
    for room in candidates:
        proposed = dict(data)
        proposed["rooms"] = [*rooms, room]
        if _full_layout_geometry_is_usable(proposed):
            rooms.append(room)
            added += 1
        else:
            print(f"[AI] Rejected recovered room with unsafe geometry: "
                  f"{room.get('name')} #{room.get('number')}")
    return added


def get_room_labels(sketch: np.ndarray, labels_only: bool = False) -> Optional[Dict]:
    """
    Get room labels from GPT-4o vision, with validation retries.

    When using the full layout prompt we also validate:
      - bbox sizes vary (CV >= 0.20) — else retry once
      - total coverage >= 40% of sketch — else retry once with missed-rooms hint
      - samples list is present — if empty and many rooms, retry asking
        explicitly for red-pen sample annotations

    Args:
        labels_only: If True, use labels-only prompt (model has geometry).
                     If False, use full layout prompt (need AI for positions too).
    """
    prompt = _LABELS_ONLY_PROMPT if labels_only else _FULL_LAYOUT_PROMPT
    result = get_room_labels_gpt4o(sketch, prompt=prompt)
    if not result or not result.get("rooms"):
        print("[AI] GPT-4o failed")
        return None

    # Validation retries only apply to full layout (labels-only has no bboxes).
    if not labels_only:
        rooms = result.get("rooms") or []
        retried = False

        # 1. Uniform-size check (raised threshold — real sketches vary a lot).
        cv = _bbox_size_variance(rooms)
        if len(rooms) >= 3 and cv < 0.30:
            print(f"[AI] Room sizes too uniform (CV={cv:.2f}) — retrying with proportion reminder")
            hint_prompt = prompt + (
                "\n\nYour previous response gave rooms similar sizes "
                f"(size coefficient of variation = {cv:.2f}, which is too "
                "low). The rooms on this sketch are drawn at clearly "
                "different sizes. Return bounding boxes where the largest "
                "room is AT LEAST 2x the area of the smallest. A Landing "
                "or corridor is narrower than a bedroom. A WC or cupboard "
                "is smaller than a bedroom. A Loft may be wider than a "
                "single bedroom. Look at the pixel extents the surveyor "
                "drew and copy those proportions."
            )
            retry = get_room_labels_gpt4o(sketch, prompt=hint_prompt)
            if retry and retry.get("rooms"):
                new_cv = _bbox_size_variance(retry["rooms"])
                print(f"[AI] Retry CV={new_cv:.2f} (was {cv:.2f})")
                if new_cv > cv:
                    result = retry
                    rooms = result.get("rooms") or []
            retried = True

        # 2. Coverage check.
        cov = _bbox_coverage(rooms)
        if cov < 0.40 and not retried:
            print(f"[AI] Low coverage ({cov*100:.0f}% of sketch) — retrying to find missed rooms")
            hint_prompt = prompt + (
                f"\n\nYour previous response identified rooms covering only "
                f"{cov*100:.0f}% of the sketch. Look again for rooms you "
                "missed, especially in corners, edges, and areas with "
                "hatching, X marks, or dense annotations. Include every "
                "enclosed space."
            )
            retry = get_room_labels_gpt4o(sketch, prompt=hint_prompt)
            if retry and retry.get("rooms") and len(retry["rooms"]) > len(rooms):
                result = retry
                rooms = result.get("rooms") or []
                print(f"[AI] Retry found {len(rooms)} rooms (was {len(rooms) - (len(rooms)-len(retry['rooms']))})")

    # Sample-detection retry (both modes).
    samples = result.get("samples") or []
    n_rooms = len(result.get("rooms") or [])
    if not samples and n_rooms >= 3:
        print("[AI] No samples returned but sketch has rooms — retrying with sample hint")
        sample_hint = (
            "Look carefully for RED-PEN annotations on the sketch — these "
            "are asbestos SAMPLE labels like 'S01 FT', 'S02 Mastic', "
            "'S003 TC', 'P001 TC', 'Ref S004'. Each sample has an id, a "
            "material, and a position. They are written in red ink, often "
            "with arrows pointing into rooms. Return them in the samples "
            "array. Return ONLY the samples JSON: "
            '{"samples": [{"id":"S01","material":"FT","acm_positive":false,'
            '"is_ref":false,"x_pct":30,"y_pct":50}]}'
        )
        retry = get_room_labels_gpt4o(sketch, prompt=sample_hint)
        if retry and retry.get("samples"):
            result["samples"] = retry["samples"]
            print(f"[AI] Retry found {len(retry['samples'])} samples")

    return result


# ============================================================================
# Step 4: Merge Model Geometry + AI Labels
# ============================================================================

def _find_detached_room_box(rooms: List[Room]) -> Optional[int]:
    if len(rooms) < 3:
        return None

    centroids = []
    for r in rooms:
        cx = r.bbox[0] + r.bbox[2] / 2
        cy = r.bbox[1] + r.bbox[3] / 2
        centroids.append((cx, cy))

    nearest_dists = []
    for i, c1 in enumerate(centroids):
        min_d = float('inf')
        for j, c2 in enumerate(centroids):
            if i == j:
                continue
            d = ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5
            if d < min_d:
                min_d = d
        nearest_dists.append((i, min_d))

    sorted_dists = sorted(nearest_dists, key=lambda x: x[1])
    median_near = sorted_dists[len(sorted_dists) // 2][1]

    candidate_idx, max_d = sorted_dists[-1]
    min_x = min(r.bbox[0] for r in rooms)
    min_y = min(r.bbox[1] for r in rooms)
    max_x = max(r.bbox[0] + r.bbox[2] for r in rooms)
    max_y = max(r.bbox[1] + r.bbox[3] for r in rooms)
    layout_diagonal = ((max_x - min_x) ** 2 + (max_y - min_y) ** 2) ** 0.5
    if max_d > max(layout_diagonal * 0.22, median_near * 2.3):
        print(f"[MERGE-DETACHED] Identified detached room box index {candidate_idx} (nearest_dist={max_d:.1f}px, median_near={median_near:.1f}px)")
        return candidate_idx

    return None


def _drop_spurious_isolated_rooms(rooms: List[Room], sketch_w: int, sketch_h: int) -> List[Room]:
    """Remove clearly-spurious room boxes that float detached from the plan.

    Real plan rooms share walls, so their bounding boxes touch (within a small
    margin). A spurious detection (e.g. a thin sliver over a sample arrow) sits
    in empty space touching nothing. We drop a room ONLY when it is BOTH
    isolated (touches no other room) AND tiny/thin — and never a loft/external,
    which are legitimately separate. Conservative by design: when in doubt, keep.
    """
    if len(rooms) < 3:
        return rooms
    margin = 0.03 * max(sketch_w, sketch_h)
    areas = sorted(r.bbox[2] * r.bbox[3] for r in rooms)
    med_area = areas[len(areas) // 2]
    mean_min_dim = sum(min(r.bbox[2], r.bbox[3]) for r in rooms) / len(rooms)

    kept = []
    for i, r in enumerate(rooms):
        x, y, w, h = r.bbox
        label = str(r.label or "").lower()
        is_special = any(k in label for k in ("loft", "attic", "external", "roof"))
        touches = any(
            i != j
            and x - margin < o.bbox[0] + o.bbox[2] and x + w + margin > o.bbox[0]
            and y - margin < o.bbox[1] + o.bbox[3] and y + h + margin > o.bbox[1]
            for j, o in enumerate(rooms)
        )
        tiny = (w * h) < 0.30 * med_area
        thin = min(w, h) < 0.45 * mean_min_dim
        if (not touches) and (tiny or thin) and not is_special:
            print(f"[POST] Dropped spurious isolated room '{r.label}' #{r.number} "
                  f"(bbox={r.bbox}; touches nothing, {'tiny' if tiny else 'thin'})")
            continue
        kept.append(r)
    return kept


def merge_results(
    model_rooms: List[Room],
    gpt4o_data: Optional[Dict],
    sketch_h: int,
    sketch_w: int,
    seg_mask: Optional[np.ndarray] = None,
) -> FloorPlan:
    """
    Merge ResNet room geometries with AI room labels.

    Strategy: MODEL provides room geometry (boundaries, positions, areas).
              AI provides room labels (names, numbers, ACM status).
              When model can't separate rooms, use AI labels + model room mask
              boundary to create proportional layout.
              Match labels to rooms by spatial ordering (top-left → bottom-right).
    """
    # Filter out wall fixtures, panel boards, and meters falsely parsed as rooms
    if gpt4o_data and gpt4o_data.get("rooms"):
        filtered_rooms = []
        for gr in gpt4o_data["rooms"]:
            name = str(gr.get("name", "")).strip().lower()
            # Ignore electrical distribution boards, fuse boxes, boilers, meters as rooms
            is_fixture = any(term in name for term in [
                "dis board", "fuse board", "distribution board", "db", "dis. board", 
                "meter", "fuse box", "boiler", "cylinder", "electrical", "elec"
            ]) or ("board" in name and "cupboard" not in name)
            # Keep plant room or waiting area or office if it contains these but is a real room
            if is_fixture and not any(ok in name for ok in ["room", "area", "office", "shop"]):
                print(f"[MERGE] Filtering out wall fixture parsed as room: {gr.get('name')}")
                continue
            filtered_rooms.append(gr)
        gpt4o_data["rooms"] = filtered_rooms

    plan = FloorPlan(image_size=(sketch_w, sketch_h))

    if gpt4o_data:
        plan.floor_title = gpt4o_data.get("floor_title", "Ground Floor")

    # Decide: use MODEL geometry or AI layout?
    ai_room_count = len(gpt4o_data.get("rooms", [])) if gpt4o_data else 0
    model_room_count = len(model_rooms)
    ai_geometry_usable = _full_layout_geometry_is_usable(gpt4o_data)
    ai_floor_indices = {
        _safe_int_floor(room.get("floor"))
        for room in (gpt4o_data.get("rooms", []) if gpt4o_data else [])
        if room.get("floor") is not None
    }
    has_explicit_loft = any(
        "loft" in str(room.get("name") or "").lower()
        or "attic" in str(room.get("name") or "").lower()
        for room in (gpt4o_data.get("rooms", []) if gpt4o_data else [])
    )
    # Only fall back to AI free-hand geometry for multi-floor sketches when the
    # detector is WEAK. When YOLO has solid coverage we keep its accurate boxes
    # and inherit the per-floor assignment from the matched AI labels (the loft
    # is still split onto its own page downstream). Discarding strong YOLO
    # geometry here was the cause of fragmented, misaligned multi-floor plans.
    model_geometry_is_strong = (
        model_room_count >= 3 and model_room_count >= ai_room_count * 0.4
    )
    prefer_ai_multifloor_geometry = (
        ai_geometry_usable
        and has_explicit_loft
        and len(ai_floor_indices) >= 2
        and ai_room_count >= 3
        and not model_geometry_is_strong
    )

    # If model found significantly fewer rooms than AI, trust AI layout
    # (model walls weren't strong enough to separate rooms)
    use_model_geometry = (
        bool(model_rooms) and not prefer_ai_multifloor_geometry and (
            not ai_geometry_usable
            or (
                model_room_count >= 3
                and model_room_count >= ai_room_count * 0.4
            )
        )
    )
    if prefer_ai_multifloor_geometry and model_rooms:
        print("[MERGE] Using validated AI multi-floor geometry; detector boxes conflict with explicit floor separation")

    if use_model_geometry and model_rooms:
        working_rooms = list(model_rooms)
        # Always sort by area descending so large rooms match major labels first
        working_rooms.sort(key=lambda r: r.area, reverse=True)
        if ai_room_count >= 3 and model_room_count > ai_room_count * 1.3:
            target = max(ai_room_count, int(ai_room_count * 1.2))
            working_rooms = working_rooms[:target]
            print(f"[MERGE] Trimmed {model_room_count} model rooms to {len(working_rooms)} "
                  f"(AI found {ai_room_count})")

        print(f"[MERGE] Using MODEL geometry ({len(working_rooms)} rooms) + AI labels ({ai_room_count})")

        # Build AI label list with centroids from position descriptions
        gpt_labels = []
        if gpt4o_data:
            for gr in gpt4o_data.get("rooms", []):
                pos = str(gr.get("position", "center")).strip().lower()
                # Convert position string to approximate centroid
                pos_map = {
                    'top-left': (0.17, 0.17), 'top-center': (0.5, 0.17), 'top-right': (0.83, 0.17),
                    'center-left': (0.17, 0.5), 'center': (0.5, 0.5), 'center-right': (0.83, 0.5),
                    'bottom-left': (0.17, 0.83), 'bottom-center': (0.5, 0.83), 'bottom-right': (0.83, 0.83),
                }
                fx, fy = pos_map.get(pos, (0.5, 0.5))
                # Also use bounding box percentages if available (more precise)
                if gr.get("x_pct") is not None and gr.get("y_pct") is not None:
                    fx = (float(gr["x_pct"]) + float(gr.get("w_pct", 20)) / 2) / 100
                    fy = (float(gr["y_pct"]) + float(gr.get("h_pct", 20)) / 2) / 100
                gpt_labels.append({
                    'name': str(gr.get("name", "")).strip(),
                    'number': _clean_number(gr.get("number")),
                    'has_acm': bool(gr.get("has_acm", False)),
                    'acm_color': gr.get("acm_color"),
                    'has_stairs': bool(gr.get("has_stairs", False)),
                    'stairs_bbox': _optional_pct_bbox(gr, "stairs", sketch_w, sketch_h),
                    'no_access': bool(gr.get("no_access", False)),
                    'floor_idx': _safe_int_floor(gr.get("floor", 0)),
                    'measured_w': gr.get("measured_width_m"),
                    'measured_h': gr.get("measured_height_m"),
                    'cx': fx * sketch_w,
                    'cy': fy * sketch_h,
                    'is_fallback': bool(gr.get("is_fallback", False)),
                    'bbox_pct': (
                        gr.get("x_pct"), gr.get("y_pct"),
                        gr.get("w_pct"), gr.get("h_pct"),
                    ) if all(gr.get(k) is not None for k in
                             ("x_pct", "y_pct", "w_pct", "h_pct")) else None,
                })

            # Check if we have a Loft/Attic label and a detached room box
            loft_label_idx = next((idx for idx, gl in enumerate(gpt_labels)
                                   if "loft" in gl['name'].lower() or "attic" in gl['name'].lower()), -1)
            if loft_label_idx != -1:
                detached_room_idx = _find_detached_room_box(working_rooms)
                if detached_room_idx is not None:
                    mr = working_rooms[detached_room_idx]
                    gl = gpt_labels[loft_label_idx]
                    # Align centroids to force matching
                    gl['cx'] = mr.bbox[0] + mr.bbox[2] / 2
                    gl['cy'] = mr.bbox[1] + mr.bbox[3] / 2
                    print(f"[MERGE-DETACHED] Aligned Loft label '{gl['name']}' centroid to match detached room box at {gl['cx']:.1f}, {gl['cy']:.1f}")

        # Match AI labels to model rooms using Scipy's Hungarian algorithm (linear_sum_assignment)
        # to find the globally optimal centroid-distance matching.
        from scipy.optimize import linear_sum_assignment
        
        # Partition gpt_labels into real_labels (with actual coordinates) and fallback_labels
        # (reconstructed missing rooms placed at arbitrary coordinates). This prevents
        # arbitrary fallback coordinates from distorting the spatial normalization bounds.
        real_labels = []
        fallback_labels = []
        for j, gl in enumerate(gpt_labels):
            gl_copy = dict(gl)
            gl_copy['orig_idx'] = j
            if gl.get('is_fallback', False):
                fallback_labels.append(gl_copy)
            else:
                real_labels.append(gl_copy)

        # Group AI labels by floor index to pre-assign floors to model rooms
        floor_centers = {}
        for gl in real_labels:
            f = gl['floor_idx']
            floor_centers.setdefault(f, []).append((gl['cx'], gl['cy']))
        
        floor_avgs = {}
        for f, pts in floor_centers.items():
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            floor_avgs[f] = (sum(xs) / len(xs), sum(ys) / len(ys))

        # Compute bounds for spatial normalization using only real labels
        min_model_x, max_model_x = float('inf'), float('-inf')
        min_model_y, max_model_y = float('inf'), float('-inf')
        for mr in working_rooms:
            mx = mr.bbox[0] + mr.bbox[2] / 2
            my = mr.bbox[1] + mr.bbox[3] / 2
            min_model_x = min(min_model_x, mx)
            max_model_x = max(max_model_x, mx)
            min_model_y = min(min_model_y, my)
            max_model_y = max(max_model_y, my)
            
        min_ai_x, max_ai_x = float('inf'), float('-inf')
        min_ai_y, max_ai_y = float('inf'), float('-inf')
        for gl in real_labels:
            min_ai_x = min(min_ai_x, gl['cx'])
            max_ai_x = max(max_ai_x, gl['cx'])
            min_ai_y = min(min_ai_y, gl['cy'])
            max_ai_y = max(max_ai_y, gl['cy'])
            
        dx_model = max_model_x - min_model_x
        dy_model = max_model_y - min_model_y
        dx_ai = max_ai_x - min_ai_x
        dy_ai = max_ai_y - min_ai_y
        
        num_rooms = len(working_rooms)
        num_real_labels = len(real_labels)
        
        # We only normalize if bounds are valid to prevent division by zero
        normalize = (
            num_rooms >= 2 and num_real_labels >= 2 and
            dx_model > sketch_w * 0.10 and dy_model > sketch_h * 0.10 and
            dx_ai > sketch_w * 0.10 and dy_ai > sketch_h * 0.10
        )

        # Build cost matrix against real labels
        cost_matrix = np.zeros((num_rooms, num_real_labels))
        
        for i, mr in enumerate(working_rooms):
            mx = mr.bbox[0] + mr.bbox[2] // 2
            my = mr.bbox[1] + mr.bbox[3] // 2
            
            mr_floor = None
            if len(floor_avgs) > 1:
                mr_floor = min(floor_avgs.keys(), key=lambda f: (mx - floor_avgs[f][0])**2 + (my - floor_avgs[f][1])**2)
                
            # Normalized model coordinates
            if normalize:
                mx_val = (mx - min_model_x) / dx_model
                my_val = (my - min_model_y) / dy_model
            else:
                mx_val = mx
                my_val = my
                
            for j, gl in enumerate(real_labels):
                if normalize:
                    cx_val = (gl['cx'] - min_ai_x) / dx_ai
                    cy_val = (gl['cy'] - min_ai_y) / dy_ai
                else:
                    cx_val = gl['cx']
                    cy_val = gl['cy']
                    
                dist = ((mx_val - cx_val) ** 2 + (my_val - cy_val) ** 2) ** 0.5
                cost = dist
                if mr_floor is not None and gl['floor_idx'] != mr_floor:
                    # Enforce floor penalty. In normalized space (costs < 1.5), 1000 is a very strong penalty
                    cost += 1000.0 if normalize else 1e6
                cost_matrix[i, j] = cost

        # Solve linear assignment for real labels
        matching = {}
        if num_rooms > 0 and num_real_labels > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matching = {r: real_labels[c]['orig_idx'] for r, c in zip(row_ind, col_ind)}
            
            # Semantic swap correction: If a cupboard and a corridor/lobby/larger room are matched,
            # but the cupboard got assigned to the larger room box and the lobby to the smaller room box,
            # swap their label assignments to preserve correct semantic layout.
            def _is_cupboard(name):
                n = str(name).strip().lower()
                return any(w in n for w in ["cpd", "cupboard", "store", "airing", "ac", "closet", "pant", "boiler"])

            def _is_lobby_or_room(name):
                n = str(name).strip().lower()
                return any(w in n for w in ["lobby", "hall", "corridor", "room", "bed", "kitchen", "bath", "wc", "toilet", "living", "lounge", "dining", "sitting", "office"])

            changed = True
            swap_count = 0
            while changed:
                changed = False
                swap_count += 1
                if swap_count > 50:
                    print("[RECONCILE] WARNING: Exceeded max semantic swaps (50). Breaking swap loop to prevent hang.")
                    break
                for i in list(matching.keys()):
                    for k in list(matching.keys()):
                        if i == k:
                            continue
                        if working_rooms[i].area < working_rooms[k].area:
                            continue
                        
                        label_i = gpt_labels[matching[i]]
                        label_k = gpt_labels[matching[k]]
                        
                        if label_i['floor_idx'] != label_k['floor_idx']:
                            continue
                        
                        name_i = label_i['name']
                        name_k = label_k['name']
                        
                        is_cpd_i = _is_cupboard(name_i)
                        is_cpd_k = _is_cupboard(name_k)
                        
                        is_lobby_i = _is_lobby_or_room(name_i) and not is_cpd_i
                        is_lobby_k = _is_lobby_or_room(name_k) and not is_cpd_k
                        
                        if is_cpd_i and is_lobby_k:
                            mx_i = working_rooms[i].bbox[0] + working_rooms[i].bbox[2] // 2
                            my_i = working_rooms[i].bbox[1] + working_rooms[i].bbox[3] // 2
                            mx_k = working_rooms[k].bbox[0] + working_rooms[k].bbox[2] // 2
                            my_k = working_rooms[k].bbox[1] + working_rooms[k].bbox[3] // 2
                            
                            if normalize:
                                n_xi = (mx_i - min_model_x) / dx_model
                                n_yi = (my_i - min_model_y) / dy_model
                                n_xk = (mx_k - min_model_x) / dx_model
                                n_yk = (my_k - min_model_y) / dy_model
                            else:
                                n_xi, n_yi = mx_i, my_i
                                n_xk, n_yk = mx_k, my_k
                                
                            dist = ((n_xi - n_xk)**2 + (n_yi - n_yk)**2)**0.5
                            dist_thresh = 0.35 if normalize else 0.35 * max(sketch_w, sketch_h)
                            
                            if dist < dist_thresh:
                                print(f"[RECONCILE] Swapping semantically swapped labels: "
                                      f"Room {i} (larger, matched to {name_i}) <-> Room {k} (smaller, matched to {name_k})")
                                matching[i], matching[k] = matching[k], matching[i]
                                changed = True
                                break
                    if changed:
                        break

            
        # Track which labels were matched
        label_used = [False] * len(gpt_labels)
        for col in matching.values():
            label_used[col] = True
        
        # Build final room list
        for i, mr in enumerate(working_rooms):
            mx = mr.bbox[0] + mr.bbox[2] // 2
            my = mr.bbox[1] + mr.bbox[3] // 2

            room = Room(
                bbox=mr.bbox,
                area=mr.area,
                contour=mr.contour,
                has_acm=mr.has_acm,
                acm_color=getattr(mr, 'acm_color', None),
                room_type='acm' if mr.has_acm else 'clear',
                floor=plan.floor_title,
                has_stairs=getattr(mr, 'has_stairs', False),
                geometry_source=getattr(mr, 'geometry_source', 'model'),
                detection_confidence=getattr(mr, 'detection_confidence', None),
            )

            best_idx = matching.get(i, -1)
            if best_idx >= 0:
                gl = gpt_labels[best_idx]
                room.label = gl['name'] or f"Room {len(plan.rooms) + 1}"
                room.number = gl['number'] or f"{len(plan.rooms) + 1:03d}"
                if gl['has_acm']:
                    room.has_acm = True
                    room.acm_color = gl.get('acm_color') or room.acm_color
                    room.room_type = 'acm'
                if gl['no_access']:
                    room.no_access = True
                    room.room_type = 'no_access'
                room.has_stairs = room.has_stairs or gl['has_stairs']
                if gl.get('stairs_bbox'):
                    room.stairs_bbox = gl['stairs_bbox']
                room.floor_idx = gl['floor_idx']
                mw, mh = gl.get('measured_w'), gl.get('measured_h')
                if mw is not None or mh is not None:
                    room.measured_width_m = float(mw) if mw is not None else None
                    room.measured_height_m = float(mh) if mh is not None else None
                    room.dimension_source = "measured"
            else:
                room.label = f"Room {len(plan.rooms) + 1}"
                room.number = f"{len(plan.rooms) + 1:03d}"
                mr_floor = None
                if len(floor_avgs) > 1:
                    mr_floor = min(floor_avgs.keys(), key=lambda f: (mx - floor_avgs[f][0])**2 + (my - floor_avgs[f][1])**2)
                if mr_floor is not None:
                    room.floor_idx = mr_floor
                elif gpt_labels:
                    room.floor_idx = gpt_labels[0]['floor_idx']
                else:
                    room.floor_idx = 0

            plan.rooms.append(room)

        # The trained YOLO model under-detects: small cupboards, landings,
        # and rooms with heavy hatching often yield no clean box. GPT-4o
        # read those labels, but with no model geometry to attach to they
        # would be silently dropped here — the merge only ever TRIMMED
        # surplus model rooms, it never ADDED rooms the model missed.
        # Add unused AI labels only when the full-layout pass supplied its
        # own usable geometry. Labels-only and fallback results remain
        # unresolved rather than becoming fabricated room rectangles.
        unused = [
            gpt_labels[j] for j in range(len(gpt_labels))
            if not label_used[j]
            and gpt_labels[j]['name']
            and not gpt_labels[j]['is_fallback']
            and gpt_labels[j]['bbox_pct'] is not None
        ]
        if unused:
            print(f"[MERGE] {len(unused)} room(s) read by AI had no model box "
                  f"— adding their AI-provided geometry: "
                  f"{[g['name'] for g in unused]}")
            for gl in unused:
                x, y, width, height = compute_normalized_bbox(
                    *gl['bbox_pct'], sketch_w, sketch_h
                )
                mw, mh = gl.get('measured_w'), gl.get('measured_h')
                rt = ('no_access' if gl['no_access']
                      else ('acm' if gl['has_acm'] else 'clear'))
                plan.rooms.append(Room(
                    bbox=(x, y, width, height), area=width * height,
                    label=gl['name'],
                    number=gl['number'] or f"{len(plan.rooms) + 1:03d}",
                    has_acm=gl['has_acm'],
                    acm_color=gl.get('acm_color'),
                    room_type=rt,
                    has_stairs=gl['has_stairs'],
                    stairs_bbox=gl.get('stairs_bbox'),
                    no_access=gl['no_access'],
                    floor=plan.floor_title,
                    floor_idx=gl['floor_idx'],
                    measured_width_m=float(mw) if mw is not None else None,
                    measured_height_m=float(mh) if mh is not None else None,
                    dimension_source="measured" if (mw or mh) else "estimated",
                    geometry_source="ai_bbox",
                ))

    elif gpt4o_data:
        # No model rooms — use AI labels + model room mask for smart layout
        print(f"[MERGE] Using AI layout ({ai_room_count} rooms) — model couldn't separate rooms")
        plan.floor_title = gpt4o_data.get("floor_title", "Ground Floor")
        gpt_rooms = gpt4o_data.get("rooms", [])

        # Find building boundary from model's room mask (class 2 + 5)
        bld_x, bld_y, bld_w, bld_h = 0, 0, sketch_w, sketch_h
        if seg_mask is not None:
            room_pixels = ((seg_mask == 2) | (seg_mask == 5)).astype(np.uint8) * 255
            if np.sum(room_pixels > 0) > 1000:
                coords = cv2.findNonZero(room_pixels)
                if coords is not None:
                    bld_x, bld_y, bld_w, bld_h = cv2.boundingRect(coords)
                    # Add small padding
                    pad = int(max(bld_w, bld_h) * 0.02)
                    bld_x = max(0, bld_x - pad)
                    bld_y = max(0, bld_y - pad)
                    bld_w = min(sketch_w - bld_x, bld_w + 2 * pad)
                    bld_h = min(sketch_h - bld_y, bld_h + 2 * pad)
                    print(f"[MERGE] Building boundary: ({bld_x},{bld_y}) {bld_w}x{bld_h}")

        # Check if AI provided usable bounding box percentages.
        # The layout-quality gate rejects missing and artificial geometry.
        has_bbox = _full_layout_geometry_is_usable(gpt4o_data)

        if has_bbox:
            if DEBUG_MODE:
                print('[DEBUG] Processing AI-provided bounding boxes')
            # AI gave proportional sizes — map to FULL sketch (AI prompt says 0-100 = entire sketch)
            for i, gr in enumerate(gpt_rooms):
                if gr.get("is_fallback"):
                    print(f"[MERGE] Skipping fallback room without verified geometry: {gr.get('name')}")
                    continue
                x, y, w, h = compute_normalized_bbox(
                    gr.get("x_pct"), gr.get("y_pct"),
                    gr.get("w_pct"), gr.get("h_pct"),
                    sketch_w, sketch_h,
                )

                na = bool(gr.get("no_access", False))
                acm = bool(gr.get("has_acm", False))
                rt = 'no_access' if na else ('acm' if acm else 'clear')
                mw = gr.get("measured_width_m")
                mh = gr.get("measured_height_m")
                room = Room(
                    bbox=(x, y, w, h), area=w * h,
                    label=str(gr.get("name", f"Room {i + 1}") or f"Room {i + 1}").strip(),
                    number=_clean_number(gr.get("number"), f"{i + 1:03d}"),
                    has_acm=acm,
                    acm_color=gr.get("acm_color"),
                    room_type=rt,
                    has_stairs=bool(gr.get("has_stairs", False)),
                    stairs_bbox=_optional_pct_bbox(gr, "stairs", sketch_w, sketch_h),
                    no_access=na,
                    floor=plan.floor_title,
                    floor_idx=_safe_int_floor(gr.get("floor", 0)),
                    measured_width_m=float(mw) if mw is not None else None,
                    measured_height_m=float(mh) if mh is not None else None,
                    dimension_source="measured" if (mw or mh) else "estimated",
                    geometry_source="ai_bbox",
                )
                plan.rooms.append(room)
        else:
            print("[MERGE] AI returned labels but no trustworthy geometry; "
                  "refusing to invent a grid layout")
    else:
        print("[MERGE] No model rooms and no GPT-4o — cannot generate plan")

    # Shared GPT-4o pass-through: samples + page/floor metadata. Runs for
    # both MODEL-PRIMARY and AI-PRIMARY branches so we never drop samples
    # just because the model had room geometry.
    if gpt4o_data:
        for s in gpt4o_data.get("samples") or []:
            try:
                sx = float(s.get("x_pct", 0) or 0)
                sy = float(s.get("y_pct", 0) or 0)
            except (TypeError, ValueError):
                sx, sy = 0.0, 0.0
            trn = s.get("target_room_number")
            trn = str(trn).strip() if trn else None
            try:
                tfi = int(s.get("target_floor", 0) or 0)
            except (TypeError, ValueError):
                tfi = 0
            plan.samples.append(Sample(
                id=_normalize_sample_id(s.get("id", "")),
                material=_normalize_sample_material(s.get("material", "")),
                x_pct=sx,
                y_pct=sy,
                acm_positive=bool(s.get("acm_positive", False)),
                is_ref=bool(s.get("is_ref", False)),
                target_room_number=trn,
                target_floor_idx=tfi,
            ))
        try:
            plan.page = int(gpt4o_data["page"]) if gpt4o_data.get("page") else None
        except (TypeError, ValueError):
            plan.page = None
        try:
            plan.total_pages = int(gpt4o_data["total_pages"]) if gpt4o_data.get("total_pages") else None
        except (TypeError, ValueError):
            plan.total_pages = None
        if gpt4o_data.get("floor_name"):
            plan.floor_title = str(gpt4o_data["floor_name"]).strip()

    # Propagation: Update room has_acm/acm_color/room_type based on the targeted samples.
    room_samples = {}
    for s in plan.samples:
        if s.target_room_number:
            num_clean = str(s.target_room_number).strip().lstrip('0') or '0'
            room_samples.setdefault((s.target_floor_idx, num_clean), []).append(s)

    for r in plan.rooms:
        r_num_clean = str(r.number or "").strip().lstrip('0') or '0'
        key = (r.floor_idx, r_num_clean)
        if key in room_samples:
            samples = room_samples[key]
            any_pos = any(s.acm_positive for s in samples)
            if any_pos:
                r.has_acm = True
                r.acm_color = 'red'
                r.room_type = 'acm'
                print(f"[PROPAGATE] Room '{r.label}' #{r.number} marked ACM positive because of positive sample(s)")
            elif not r.has_acm:
                # All targeted samples negative AND the room wasn't already
                # flagged ACM by the model/geometry -> leave it clear.
                r.acm_color = None
                r.room_type = 'clear'
            else:
                # SAFETY: do NOT clear a room the model already flagged ACM just
                # because its samples read negative — a false negative on an
                # asbestos plan is unsafe. Keep the ACM flag for human review.
                print(f"[PROPAGATE] Room '{r.label}' #{r.number} kept ACM (model-flagged) "
                      f"despite negative sample(s) — not auto-downgraded")

    # Validate room bboxes — fix any negative dimensions
    for room in plan.rooms:
        x, y, w, h = room.bbox
        w = max(10, w)
        h = max(10, h)
        room.bbox = (x, y, w, h)
        room.area = w * h

    # Snap shared walls
    _snap_shared_walls(plan.rooms, sketch_w, sketch_h)

    # POST-PROCESS 1: Drop any standalone "Stairs"/"Staircase" rooms — stairs
    # are a symbol inside a landing/hallway, never their own room. If one
    # slipped through, merge its has_stairs flag into the nearest room and
    # drop it. This fixes GPT-4o emitting "002 Stairs" as a phantom room.
    dropped = []
    keep = []
    for room in plan.rooms:
        label_l = (room.label or '').strip().lower()
        if label_l in ('stairs', 'staircase', 'stair'):
            dropped.append(room)
            continue
        keep.append(room)
    if dropped and keep:
        for drop in dropped:
            dx, dy, dw, dh = drop.bbox
            dcx, dcy = dx + dw // 2, dy + dh // 2
            # Find nearest surviving room by centre distance
            nearest = min(keep, key=lambda r: (
                (r.bbox[0] + r.bbox[2] // 2 - dcx) ** 2 +
                (r.bbox[1] + r.bbox[3] // 2 - dcy) ** 2
            ))
            nearest.has_stairs = True
            print(f"[POST] Dropped phantom '{drop.label}' room, moved stairs flag to '{nearest.label}'")
    plan.rooms = keep

    # POST-PROCESS 1.5: Merge DUPLICATE rooms. The labels-only pass, the
    # full-layout pass and the missing-room recovery each read the sketch
    # independently, so one physical room can enter the plan twice under
    # slightly different names/numbers ("Cupboard" vs "Cupboard 002", or
    # "Living Room" listed twice). POST-PROCESS 2/3 only renumber/relabel
    # duplicates — they never REMOVE one. Here we fold same-floor rooms
    # that are the same physical space into a single room, keeping the
    # best geometry and OR-ing the asbestos/stairs/access flags.
    def _num_norm(n):
        c = _clean_number(n)
        return (c.lstrip('0') or '0') if c else ''

    def _same_room(a, b):
        # A merge requires geometric overlap AND that the names do not
        # clearly disagree. Two rooms that merely share a number but sit
        # apart are a numbering MISTAKE on distinct rooms (POST-PROCESS 2
        # renumbers them). And two overlapping boxes with CONFLICTING names
        # ("Bed" vs "Bathroom") are distinct rooms with imprecise geometry —
        # merging them silently destroys a surveyed room, so we never do.
        if a.floor_idx != b.floor_idx:
            return False
        ax, ay, aw, ah = a.bbox
        bx, by, bw, bh = b.bbox
        ov = _bbox_iou_overlap((ax, ay, ax + aw, ay + ah),
                               (bx, by, bx + bw, by + bh))
        if ov < 0.35:
            return False
        an, bn = _num_norm(a.number), _num_norm(b.number)
        la = (a.label or '').strip().lower()
        lb = (b.label or '').strip().lower()

        # Helper to standardise common abbreviations and naming variations
        def _std(s):
            s = s.lower().replace("room", "").replace(" ", "").replace("_", "").replace("-", "").strip()
            replacements = {
                "bedroom": "bedroom",
                "bed": "bedroom",
                "bathroom": "bathroom",
                "bath": "bathroom",
                "kitchen": "kitchen",
                "kit": "kitchen",
                "living": "living",
                "liv": "living",
                "lounge": "living",
                "cupboard": "cupboard",
                "cup": "cupboard",
                "cpd": "cupboard",
                "stairs": "stair",
                "staircase": "stair",
                "stair": "stair",
                "corridor": "hall",
                "hallway": "hall",
                "hall": "hall",
                "lobby": "hall",
                "toilet": "wc",
                "watercloset": "wc",
                "wc": "wc",
            }
            for k, v in replacements.items():
                if s.startswith(k):
                    suffix = s[len(k):]
                    return v + suffix
            return s

        # Hard gate: if BOTH names are present and they disagree (neither is
        # the other, nor a prefix of the other), these are different rooms.
        sa, sb = _std(la), _std(lb)
        if la and lb and not (
            sa == sb or sa.startswith(sb) or sb.startswith(sa)
        ):
            return False
        # Names are compatible or at least one is missing. Merge when they
        # share an explicit number, or both names are present (and, per the
        # gate above, compatible).
        if an and bn and an == bn:
            return True
        if la and lb:
            return True
        # One name missing and numbers don't tie them — only near-identical
        # boxes are safe to treat as the same physical space.
        if ov >= 0.70:
            return True
        return False


    def _is_generic_model_label(room):
        return bool(re.fullmatch(r"Room\s+\d+", (room.label or "").strip(), re.IGNORECASE))

    # YOLO can split one physical room into multiple overlapping boxes. When
    # the AI has named the real room, discard an overlapping model-only
    # placeholder instead of exporting it as "Room 3". Keep non-overlapping
    # placeholders because they may represent a room the AI failed to read.
    filtered_rooms = []
    for room in plan.rooms:
        if _is_generic_model_label(room):
            rx, ry, rw, rh = room.bbox
            overlaps_named = any(
                other is not room
                and other.floor_idx == room.floor_idx
                and not _is_generic_model_label(other)
                and _bbox_iou_overlap(
                    (rx, ry, rx + rw, ry + rh),
                    (
                        other.bbox[0],
                        other.bbox[1],
                        other.bbox[0] + other.bbox[2],
                        other.bbox[1] + other.bbox[3],
                    ),
                ) >= 0.50
                for other in plan.rooms
            )
            if overlaps_named:
                print(f"[POST] Removed overlapping unnamed model room '{room.label}' #{room.number}")
                continue
        filtered_rooms.append(room)
    plan.rooms = filtered_rooms

    merged_out = []
    for room in sorted(plan.rooms, key=lambda r: -(r.bbox[2] * r.bbox[3])):
        dup_of = next((m for m in merged_out if _same_room(m, room)), None)
        if dup_of is None:
            merged_out.append(room)
            continue
        # `dup_of` is the larger room (largest-area first) — keep its
        # geometry, absorb the smaller duplicate's attributes.
        dup_of.has_acm = dup_of.has_acm or room.has_acm
        dup_of.has_stairs = dup_of.has_stairs or room.has_stairs
        if not dup_of.stairs_bbox and room.stairs_bbox:
            dup_of.stairs_bbox = room.stairs_bbox
        dup_of.no_access = dup_of.no_access or room.no_access
        dup_of.acm_color = dup_of.acm_color or room.acm_color
        dup_of.room_type = ('no_access' if dup_of.no_access
                            else 'acm' if dup_of.has_acm else 'clear')
        # Prefer a real number over a blank/placeholder one.
        if not _clean_number(dup_of.number) and _clean_number(room.number):
            dup_of.number = room.number
        # Prefer the more specific label (longer, or carrying a digit).
        kl, rl = (dup_of.label or '').strip(), (room.label or '').strip()
        def _clean_std(s):
            clean = re.sub(r'[\d\s]+', '', s).lower()
            repls = {"bed": "bedroom", "kit": "kitchen", "bath": "bathroom", "cup": "cupboard", "cpd": "cupboard"}
            return repls.get(clean, clean)
        kl_std, rl_std = _clean_std(kl), _clean_std(rl)
        if rl and (
            not kl
            or (rl_std == kl_std and len(rl) > len(kl))
            or (len(rl_std) > len(kl_std))
            or (any(c.isdigit() for c in rl) and not any(c.isdigit() for c in kl))
        ):
            dup_of.label = rl
        # Prefer measured dimensions if the duplicate carried them.
        if (dup_of.dimension_source != "measured"
                and room.dimension_source == "measured"):
            dup_of.measured_width_m = room.measured_width_m
            dup_of.measured_height_m = room.measured_height_m
            dup_of.dimension_source = "measured"
        print(f"[POST] Merged duplicate '{room.label}' #{room.number} into "
              f"'{dup_of.label}' #{dup_of.number} (floor {dup_of.floor_idx})")
    if len(merged_out) != len(plan.rooms):
        print(f"[POST] Room merge: {len(plan.rooms)} -> {len(merged_out)} rooms")
    plan.rooms = merged_out

    # POST-PROCESS 2: Dedupe room numbers WITHIN EACH FLOOR. Numbers restart
    # per floor — a ground-floor "001" and a first-floor "001" are both
    # valid and must NOT collide. We only reassign when two rooms on the
    # SAME floor share a number; then the larger room keeps it and the
    # smaller takes the next value unused on that floor. This fixes both the
    # "Bedroom 005 / Bathroom 005" same-floor duplicate and the bug where a
    # multi-floor sketch had its upstairs rooms needlessly renumbered.
    seen = {}  # (floor_idx, number) -> room
    for room in sorted(plan.rooms, key=lambda r: -(r.bbox[2] * r.bbox[3])):
        num = str(room.number or '').strip()
        if not num:
            continue
        fkey = room.floor_idx
        if (fkey, num) in seen:
            used = {str(r.number).strip() for r in plan.rooms
                    if r.number and r.floor_idx == fkey}
            try:
                n = int(num)
                cand = n + 1
                while str(cand).zfill(len(num)) in used or str(cand) in used:
                    cand += 1
                new_num = str(cand).zfill(len(num))
            except ValueError:
                cand = 1
                while str(cand).zfill(2) in used:
                    cand += 1
                new_num = str(cand).zfill(2)
            print(f"[POST] Duplicate number '{num}' on '{room.label}' "
                  f"(floor {fkey}) -> reassigned to '{new_num}'")
            room.number = new_num
        seen[(fkey, str(room.number).strip())] = room

    # POST-PROCESS 3: Dedupe room LABELS WITHIN EACH FLOOR. Two rooms on the
    # same floor should never share a display label (e.g. "Landing" + "Landing").
    # Keep the first occurrence verbatim; append " (number)" or " 2", " 3", ... to later
    # duplicates. Done per-floor so a "Bathroom" on the ground floor and a
    # "Bathroom" upstairs both keep their plain name.
    # Exempt generic rooms like "Bed", "Bedroom", "Cupboard", "Store", "Office", "Room", "WC", "CPD"
    # from being forced to have unique display labels since their room numbers distinguish them.
    EXEMPT_DEDUP_LABELS = {
        "bed", "bedroom", "cupboard", "store", "office", "room", "wc", "cpd",
        "bathroom", "bath", "kitchen", "kit", "living room", "lounge", "lobby",
        "landing", "hallway", "hall", "stairs", "staircase", "utility", "toilet",
        "shop", "shop floor"
    }
    label_counts = {}
    for room in plan.rooms:
        base = (room.label or '').strip()
        if not base:
            continue
        key = (room.floor_idx, base.lower())
        if key not in label_counts:
            label_counts[key] = 1
        else:
            label_counts[key] += 1
            if base.lower() in EXEMPT_DEDUP_LABELS:
                # Do not modify the display label for exempt generic rooms
                continue
            
            clean_num = str(room.number or '').strip()
            if clean_num and not clean_num.lower().startswith('none'):
                # Keep room label verbatim if a valid room number is already present
                # to avoid redundant parentheses in Visio drawings
                continue
            
            new_label = f"{base} {label_counts[key]}"
            print(f"[POST] Duplicate label '{base}' (floor {room.floor_idx}) "
                  f"-> '{new_label}' (room #{room.number})")
            room.label = new_label


    # Backfill sample target_floor_idx from target_room_number when possible.
    num_to_floor = {}
    for r in plan.rooms:
        if r.number:
            num_to_floor[str(r.number).strip().lstrip("0") or "0"] = r.floor_idx
            num_to_floor[str(r.number).strip()] = r.floor_idx
    for s in plan.samples:
        if s.target_room_number:
            key = s.target_room_number.strip()
            if key in num_to_floor:
                s.target_floor_idx = num_to_floor[key]
            elif key.lstrip("0") in num_to_floor:
                s.target_floor_idx = num_to_floor[key.lstrip("0")]

    # Check for 2x2 multi-floor grid layout using YOLO room centroids
    is_single_floor = False
    if gpt4o_data and gpt4o_data.get("floor_name"):
        panel = str(gpt4o_data.get("floor_name")).strip().lower()
        if panel and panel not in _PANEL_FLOOR_MULTI:
            is_single_floor = True

    left_rooms = [r for r in plan.rooms if (r.bbox[0] + r.bbox[2]/2) < sketch_w * 0.5]
    right_rooms = [r for r in plan.rooms if (r.bbox[0] + r.bbox[2]/2) >= sketch_w * 0.5]
    top_rooms = [r for r in plan.rooms if (r.bbox[1] + r.bbox[3]/2) < sketch_h * 0.5]
    bottom_rooms = [r for r in plan.rooms if (r.bbox[1] + r.bbox[3]/2) >= sketch_h * 0.5]

    if len(plan.rooms) >= 10 and not is_single_floor:
        if len(left_rooms) >= 2 and len(right_rooms) >= 2 and len(top_rooms) >= 2 and len(bottom_rooms) >= 2:
            print("[FLOOR] Detected 2x2 grid multi-floor sheet! Applying spatial floor classification...")
            for r in plan.rooms:
                cx = r.bbox[0] + r.bbox[2] / 2
                cy = r.bbox[1] + r.bbox[3] / 2
                
                # Quadrants:
                # - Top-Left: First Floor (floor_idx = 1)
                # - Bottom-Left: Second Floor (floor_idx = 2)
                # - Top-Right: Loft (floor_idx = 3)
                # - Bottom-Right: Ground Floor (floor_idx = 0)
                if cx < sketch_w * 0.5:
                    if cy < sketch_h * 0.5:
                        r.floor_idx = 1
                    else:
                        r.floor_idx = 2
                else:
                    if cy < sketch_h * 0.5:
                        r.floor_idx = 3
                    else:
                        r.floor_idx = 0

    # Drop spurious isolated room slivers (e.g. detections over sample arrows)
    # before floor assignment so they don't render as a floating room.
    plan.rooms = _drop_spurious_isolated_rooms(plan.rooms, sketch_w, sketch_h)

    # External surveys are a single area, but YOLO can over-detect the boundary
    # as extra boxes (a duplicate "External" or an unnamed "Room N" placeholder).
    # Collapse to one "External" room. Guarded to an EXACT "External" label in a
    # SMALL plan, so a real room like "External Entrance" in a full multi-room
    # plan is never collapsed.
    _is_external_survey = (
        len(plan.rooms) <= 3
        and any(str(r.label or "").strip().lower() == "external" for r in plan.rooms)
    )
    if _is_external_survey and len(plan.rooms) > 1:
        primary = max(plan.rooms, key=lambda r: float(r.bbox[2]) * float(r.bbox[3]))
        primary.label = "External"
        print(f"[POST] External survey: collapsed {len(plan.rooms)} boxes to 1 area")
        plan.rooms = [primary]

    # Explicit Loft/Attic rooms must have their own floor index. AI commonly
    # assigns the detached loft drawing the same index as the first-floor
    # rooms; leaving that unchanged causes the exporter to put everything on
    # one page or rename the entire first floor to Loft.
    explicit_loft_rooms = [
        r for r in plan.rooms
        if "loft" in str(r.label or "").lower()
        or "attic" in str(r.label or "").lower()
        or "loft" in str(r.floor or "").lower()
        or "attic" in str(r.floor or "").lower()
    ]
    if explicit_loft_rooms and len(explicit_loft_rooms) < len(plan.rooms):
        non_loft_indices = {r.floor_idx for r in plan.rooms if r not in explicit_loft_rooms}
        loft_idx = 3
        while loft_idx in non_loft_indices:
            loft_idx += 1
        for room in explicit_loft_rooms:
            room.floor_idx = loft_idx
            room.floor = "Loft"
        loft_numbers = {
            str(r.number or "").strip().lstrip("0") or "0"
            for r in explicit_loft_rooms
            if r.number
        }
        for sample in plan.samples:
            sample_number = str(sample.target_room_number or "").strip().lstrip("0") or "0"
            if sample.target_room_number and sample_number in loft_numbers:
                sample.target_floor_idx = loft_idx
        print(f"[FLOOR] Separated {len(explicit_loft_rooms)} explicit Loft/Attic room(s) "
              f"onto floor index {loft_idx}")

    # Build floor_names map (idx -> title) from per-room floor indices.
    # Ground=0, First=1, Second=2, Loft=3. Room.floor (the string) is updated
    # to match the map so the Visio export can key off it.
    DEFAULT_FLOOR_NAMES = {0: "Ground Floor", 1: "First Floor", 2: "Second Floor", 3: "Loft"}
    present_idx = sorted({r.floor_idx for r in plan.rooms})
    plan.floor_names = {}
    for idx in present_idx:
        # Check if any room on this floor is a loft/attic room or has loft in its floor name
        is_loft_floor = False
        for r in plan.rooms:
            if r.floor_idx == idx:
                fl_name = str(r.floor or "").lower()
                lbl_name = str(r.label or "").lower()
                if "loft" in fl_name or "attic" in fl_name or "loft" in lbl_name or "attic" in lbl_name:
                    is_loft_floor = True
                    break
        if is_loft_floor:
            plan.floor_names[idx] = "Loft"
        else:
            if gpt4o_data and gpt4o_data.get("floor_name") and plan.floor_title and len(present_idx) == 1:
                plan.floor_names[idx] = plan.floor_title
            else:
                plan.floor_names[idx] = DEFAULT_FLOOR_NAMES.get(idx, f"Floor {idx}")

    for r in plan.rooms:
        r.floor = plan.floor_names.get(r.floor_idx, f"Floor {r.floor_idx}")

    n_floors = len(plan.floor_names)
    print(f"[MERGE] Final: {len(plan.rooms)} rooms on {n_floors} floor(s), "
          f"{len(plan.samples)} samples")
    return plan


def _check_acm_in_region(model_rooms: List[Room], x, y, w, h) -> Tuple[bool, Optional[str]]:
    """Check if any model-detected ACM room overlaps with this region."""
    for mr in model_rooms:
        if not mr.has_acm:
            continue
        mx, my, mw, mh = mr.bbox
        # Check overlap
        ox = max(0, min(x + w, mx + mw) - max(x, mx))
        oy = max(0, min(y + h, my + mh) - max(y, my))
        overlap = ox * oy
        if overlap > 0.3 * min(w * h, mw * mh):
            return True, mr.acm_color
    return False, None


def _room_snap_threshold_px(rooms: List[Room], sketch_w: int, sketch_h: int) -> int:
    """Return a conservative shared-wall snap distance."""
    short_sides = sorted(
        min(float(room.bbox[2]), float(room.bbox[3]))
        for room in rooms
        if room.bbox[2] > 0 and room.bbox[3] > 0
    )
    if not short_sides:
        return max(2, int(round(min(sketch_w, sketch_h) * 0.005)))
    median_short = short_sides[len(short_sides) // 2]
    image_cap = max(3.0, min(sketch_w, sketch_h) * 0.015)
    return max(2, int(round(min(median_short * 0.12, image_cap))))


def _snap_shared_walls(rooms: List[Room], sketch_w: int, sketch_h: int):
    """Snap nearby room edges together to form clean shared walls."""
    if len(rooms) < 2:
        return

    SNAP = _room_snap_threshold_px(rooms, sketch_w, sketch_h)
    min_size = max(15, int(min(sketch_w, sketch_h) * 0.01))
    edges = [list(r.bbox) for r in rooms]

    for i in range(len(rooms)):
        x1i, y1i, wi, hi = edges[i]
        x2i, y2i = x1i + wi, y1i + hi

        for j in range(i + 1, len(rooms)):
            x1j, y1j, wj, hj = edges[j]
            x2j, y2j = x1j + wj, y1j + hj

            v_overlap = min(y2i, y2j) - max(y1i, y1j)
            h_overlap = min(x2i, x2j) - max(x1i, x1j)

            # Snap vertical walls (side by side)
            if v_overlap > 0:
                if abs(x2i - x1j) < SNAP:
                    mid = (x2i + x1j) // 2
                    # Safeguard: check if new widths stay above min_size
                    new_wi = mid - edges[i][0]
                    new_wj = (x1j + wj) - mid
                    if new_wi >= min_size and new_wj >= min_size:
                        edges[i][2] = new_wi
                        edges[j][2] = new_wj
                        edges[j][0] = mid
                        x2i = mid
                        x1j = mid
                elif abs(x1i - x2j) < SNAP:
                    mid = (x1i + x2j) // 2
                    new_wi = (x1i + wi) - mid
                    new_wj = mid - edges[j][0]
                    if new_wi >= min_size and new_wj >= min_size:
                        edges[j][2] = new_wj
                        edges[i][2] = new_wi
                        edges[i][0] = mid
                        x1i = mid
                        x2j = mid

            # Snap horizontal walls (stacked)
            if h_overlap > 0:
                if abs(y2i - y1j) < SNAP:
                    mid = (y2i + y1j) // 2
                    new_hi = mid - edges[i][1]
                    new_hj = (y1j + hj) - mid
                    if new_hi >= min_size and new_hj >= min_size:
                        edges[i][3] = new_hi
                        edges[j][3] = new_hj
                        edges[j][1] = mid
                        y2i = mid
                        y1j = mid
                elif abs(y1i - y2j) < SNAP:
                    mid = (y1i + y2j) // 2
                    new_hi = (y1i + hi) - mid
                    new_hj = mid - edges[j][1]
                    if new_hi >= min_size and new_hj >= min_size:
                        edges[j][3] = new_hj
                        edges[i][3] = new_hi
                        edges[i][1] = mid
                        y1i = mid
                        y2j = mid

    # Write back
    for i, room in enumerate(rooms):
        # Additional sanity checks to prevent division by zero or negative size
        edges[i][2] = max(min_size, edges[i][2])
        edges[i][3] = max(min_size, edges[i][3])
        room.bbox = tuple(edges[i])
        room.area = edges[i][2] * edges[i][3]



# ============================================================================
# Step 5: Visio Export
# ============================================================================

def _estimate_pixel_scale(rooms: List[Room], sketch_w: int, sketch_h: int) -> float:
    """Estimate meters-per-pixel from building footprint and room count.

    Uses heuristics based on UK building sizes:
    - 1-5 rooms: small flat/house ~8-12m wide
    - 6-10 rooms: large house ~12-18m wide
    - 11-20 rooms: commercial unit ~20-30m wide
    - 20+ rooms: large commercial ~30-50m wide
    """
    if not rooms:
        return 0.05  # a sensible default of 5cm per pixel if no rooms are present

    # Get building bounding box from room bboxes
    all_x1 = min(r.bbox[0] for r in rooms)
    all_y1 = min(r.bbox[1] for r in rooms)
    all_x2 = max(r.bbox[0] + r.bbox[2] for r in rooms)
    all_y2 = max(r.bbox[1] + r.bbox[3] for r in rooms)

    bld_w_px = max(1, all_x2 - all_x1)
    bld_h_px = max(1, all_y2 - all_y1)
    bld_max_px = max(bld_w_px, bld_h_px, 10)  # Safeguard: prevent division by zero or tiny numbers

    from config import (
        EST_WIDTH_M_SMALL, EST_WIDTH_M_MEDIUM, EST_WIDTH_M_LARGE, EST_WIDTH_M_XLARGE,
    )
    n = len(rooms)
    if n <= 5:
        est_width_m = EST_WIDTH_M_SMALL
    elif n <= 10:
        est_width_m = EST_WIDTH_M_MEDIUM
    elif n <= 20:
        est_width_m = EST_WIDTH_M_LARGE
    else:
        est_width_m = EST_WIDTH_M_XLARGE

    scale = est_width_m / bld_max_px
    print(f"[SCALE] {n} rooms, building ~{bld_w_px}x{bld_h_px}px, "
          f"est {est_width_m:.0f}m -> {scale:.5f} m/px")
    return scale


def _plan_to_detected(plan: FloorPlan, style: str = "CLEAN_TEMPLATE") -> Dict[str, Any]:
    """Build the dict format expected by the Visio exporters."""
    rooms_out = []
    for room in plan.rooms:
        label = room.label
        if room.no_access:
            # Render "NO ACCESS" as a prefix so users see it prominently but
            # the room name is still visible underneath.
            label = f"NO ACCESS\n{room.label}" if room.label else "NO ACCESS"
        rooms_out.append({
            "label": label,
            "type": room.room_type,
            "bbox": list(room.bbox),
            "room_number": room.number,
            "floor": room.floor,
            "floor_idx": room.floor_idx,
            "has_stairs": room.has_stairs,
            "stairs_bbox": list(room.stairs_bbox) if room.stairs_bbox else None,
            "no_access": room.no_access,
            "acm_color": room.acm_color,
            "measured_width_m": room.measured_width_m,
            "measured_height_m": room.measured_height_m,
            "dimension_source": room.dimension_source,
            "label_bbox": list(room.label_bbox) if room.label_bbox else None,
            "geometry_source": room.geometry_source,
            "detection_confidence": room.detection_confidence,
            "is_fallback": room.is_fallback,
        })

    sample_details = []
    for s in plan.samples:
        sample_details.append({
            "id": s.id,
            "material": s.material,
            "location": [int(s.x_pct / 100 * plan.image_size[0]),
                         int(s.y_pct / 100 * plan.image_size[1])],
            "acm_positive": s.acm_positive,
            "is_reference": s.is_ref,
            "target_floor_idx": s.target_floor_idx,
            "target_room_number": s.target_room_number,
        })

    # Sort floor pages in numeric order so Ground (0) comes before First (1).
    floor_pages = [
        {"idx": idx, "title": plan.floor_names.get(idx, f"Floor {idx}")}
        for idx in sorted(plan.floor_names.keys())
    ] or [{"idx": 0, "title": plan.floor_title}]

    detected = {
        "floor_title": plan.floor_title,
        "room_count": len(rooms_out),
        "rooms": rooms_out,
        "sample_labels": [s.id for s in plan.samples],
        "sample_details": sample_details,
        "doors": plan.doors,
        "windows": plan.windows,
        "stairs_detected": plan.stairs,
        "atm_location": getattr(plan, "atm_location", None),
        "db_location": getattr(plan, "db_location", None),
        "gas_meter": None,
        "water_stop_tap": None,
        "has_cable_route": getattr(plan, "has_cable_route", False),
        "cable_route_path": None,
        "caveats": None,
        "image_quality": "GOOD",
        "sketch_size": list(plan.image_size),
        "floors": floor_pages,
        "style": style,
        "pixel_scale": plan.pixel_scale,
        "page": plan.page,
        "total_pages": plan.total_pages,
    }

    return detected


def export_visio(plan: FloorPlan, output_path: str) -> str:
    """Export FloorPlan to professional .vsdx via Aspose or COM."""
    detected = _plan_to_detected(plan)

    abs_output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    # Default renderer is platform-aware: COM needs Microsoft Visio (Windows
    # only), so Linux/macOS default to the native Aspose renderer. Override
    # with the RENDERER env var.
    import sys
    _default_renderer = "com" if sys.platform == "win32" else "aspose"
    renderer = os.environ.get("RENDERER", _default_renderer).strip().lower()

    # Determine template path: env override, else the repo's bundled template
    # (no hardcoded per-machine paths — must work on Windows and Linux alike).
    _repo_template = os.path.join(os.path.dirname(__file__), "utils", "visio", "template.vsdx")
    template_path = os.environ.get("VISIO_TEMPLATE_PATH") or _repo_template
    if not os.path.exists(template_path):
        template_path = _repo_template

    # Mode 1: Force Aspose
    if renderer == "aspose":
        try:
            from plans.aspose_renderer import render_plan_to_vsdx
            out = render_plan_to_vsdx(detected, plan.project_number, template_path, abs_output)
            if out and os.path.exists(out):
                return out
        except Exception as e:
            print(f"[VISIO] Force Aspose export failed: {e}")

    # Mode 2: COM (or fallbacks)
    if renderer == "com" or renderer != "aspose":
        # Try professional Visio COM export
        try:
            from utils.visio.professional_visio import generate_visio_from_detected
            # Match Acorn's manual plans: the Loft gets its own tab/page, and all
            # other floors share ONE page as labelled sections ("Ground Floor:",
            # "First Floor:").
            out = generate_visio_from_detected(
                detected, plan.project_number, abs_output,
                detection_hints={"split_loft": True},
            )
            if out and os.path.exists(out):
                return out
        except Exception as e:
            print(f"[VISIO] Professional COM export failed: {e}")

    # Fallback 1: Aspose.Diagram (if not already tried/forced)
    if renderer != "aspose":
        try:
            print("[VISIO] COM export failed or not supported. Falling back to native Aspose.Diagram export...")
            from plans.aspose_renderer import render_plan_to_vsdx
            out = render_plan_to_vsdx(detected, plan.project_number, template_path, abs_output)
            if out and os.path.exists(out):
                return out
        except Exception as e:
            print(f"[VISIO] Aspose fallback failed: {e}")

    # Fallback 2: simple COM export
    try:
        from utils.visio.visio_com_export import create_visio_plan

        class _SimpleRoom:
            def __init__(self, bbox, label, room_type):
                self.bbox = bbox
                self.label = label
                self.room_type = room_type
                self.has_acm = room_type == 'acm'

        simple_rooms = [_SimpleRoom(r.bbox, r.label, r.room_type) for r in plan.rooms]
        out = create_visio_plan(
            rooms=simple_rooms,
            output_path=abs_output,
            image_size=plan.image_size,
            title=f"Floor Plan - {plan.project_number}",
        )
        if out:
            return out
    except Exception as e:
        print(f"[VISIO] Simple COM export failed: {e}")

    # Fallback 3: XML export (platform-independent, no Visio or Aspose required)
    try:
        print("[VISIO] COM and Aspose exports failed or not supported. Falling back to native XML-based VSDX export...")
        from utils.visio.visio_xml_export import create_visio_plan as create_xml_visio_plan

        class _SimpleRoom:
            def __init__(self, bbox, label, room_type):
                self.bbox = bbox
                self.label = label
                self.room_type = room_type
                self.has_acm = room_type == 'acm'

        simple_rooms = [_SimpleRoom(r.bbox, r.label, r.room_type) for r in plan.rooms]
        out = create_xml_visio_plan(
            rooms=simple_rooms,
            output_path=abs_output,
            image_size=plan.image_size,
            title=f"Floor Plan - {plan.project_number}",
        )
        if out and os.path.exists(out):
            print(f"[VISIO] Native XML VSDX export succeeded: {out}")
            return out
    except Exception as e:
        print(f"[VISIO] XML export failed: {e}")

    raise RuntimeError("Visio export failed — is Microsoft Visio or Aspose.Diagram installed?")


def export_visio_overlay(plan: FloorPlan, output_path: str, sketch: np.ndarray) -> str:
    """Export a Visio file with the processed plan image locked as background."""
    detected = _plan_to_detected(plan, style="SOURCE_OVERLAY")

    bg_dir = _PROJECT_ROOT / "output" / "cache" / "overlay_backgrounds"
    bg_dir.mkdir(parents=True, exist_ok=True)
    safe_project = re.sub(r"[^A-Za-z0-9_.-]+", "_", plan.project_number or "plan")
    background_path = bg_dir / f"{safe_project}.png"
    if not cv2.imwrite(str(background_path), sketch):
        raise RuntimeError(f"Could not save overlay background: {background_path}")

    abs_output = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_output), exist_ok=True)

    try:
        from utils.visio.overlay_visio import generate_overlay_visio
        out = generate_overlay_visio(
            detected=detected,
            background_image_path=str(background_path),
            project_number=plan.project_number,
            output_path=abs_output,
        )
        if out and os.path.exists(out):
            return out
    except Exception as e:
        print(f"[VISIO-OVERLAY] Export failed: {e}")

    raise RuntimeError("Visio overlay export failed - is Microsoft Visio installed?")


# ============================================================================
# Cache (simple file-based)
# ============================================================================

_CACHE_DIR = _PROJECT_ROOT / "output" / "cache"


_AI_CACHE_VERSION = os.environ.get("AI_CACHE_VERSION", "2026-06-13-evidence-v1")


def _cache_key(sketch: np.ndarray) -> str:
    """Version cache entries so prompt/validation changes invalidate stale AI data."""
    digest = hashlib.sha256()
    digest.update(_AI_CACHE_VERSION.encode("utf-8"))
    digest.update(sketch.tobytes())
    return digest.hexdigest()[:16]


def _get_cached(key: str) -> Optional[Dict]:
    path = _CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            print(f"[CACHE] HIT for {key}")
            return data
        except Exception:
            pass
    return None


def _save_cache(key: str, data: Dict):
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{key}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"[CACHE] Saved {key}")


def _correct_samples_by_color(sketch: np.ndarray, gpt4o_data: Optional[Dict]):
    """Classify sample ACM status using color analysis of the preprocessed sketch."""
    if not gpt4o_data or "samples" not in gpt4o_data or not gpt4o_data["samples"]:
        return
    
    from utils.room_detection.color_analysis import analyze_colors
    import cv2
    
    h, w = sketch.shape[:2]
    try:
        color_info = analyze_colors(sketch)
        red_mask = color_info["red_mask"]
        green_mask = color_info["green_mask"]
    except Exception as e:
        print(f"[COLOR CLASSIFY] Color analysis failed: {e}")
        return

    for s in gpt4o_data["samples"]:
        try:
            x_pct = float(s.get("x_pct", 0) or 0)
            y_pct = float(s.get("y_pct", 0) or 0)
        except (TypeError, ValueError):
            continue
            
        px = int(x_pct / 100 * w)
        py = int(y_pct / 100 * h)
        
        # Scale the color window with image resolution.
        r = max(5, int(round(min(h, w) * 0.01)))
        y1, y2 = max(0, py - r), min(h, py + r)
        x1, x2 = max(0, px - r), min(w, px + r)
        
        if y2 > y1 and x2 > x1:
            window_red = red_mask[y1:y2, x1:x2]
            window_green = green_mask[y1:y2, x1:x2]
            
            red_count = cv2.countNonZero(window_red)
            green_count = cv2.countNonZero(window_green)
            
            min_color_pixels = max(3, int(round(window_red.size * 0.01)))
            if red_count > min_color_pixels and red_count > green_count:
                print(f"[COLOR CLASSIFY] Sample '{s.get('id')}' classified as RED (ACM Positive) - red pixels: {red_count}")
                s["acm_positive"] = True
            elif green_count > min_color_pixels and green_count > red_count and not s.get("acm_positive"):
                # SAFETY: a GREEN reading only confirms a negative sample; it must
                # NOT downgrade one the model already read as ACM-positive (green
                # grid lines / logo could otherwise cause a false negative).
                print(f"[COLOR CLASSIFY] Sample '{s.get('id')}' classified as GREEN (ACM Negative) - green pixels: {green_count}")
                s["acm_positive"] = False


def _is_generic_sheet_name(val) -> bool:
    if not val:
        return True
    s = str(val).strip().lower()
    return any(x in s for x in ["page", "plan", "survey", "sheet", "sketch", "drawing", "asbestos", "acorn"])


def _promote_detached_loft_candidate(ai: Optional[Dict]) -> bool:
    """Promote one dominant detached AI box to Loft when OCR missed its label."""
    if not ai or not _is_generic_sheet_name(ai.get("floor_name")):
        return False
    rooms = ai.get("rooms") or []
    if any("loft" in str(r.get("name") or "").lower()
           or "attic" in str(r.get("name") or "").lower() for r in rooms):
        return False

    boxed = []
    for room in rooms:
        if room.get("is_fallback"):
            continue
        try:
            x = float(room["x_pct"])
            y = float(room["y_pct"])
            w = float(room["w_pct"])
            h = float(room["h_pct"])
        except (KeyError, TypeError, ValueError):
            continue
        if w > 0 and h > 0:
            boxed.append((room, x, y, w, h))
    if len(boxed) < 5:
        return False

    areas = sorted(w * h for _, _, _, w, h in boxed)
    median_area = areas[len(areas) // 2]
    candidate = max(boxed, key=lambda item: item[3] * item[4])
    room, x, y, w, h = candidate
    area_ratio = (w * h) / max(median_area, 1.0)
    if area_ratio < 2.0:
        return False

    def _gap(a, b):
        _, ax, ay, aw, ah = a
        _, bx, by, bw, bh = b
        horizontal = max(0.0, bx - (ax + aw), ax - (bx + bw))
        vertical = max(0.0, by - (ay + ah), ay - (by + bh))
        return (horizontal ** 2 + vertical ** 2) ** 0.5

    nearest_gap = min(_gap(candidate, other) for other in boxed if other is not candidate)
    other_centres_x = sorted(ox + ow / 2 for other, ox, oy, ow, oh in boxed if other is not room)
    other_centres_y = sorted(oy + oh / 2 for other, ox, oy, ow, oh in boxed if other is not room)
    median_x = other_centres_x[len(other_centres_x) // 2]
    median_y = other_centres_y[len(other_centres_y) // 2]
    centre_distance = ((x + w / 2 - median_x) ** 2 + (y + h / 2 - median_y) ** 2) ** 0.5
    min_x = min(bx for _, bx, by, bw, bh in boxed)
    min_y = min(by for _, bx, by, bw, bh in boxed)
    max_x = max(bx + bw for _, bx, by, bw, bh in boxed)
    max_y = max(by + bh for _, bx, by, bw, bh in boxed)
    layout_diagonal = ((max_x - min_x) ** 2 + (max_y - min_y) ** 2) ** 0.5
    gap_threshold = max(2.0, min(w, h) * 0.10)
    distance_threshold = layout_diagonal * 0.22
    if nearest_gap < gap_threshold or centre_distance < distance_threshold:
        return False

    old_name = room.get("name")
    room["name"] = "Loft"
    room["floor"] = 2
    room["inferred_loft"] = True
    print(f"[FLOOR] Promoted detached dominant room '{old_name}' #{room.get('number')} "
          f"to Loft (area_ratio={area_ratio:.1f}, nearest_gap={nearest_gap:.1f}%, "
          f"centre_distance={centre_distance:.1f}%)")
    return True


def _resolve_overlay_mode(overlay: Optional[bool]) -> bool:
    """Choose between the production vector renderer and review overlay.

    Production requests are expected to generate an editable vector plan. The
    final geometry quality gate prevents uncertain room boxes from being
    exported. Source overlay remains available explicitly for manual review.
    """
    if overlay is not None:
        return overlay

    mode = os.environ.get("DRAW_OUTPUT_MODE", "vector").strip().lower()
    if mode in {"overlay", "source", "faithful"}:
        return True
    if mode in {"vector", "reconstructed"}:
        return False
    raise ValueError(
        "DRAW_OUTPUT_MODE must be 'overlay' (faithful source) or 'vector' "
        f"(AI-reconstructed geometry), got {mode!r}"
    )


# ============================================================================
# Main Pipeline
# ============================================================================

def process_sketch(
    image_path: str,
    output_path: str = None,
    model_path: str = None,
    no_model: bool = False,
    overlay: Optional[bool] = None,
) -> Tuple[str, "FloorPlan"]:
    """
    Process a single sketch image end-to-end.

    Pipeline flow (the trained YOLO model and GPT-4o are BOTH used):
      preprocess
      -> GPT-4o labels pass      (room names/numbers, ACM, samples)
      -> GPT-4o full-layout pass (proportional room positions)
      -> YOLO geometry           (room boxes; config.USE_MODEL, --no-model)
      -> merge                   (YOLO geometry + GPT-4o labels)
      -> Visio export.

    GPT-4o is the only source of room *names* — YOLO only outputs class
    IDs. When YOLO under-detects, merge_results re-adds the AI-only rooms.

    Args:
        image_path: Path to sketch image (jpg/png)
        output_path: Output .vsdx path (auto-generated if None)
        model_path: Optional YOLO model path.
        no_model: Skip YOLO geometry and use AI layout only.
        overlay: True preserves the source plan, False reconstructs vector
            geometry, and None uses DRAW_OUTPUT_MODE (default: vector).
    """
    from config import OUTPUT_FOLDER

    # Derive project number from filename
    fname = os.path.basename(image_path)
    base_name = os.path.splitext(fname)[0]
    match = re.search(r'N-?\d{5}', fname, re.IGNORECASE)
    project_number = match.group(0).upper() if match else base_name

    if output_path is None:
        output_path = os.path.join(OUTPUT_FOLDER, "visio", f"{base_name}.vsdx")

    print(f"\n{'=' * 60}")
    print(f"  ACORN ATLAS — {fname}")
    print(f"  Project: {project_number}")
    print(f"{'=' * 60}")

    t0 = time.time()

    # Step 1: Load and preprocess
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load: {image_path}")
    print(f"  Image: {image.shape[1]}x{image.shape[0]}")

    sketch = preprocess_sketch(image)
    sketch_h, sketch_w = sketch.shape[:2]
    print(f"  Sketch: {sketch_w}x{sketch_h}")

    # Step 2+3: Two-pass approach
    # PASS 1: Get AI labels first (fast, gives us expected room count)
    cache_key = _cache_key(sketch)
    ai_data = _get_cached(cache_key)
    # Invalidate cache if it has suspiciously few rooms (likely a failed call)
    if ai_data and len(ai_data.get("rooms", [])) <= 1:
        print(f"  CACHE: Only {len(ai_data.get('rooms', []))} rooms — likely stale, re-calling AI")
        ai_data = None
    if not ai_data:
        t2 = time.time()
        # First pass always uses labels-only (cheaper, faster)
        ai_data = get_room_labels(sketch, labels_only=True)
        ai_time = time.time() - t2
        if ai_data:
            _save_cache(cache_key, ai_data)
            print(f"  AI: {len(ai_data.get('rooms', []))} rooms ({ai_time:.1f}s)")
        else:
            print(f"  AI: FAILED ({ai_time:.1f}s)")

    # Clean the tiled read (collapse re-read duplicates, drop sample labels)
    # then stamp a consistent floor from the form panel — both BEFORE
    # reconciliation, so it works from a clean, consistent room list.
    _apply_panel_floor(ai_data)
    _dedup_room_list(ai_data)

    # GPT-4o is the layout source alongside YOLO geometry.
    # If the labels-only pass lacks bboxes (it always does), call full-layout
    # to get proportional room positions.
    if ai_data:
        has_usable_layout = _full_layout_geometry_is_usable(ai_data)
        if not has_usable_layout:
            print(f"\n  AI: Re-calling with full-layout prompt...")
            t2b = time.time()
            labels_only_count = len(ai_data.get('rooms', []))
            ai_data_full = _get_cached(cache_key + "_full")
            if ai_data_full and not _full_layout_geometry_is_usable(ai_data_full):
                print("  CACHE: Full-layout geometry is artificial or incomplete; re-calling AI")
                ai_data_full = None
            if ai_data_full and ai_data_full.get("rooms"):
                print(f"  AI: Using cached full-layout ({len(ai_data_full.get('rooms', []))} rooms)")
            else:
                ai_data_full = get_room_labels(sketch, labels_only=False)
                if ai_data_full and not _full_layout_geometry_is_usable(ai_data_full):
                    print("  AI: Full-layout geometry failed quality checks; keeping labels-only data")
                    ai_data_full = None
            # Reconcile floor names between the two passes before applying them.
            # If one pass has a valid floor and the other does not (e.g. read noise like 'mates'),
            # we overwrite the invalid one with the valid one so both passes agree.
            if ai_data and ai_data_full:
                fn_lo = ai_data.get("floor_name")
                fn_full = ai_data_full.get("floor_name")
                valid_lo = _is_valid_single_floor(fn_lo)
                valid_full = _is_valid_single_floor(fn_full)
                
                if valid_full and not valid_lo:
                    if _is_generic_sheet_name(fn_lo):
                        print(f"  [FLOOR] Floor is generic multi-floor sheet. Overwriting full-layout floor_name '{fn_full}' -> '{fn_lo}'")
                        ai_data_full["floor_name"] = fn_lo
                    else:
                        print(f"  [FLOOR] Overriding labels-only floor_name '{fn_lo}' with valid full-layout '{fn_full}'")
                        ai_data["floor_name"] = fn_full
                        # Re-apply floor and re-dedup labels-only rooms because floor changed!
                        _apply_panel_floor(ai_data)
                        _dedup_room_list(ai_data)
                elif valid_lo and not valid_full:
                    print(f"  [FLOOR] Overriding full-layout floor_name '{fn_full}' with valid labels-only '{fn_lo}'")
                    ai_data_full["floor_name"] = fn_lo
                elif not valid_full and not valid_lo and fn_lo:
                    # Borrow anyway if full has none but labels has something
                    ai_data_full["floor_name"] = fn_lo
            
            _propagate_explicit_multifloor_evidence(ai_data, ai_data_full)
            _apply_panel_floor(ai_data_full)
            _dedup_room_list(ai_data_full)
            if ai_data_full and ai_data_full.get("rooms"):
                # Merge names/numbers from the labels-only pass into the
                # full-layout pass. Labels-only is usually better at reading
                # room names ("Bed 1" vs "Bed"), full-layout is better at
                # spatial bboxes.
                #
                # CRITICAL: each labels-only room can be consumed at most once
                # as an upgrade source. Otherwise a single "Landing" in
                # labels-only can be applied to multiple unnamed rooms in
                # full-layout, producing duplicate names.
                # Numbers restart per floor, so a room is identified by the
                # (floor, number) pair — never the bare number, or an
                # upstairs "001" would be matched to a downstairs "001".
                lo_by_num = {}  # (floor_idx, number) -> labels-only room
                lo_by_prefix = {}  # prefix -> list of labels-only rooms (ordered)
                consumed_ids = set()  # id(room) for rooms already used as a source
                for r in ai_data.get("rooms", []):
                    n = str(r.get("number", "")).strip()
                    nm = str(r.get("name", "")).strip()
                    fl = _safe_int_floor(r.get("floor", 0))
                    if n and (fl, n) not in lo_by_num:
                        lo_by_num[(fl, n)] = r
                    if nm:
                        lo_by_prefix.setdefault(nm.lower().split()[0], []).append(r)

                def _take_by_prefix(prefix):
                    """Pop the first unconsumed labels-only room with this prefix."""
                    for cand in lo_by_prefix.get(prefix, []):
                        if id(cand) not in consumed_ids:
                            consumed_ids.add(id(cand))
                            return cand
                    return None

                def _take_by_number(floor_num):
                    """floor_num is a (floor_idx, number) tuple."""
                    cand = lo_by_num.get(floor_num)
                    if cand and id(cand) not in consumed_ids:
                        consumed_ids.add(id(cand))
                        return cand
                    return None

                # First pass: match by name prefix first (since names are more stable than numbers).
                # This ensures a "Kitchen" in full-layout matches "Kitchen 008" in labels-only,
                # even if full-layout incorrectly read its number.
                for fr in ai_data_full["rooms"]:
                    fname = str(fr.get("name", "")).strip()
                    if not fname:
                        continue
                    prefix = fname.lower().split()[0]
                    # Don't match multiple generic "bed" or "bath" rooms by prefix if they have different numbers
                    if prefix in ("bed", "bedroom", "cupboard", "cpd", "bathroom", "bath", "wc", "toilet", "cp"):
                        continue
                    lo_match = _take_by_prefix(prefix)
                    if lo_match:
                        lo_name = str(lo_match.get("name", "")).strip()
                        lo_num = str(lo_match.get("number", "")).strip()
                        print(f"  AI: Matched by prefix '{fname}' -> '{lo_name}' (upgrade number '{fr.get('number')}' -> '{lo_num}')")
                        if lo_name:
                            fr["name"] = lo_name
                        if lo_num:
                            fr["number"] = lo_num

                # Second pass: match by exact (floor, room_number) for the remaining unmatched rooms (like Bedrooms)
                for fr in ai_data_full["rooms"]:
                    fn = str(fr.get("number", "")).strip()
                    if not fn:
                        continue
                    ffl = _safe_int_floor(fr.get("floor", 0))
                    lo_match = _take_by_number((ffl, fn))
                    if lo_match:
                        lo_name = str(lo_match.get("name", "")).strip()
                        lo_num = str(lo_match.get("number", "")).strip()
                        print(f"  AI: Matched by number '{fr.get('name')}' -> '{lo_name}' (matched by #{fn})")
                        if lo_name:
                            fr["name"] = lo_name
                        if lo_num:
                            fr["number"] = lo_num

                # Third pass: match any remaining unnamed generic rooms by prefix
                for fr in ai_data_full["rooms"]:
                    fname = str(fr.get("name", "")).strip()
                    if not fname:
                        continue
                    # Skip if full-layout already has a specific name with digits
                    if any(ch.isdigit() for ch in fname):
                        continue
                    prefix = fname.lower().split()[0]
                    lo_match = _take_by_prefix(prefix)
                    if lo_match:
                        lo_name = str(lo_match.get("name", "")).strip()
                        lo_num = str(lo_match.get("number", "")).strip()
                        print(f"  AI: Matched remaining by prefix '{fname}' -> '{lo_name}' (upgrade number '{fr.get('number')}' -> '{lo_num}')")
                        if lo_name:
                            fr["name"] = lo_name
                        if lo_num:
                            fr["number"] = lo_num
                # If the full-layout call lost rooms that labels-only found,
                # first RETRY with an explicit hint naming the missing rooms —
                # GPT-4o responds well to being told what it skipped. Only
                # fall back to smart placement if the retry still misses them.
                def _missing(full, labels_only_rooms):
                    # Robust unconsumed-greedy matching: matches each labels-only room to a
                    # full-layout room. Unmatched labels-only rooms are marked as missed.
                    def _fl(r):
                        return _safe_int_floor(r.get("floor", 0))
                    def _clean_num(n):
                        if n is None:
                            return None
                        s = str(n).strip()
                        if not s or s.lower() in ("none", "null", "?"):
                            return None
                        return s
                    
                    full_list = list(full)
                    consumed = set()
                    missed = []
                    
                    # 1st pass: Match by exact (floor, room_number)
                    matched_lo_indices = set()
                    for idx, r in enumerate(labels_only_rooms):
                        num = _clean_num(r.get("number"))
                        if not num:
                            continue
                        fl = _fl(r)
                        for f_idx, fr in enumerate(full_list):
                            if f_idx in consumed:
                                continue
                            if _fl(fr) == fl and _clean_num(fr.get("number")) == num:
                                consumed.add(f_idx)
                                matched_lo_indices.add(idx)
                                break
                    
                    # 2nd pass: Match remaining by (floor, name prefix)
                    for idx, r in enumerate(labels_only_rooms):
                        if idx in matched_lo_indices:
                            continue
                        nm = str(r.get("name", "")).strip().lower()
                        if not nm:
                            continue
                        fl = _fl(r)
                        prefix = nm.split()[0]
                        found = False
                        for f_idx, fr in enumerate(full_list):
                            if f_idx in consumed:
                                continue
                            fr_name = str(fr.get("name", "")).strip().lower()
                            if _fl(fr) == fl and fr_name and fr_name.split()[0] == prefix:
                                consumed.add(f_idx)
                                matched_lo_indices.add(idx)
                                found = True
                                break
                        if not found:
                            missed.append(r)
                            
                    return missed

                missed = _missing(ai_data_full["rooms"], ai_data.get("rooms", []))
                if missed:
                    miss_list = ", ".join(f"#{r.get('number','?')} {r.get('name','?')}"
                                          for r in missed)
                    print(f"  AI: Full-layout missed {len(missed)} room(s): {miss_list} — "
                          f"retrying with explicit hint")
                    retry_prompt = _FULL_LAYOUT_PROMPT + (
                        f"\n\nYour previous response OMITTED these rooms that you "
                        f"confirmed exist on this sketch: {miss_list}. Include "
                        "them in the rooms array with accurate bounding boxes. "
                        "Look at where these rooms are drawn on the sketch and "
                        "give each a proportional x_pct/y_pct/w_pct/h_pct."
                    )
                    retry = get_room_labels_gpt4o(sketch, prompt=retry_prompt)
                    if retry and retry.get("rooms"):
                        still_missed = _missing(retry["rooms"], ai_data.get("rooms", []))
                        if len(still_missed) < len(missed):
                            # Merge recovered rooms from retry into the original ai_data_full
                            recovered_candidates = []
                            for r in retry["rooms"]:
                                r_num = str(r.get("number") or "").strip()
                                r_name = str(r.get("name") or "").strip().lower()
                                exists = any(
                                    str(er.get("number") or "").strip() == r_num and
                                    str(er.get("name") or "").strip().lower() == r_name
                                    for er in ai_data_full["rooms"]
                                )
                                if not exists:
                                    recovered_candidates.append(r)
                            added_count = _append_geometry_safe_rooms(
                                ai_data_full, recovered_candidates
                            )
                            still_missed = _missing(
                                ai_data_full["rooms"], ai_data.get("rooms", [])
                            )
                            print(f"  AI: Retry safely recovered {len(missed) - len(still_missed)} room(s) "
                                  f"(added {added_count} geometry boxes), {len(still_missed)} still missing")
                            missed = still_missed

                if missed:
                    ai_data_full["unresolved_rooms"] = [dict(r) for r in missed]
                    print(f"  AI: {len(missed)} room label(s) remain unresolved; "
                          "not inventing geometry for them")

                ai_data = ai_data_full
                _save_cache(cache_key + "_full", ai_data)
                print(f"  AI: {len(ai_data.get('rooms', []))} rooms ({time.time() - t2b:.1f}s)")

    gpt4o_data = ai_data
    _promote_detached_loft_candidate(gpt4o_data)
    _normalize_explicit_loft_access(gpt4o_data)
    # Final stamp — rooms added during reconciliation (recovery / empty-region
    # placement) inherit the panel floor too.
    _apply_panel_floor(gpt4o_data)

    # Step 4: YOLO model (if enabled) — provides room geometry that
    # merge_results combines with GPT-4o labels. Falls back to pure
    # AI-primary branch if YOLO is off or returns no usable rooms.
    # `image` is the ORIGINAL loaded image (pre-preprocessing). YOLO uses
    # the auto-rotated original to keep small features intact, then
    # translates boxes back into the preprocessed-sketch coord system.
    import config as _cfg
    yolo_rooms: List[Room] = []
    if getattr(_cfg, 'USE_MODEL', False) and not no_model:
        t_yolo = time.time()
        try:
            yolo_rooms = yolo_detect_rooms(sketch, original=image, model_path=model_path, project_number=project_number)
        except Exception as e:
            print(f"  YOLO: exception {e!r} — falling back to GPT-4o only")
            yolo_rooms = []
        print(f"  YOLO: {len(yolo_rooms)} rooms ({time.time() - t_yolo:.1f}s)")
    elif no_model:
        print("  YOLO: skipped (--no-model)")

    _correct_samples_by_color(sketch, gpt4o_data)
    plan = merge_results(yolo_rooms, gpt4o_data, sketch_h, sketch_w, seg_mask=None)
    plan.project_number = project_number
    plan.doors = []
    plan.windows = []
    plan.stairs = []
    if not plan.rooms:
        raise RuntimeError("Plan generation produced no rooms; refusing to export an empty VSDX")

    # Estimate pixel scale (meters per pixel) from building footprint
    # Typical UK residential: ~10-15m wide. Commercial: ~20-40m.
    # Use room count as heuristic: more rooms = larger building
    plan.pixel_scale = _estimate_pixel_scale(plan.rooms, sketch_w, sketch_h)

    acm_count = sum(1 for r in plan.rooms if r.has_acm)
    na_count = sum(1 for r in plan.rooms if r.no_access)
    print(f"\n  FINAL: {len(plan.rooms)} rooms ({acm_count} ACM, {na_count} no-access), "
          f"{len(plan.samples)} samples")
    for i, room in enumerate(plan.rooms):
        tags = []
        if room.has_acm: tags.append("ACM")
        if room.no_access: tags.append("NO-ACCESS")
        tag_str = f" [{','.join(tags)}]" if tags else ""
        print(f"    {room.number: >3} {room.label}{tag_str}")

    # Step 5: Export. Faithful overlay is the default because the detected
    # bounding boxes are approximate and cannot prove wall topology.
    t3 = time.time()
    use_overlay = _resolve_overlay_mode(overlay)
    # Safety net: in DEFAULT mode (overlay not explicitly chosen), if the
    # reconstructed rooms are fragmented into disconnected clusters (e.g. a dense
    # multi-floor sketch whose floor detection collapsed), fall back to a faithful
    # overlay rather than shipping a scattered vector plan. Explicit --vector /
    # --overlay are respected.
    if overlay is None and not use_overlay and _rooms_are_fragmented(plan):
        print("  [LAYOUT] Reconstructed rooms are fragmented (disconnected "
              "clusters) — falling back to source-faithful overlay")
        use_overlay = True
    print(f"  OUTPUT MODE: {'overlay (source-faithful)' if use_overlay else 'vector (reconstructed)'}")
    if use_overlay:
        vsdx_path = export_visio_overlay(plan, output_path, sketch)
    else:
        if not _vector_plan_geometry_is_usable(plan):
            raise RuntimeError(
                "Reconstructed room geometry failed quality checks; "
                "refusing to export a misleading vector plan"
            )
        vsdx_path = export_visio(plan, output_path)
    export_time = time.time() - t3

    total_time = time.time() - t0
    plan.detection_time = total_time
    ai_time_local = locals().get('ai_time', 0.0)
    print(f"\n  OUTPUT: {vsdx_path}")
    print(f"  TIME: {total_time:.1f}s (ai={ai_time_local:.1f}s, export={export_time:.1f}s)")

    return vsdx_path, plan
